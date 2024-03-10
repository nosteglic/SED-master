import dataclasses
import glob
import logging
import math
import os
import pickle
import random
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import wandb
from dcase_util.data import ProbabilityEncoder
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from baseline_utils import ramps
from baseline_utils.ManyHotEncoder import ManyHotEncoder
from baseline_utils.utils import AverageMeter, EarlyStopping, get_durations_df
from evaluation_measures import ConfusionMatrix, compute_metrics
from post_processing import PostProcess
from swtich_bg import real_pseudo_strong_labeling, get_bgs_idx, switch_bg

logging.basicConfig(level=logging.INFO)


def cycle_iteration(iterable):
    while True:
        for i in iterable:
            yield i

def create_bg_pool(bg_pool, data, domain_data):
    for i in range(len(data)):  # background segments from weak set
        if len(domain_data[i]) != 0:
            for j in range(len(domain_data[i])):
                _seg_start_idx = domain_data[i][j][0]
                _seg_end_idx = domain_data[i][j][1]
                bg_pool.append(data[i, :, _seg_start_idx:_seg_end_idx])
    return bg_pool


@dataclasses.dataclass
class MTBGTrainerOptions:
    accum_grad: int = 1
    grad_clip: float = 5
    log_interval: Optional[int] = 250
    train_steps: int = 20000
    use_mixup: bool = True
    rampup_length: int = 15000
    consistency_cost: float = 2.0
    binarization_type: str = "global_threshold"
    threshold: float = 0.5
    early_stopping: bool = True
    patience: int = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _set_validation_options(
        self,
        valid_meta: str,
        valid_audio_dir: str,
        max_len_seconds: float,
        sample_rate: int,
        hop_size: int,
        pooling_time_ratio: int = 1,
    ):
        self.valid_meta = valid_meta
        self.valid_audio_dir = valid_audio_dir
        self.max_len_seconds = max_len_seconds
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.pooling_time_ratio = pooling_time_ratio

        self.validation_df = pd.read_csv(valid_meta, header=0, sep="\t")
        self.durations_validation = get_durations_df(valid_meta, valid_audio_dir)
        self.classes = self.validation_df.event_label.dropna().sort_values().unique()
        max_frames = math.ceil(self.max_len_seconds * self.sample_rate / self.hop_size)
        self.many_hot_encoder = ManyHotEncoder(labels=self.classes, n_frames=max_frames)
        self.decoder = self.many_hot_encoder.decode_strong
        self.classification_criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.consistency_criterion = torch.nn.MSELoss().to(self.device)


class MTBGTrainer(object):
    def __init__(
        self,
        model,
        ema_model,
        loader_train_sync,
        loader_train_real_weak,
        loader_train_real_unlabel,
        loader_valid,
        optimizer,
        scheduler,
        exp_path,
        bds_options,
        pretrained=None,
        resume=None,
        trainer_options=None,
    ):
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.ema_model = ema_model.cuda() if torch.cuda.is_available() else ema_model
        for param in self.ema_model.parameters():
            param.detach_()

        self.bds_options = bds_options
        self.min_frames = self.bds_options["min_frames"]

        self.best_score = 0
        self.best_psds = 0
        self.best_val_loss = np.inf
        self.forward_count = 0
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.options = trainer_options

        if pretrained is not None:
            logging.info(f"load pretrained model: {pretrained}")
            self.load(pretrained, pretrained)

        if resume is not None:
            logging.info(f"Resuming from {resume}")
            self.resume(exp_path / "model" / resume)

        self.exp_path = exp_path

        # 分类损失
        self.classification_criterion = trainer_options.classification_criterion
        self.loss_cls = lambda pred, target: self.classification_criterion(pred, target)
        # 一致性损失
        self.consistency_criterion = trainer_options.consistency_criterion
        self.loss_con = lambda student_output, teacher_output: self.consistency_criterion(
            torch.sigmoid(student_output), torch.sigmoid(teacher_output)
        )
        self.max_consistency_cost = trainer_options.consistency_cost

        self.accum_grad = trainer_options.accum_grad
        self.grad_clip_threshold = trainer_options.grad_clip
        self.device = trainer_options.device

        self.train_iter_sync = cycle_iteration(loader_train_sync)
        self.train_iter_real_weak = cycle_iteration(loader_train_real_weak)
        self.train_iter_real_unlabel = cycle_iteration(loader_train_real_unlabel)
        self.valid_loader = loader_valid


        self.losses_cls_strong = AverageMeter()
        self.losses_cls_weak = AverageMeter()
        self.losses_con_strong = AverageMeter()
        self.losses_con_weak = AverageMeter()
        self.losses_valid_strong = AverageMeter()
        self.losses_valid_weak = AverageMeter()

        if self.options.early_stopping:
            self.early_stopping_call = EarlyStopping(trainer_options.patience, val_comp="sup")

    def run(self):
        for i in tqdm(range(self.forward_count + 1, self.options.train_steps + 1)):
            self.train_one_step()

            if i % self.options.log_interval == 0:
                metrics = self.validation()
                val_loss = metrics["loss_valid_strong"] + metrics["loss_valid_weak"]
                score = metrics["macro_f1_event"]
                psds = metrics["psds_macro_f1"]
                is_best_loss = val_loss < self.best_val_loss
                is_best_score = score > self.best_score
                is_best_psds = psds > self.best_psds
                if is_best_loss:
                    self.best_val_loss = val_loss
                if is_best_score:
                    self.best_score = score
                if is_best_psds:
                    self.best_psds = psds
                self.save_checkpoint((self.exp_path / "model" / f"{i}th_iterations.pth"),
                                     is_best_loss, is_best_score, is_best_psds)
                if self.options.early_stopping:
                    if self.early_stopping_call.apply(metrics["macro_f1_event"]):
                        logging.warning("Early stopping")
                        break

    def update_ema_variables(self, alpha):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.forward_count + 1), alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data.mul_(alpha).add_(param.data, alpha=(1 - alpha))

    def mixup(self, data, data_ema, target, alpha=0.2):
        """Mixup data augmentation
        https://arxiv.org/abs/1710.09412
        Apply the same parameter for data and data_ema since to obtain the same target
        Args:
            data: input tensor for the student model
            data_ema: input tensor for the teacher model
        """
        batch_size = data.size(0)
        c = np.random.beta(alpha, alpha)

        perm = torch.randperm(batch_size)

        mixed_data = c * data + (1 - c) * data[perm, :]
        mixed_data_ema = c * data_ema + (1 - c) * data_ema[perm, :]
        mixed_target = c * target + (1 - c) * target[perm, :]

        return mixed_data, mixed_data_ema, mixed_target

    def train_one_step(self) -> None:
        """ One iteration of a Mean Teacher model """
        self.forward_count += 1
        self.model.train()
        self.ema_model.train()

        '''
        load data
        '''
        sample_sync, sample_sync_ema, target_sync, ids_sync = next(self.train_iter_sync)
        # sample_sync - (12, 1, 625, 128)
        # sample_sync_ema - (12, 1, 625, 128)
        # target_sync - (12, 156, 10)

        sample_real_weak, sample_real_weak_ema, target_real_weak, ids_real_weak = next(self.train_iter_real_weak)
        sample_real_unlabel, sample_real_unlabel_ema, target_real_unlabel, ids_real_unlabel = next(self.train_iter_real_unlabel)

        if self.options.use_mixup and 0.5 > random.random():
            sample_sync, sample_sync_ema, target_sync = self.mixup(
                sample_sync, sample_sync_ema, target_sync
            )
            sample_real_weak, sample_real_weak_ema, target_real_weak = self.mixup(
                sample_real_weak, sample_real_weak_ema, target_real_weak
            )
            sample_real_unlabel, sample_real_unlabel_ema, target_real_unlabel = self.mixup(
                sample_real_unlabel, sample_real_unlabel_ema, target_real_unlabel
            )

        sample_sync, sample_sync_ema = (
            sample_sync.to(self.device),
            sample_sync_ema.to(self.device),
        )
        sample_real_weak, sample_real_weak_ema = (
            sample_real_weak.to(self.device),
            sample_real_weak_ema.to(self.device)
        )
        sample_real_unlabel, sample_real_unlabel_ema = (
            sample_real_unlabel.to(self.device),
            sample_real_unlabel_ema.to(self.device)
        )
        target_sync = target_sync.to(self.device)
        target_real_weak = target_real_weak.max(dim=1)[0].to(self.device)

        rampup_value = ramps.exp_rampup(self.forward_count, self.options.rampup_length)
        consistency_cost = self.max_consistency_cost * rampup_value

        if self.forward_count > (self.options.train_steps * 0.6):
            # 交换背景
            with torch.no_grad():
                weak_pseudo_strong_label = real_pseudo_strong_labeling(sample_real_weak, self.model,
                                                                       **self.bds_options["labeling"])
                unlabel_pseudo_strong_label = real_pseudo_strong_labeling(sample_real_unlabel, self.model,
                                                                          **self.bds_options["labeling"])
                sync_strong_label = target_sync.cpu().detach().numpy().astype('int')
                bg_idx_sync = get_bgs_idx(sync_strong_label, min_frames=self.min_frames)
                bg_idx_weak = get_bgs_idx(weak_pseudo_strong_label, min_frames=self.min_frames)
                bg_idx_unlabel = get_bgs_idx(unlabel_pseudo_strong_label, min_frames=self.min_frames)
                bg_pool = []
                bg_pool = create_bg_pool(bg_pool=bg_pool, data=sample_real_weak, domain_data=bg_idx_weak)
                bg_pool = create_bg_pool(bg_pool=bg_pool, data=sample_real_unlabel, domain_data=bg_idx_unlabel)

                if len(bg_pool) != 0:
                    sample_sync = switch_bg(sample_sync, bg_idx_sync, bg_pool)

        # Outputs
        with torch.no_grad():
            output_ema_sync = self.ema_model(sample_sync)
            output_ema_weak = self.ema_model(sample_real_weak)
            output_ema_unlabel = self.ema_model(sample_real_unlabel)

        output_sync = self.model(sample_sync)
        output_weak = self.model(sample_real_weak)
        output_unlabel = self.model(sample_real_unlabel)

        # compute classification loss
        loss_cls_strong = self.loss_cls(output_sync["strong"], target_sync)
        loss_cls_weak = (
            self.loss_cls(output_weak["weak"], target_real_weak) +
            self.loss_cls(output_sync["weak"], target_sync.max(dim=1)[0])
        ) / 2

        # compute consistency loss
        loss_con_strong = consistency_cost * (
            self.loss_con(output_sync["strong"], output_ema_sync["strong"]) +
            self.loss_con(output_weak["strong"], output_ema_weak["strong"]) +
            self.loss_con(output_unlabel["strong"], output_ema_unlabel["strong"])
        ) / 3
        loss_con_weak = consistency_cost * (
            self.loss_con(output_sync["weak"], output_ema_sync["weak"]) +
            self.loss_con(output_weak["weak"], output_ema_weak["weak"]) +
            self.loss_con(output_unlabel["weak"], output_ema_unlabel["weak"])
        ) / 3

        loss_total = (loss_cls_weak + loss_cls_strong + loss_con_weak + loss_con_strong) / self.accum_grad
        loss_total.backward()  # Backprop
        loss_total.detach()  # Truncate the graph

        self.losses_cls_strong.update(loss_cls_strong.item())
        self.losses_cls_weak.update(loss_cls_weak.item())
        self.losses_con_strong.update(loss_con_strong.item())
        self.losses_con_weak.update(loss_con_weak.item())

        if self.forward_count % self.options.log_interval == 0:
            logging.info("After {} iteration".format(self.forward_count))
            logging.info("\t loss cls strong: {}".format(loss_cls_strong.item()))
            logging.info("\t loss cls weak: {}".format(loss_cls_weak.item()))
            logging.info("\t loss con strong: {}".format(loss_con_strong.item()))
            logging.info("\t loss con weak: {}".format(loss_con_weak.item()))

            self.losses_cls_strong.reset()
            self.losses_cls_weak.reset()
            self.losses_con_strong.reset()
            self.losses_con_weak.reset()

        if self.forward_count % self.accum_grad != 0:
            return
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_threshold)
        if math.isnan(grad_norm):
            logging.warning("grad norm is nan. Do not update model.")
        else:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            self.update_ema_variables(alpha=0.999)

    @torch.no_grad()
    def validation(self, post_processing=None) -> None:
        self.losses_valid_strong.reset()
        self.losses_valid_weak.reset()

        prediction_df = pd.DataFrame()
        threshold = self.options.threshold
        decoder = self.options.decoder
        binarization_type = self.options.binarization_type
        post_processing = None

        ptr = self.options.pooling_time_ratio

        # Frame level measure
        frame_measure = [ConfusionMatrix() for i in range(len(self.options.classes))]
        tag_measure = ConfusionMatrix()

        self.model.eval()

        for (data, target, data_ids) in self.valid_loader:
            data, target = data.to(self.device), target.to(self.device)
            predicts = self.model(data)

            # compute classification loss
            loss_cls_strong = self.loss_cls(predicts["strong"], target)
            loss_cls_weak = self.loss_cls(predicts["weak"], target.max(dim=1)[0])
            self.losses_valid_strong.update(loss_cls_strong.item())
            self.losses_valid_weak.update(loss_cls_weak.item())

            predicts["strong"] = torch.sigmoid(predicts["strong"]).cpu().data.numpy()
            predicts["weak"] = torch.sigmoid(predicts["weak"]).cpu().data.numpy()

            if binarization_type == "class_threshold":
                for i in range(predicts["strong"].shape[0]):
                    predicts["strong"][i] = ProbabilityEncoder().binarization(
                        predicts["strong"][i],
                        binarization_type=binarization_type,
                        threshold=threshold,
                        time_axis=0,
                    )
            else:
                predicts["strong"] = ProbabilityEncoder().binarization(
                    predicts["strong"],
                    binarization_type=binarization_type,
                    threshold=threshold,
                )
                predicts["weak"] = ProbabilityEncoder().binarization(
                    predicts["weak"],
                    binarization_type=binarization_type,
                    threshold=threshold,
                )

            # For debug, frame level measure
            for i in range(len(predicts["strong"])):
                target_np = target.cpu().numpy()
                tn, fp, fn, tp = confusion_matrix(target_np[i].max(axis=0), predicts["weak"][i], labels=[0, 1]).ravel()
                tag_measure.add_cf(tn, fp, fn, tp)
                for j in range(len(self.options.classes)):
                    tn, fp, fn, tp = confusion_matrix(
                        target_np[i][:, j], predicts["strong"][i][:, j], labels=[0, 1]
                    ).ravel()
                    frame_measure[j].add_cf(tn, fp, fn, tp)

            if post_processing is not None:
                for i in range(predicts["strong"].shape[0]):
                    for post_process_fn in post_processing:
                        predicts["strong"][i] = post_process_fn(predicts["strong"][i])

            for pred, data_id in zip(predicts["strong"], data_ids):
                pred = decoder(pred)
                pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])

                # Put them in seconds
                pred.loc[:, ["onset", "offset"]] *= ptr / (self.options.sample_rate / self.options.hop_size)
                pred.loc[:, ["onset", "offset"]] = pred[["onset", "offset"]].clip(0, self.options.max_len_seconds)

                pred["filename"] = data_id
                prediction_df = prediction_df.append(pred)

            # save predictions
            prediction_df.to_csv(
                self.exp_path / "predictions" / f"{self.forward_count}th_iterations.csv",
                index=False,
                sep="\t",
                float_format="%.3f",
            )

            # Compute evaluation metrics
            events_metric, segments_metric, psds_macro_f1 = compute_metrics(
                prediction_df,
                self.options.validation_df,
                self.options.durations_validation,
            )
            macro_f1_event = events_metric.results_class_wise_average_metrics()["f_measure"]["f_measure"]
            macro_f1_segment = segments_metric.results_class_wise_average_metrics()["f_measure"]["f_measure"]

            # Compute frame level macro f1 score
            ave_precision = 0
            ave_recall = 0
            macro_f1 = 0
            for i in range(len(self.options.classes)):
                ave_precision_, ave_recall_, macro_f1_ = frame_measure[i].calc_f1()
                ave_precision += ave_precision_
                ave_recall += ave_recall_
                macro_f1 += macro_f1_
            ave_precision /= len(self.options.classes)
            ave_recall /= len(self.options.classes)
            macro_f1 /= len(self.options.classes)
            weak_f1 = tag_measure.calc_f1()[2]

            metrics = {
                "loss_valid_strong": self.losses_valid_strong.avg,
                "loss_valid_weak": self.losses_valid_weak.avg,
                "macro_f1_event": macro_f1_event,
                "macro_f1_segment": macro_f1_segment,
                "psds_macro_f1": psds_macro_f1,
                "frame_level_precision": ave_precision,
                "frame_level_recall": ave_recall,
                "frame_level_macro_f1": macro_f1,
                "weak_f1": weak_f1,
            }

            wandb.log(metrics, step=self.forward_count)

        return metrics

    @torch.no_grad()
    def optimize_post_processing(self):
        post_process = PostProcess(self.model, self.valid_loader, Path(self.exp_path), self.options)
        pp_params = post_process.tune_all()
        with open(self.exp_path / "post_process_params.pickle", "wb") as f:
            pickle.dump(pp_params, f)
        post_process.compute_psds()

    def save(self, filename, ema_filename):
        torch.save(self.model.state_dict(), str(filename))
        torch.save(self.ema_model.state_dict(), str(ema_filename))

    def save_checkpoint(self, filename: str, is_best_loss: float, is_best_score: float, is_best_psds: float) -> None:
        state = {
            "iteration": self.forward_count,
            "state_dict": self.model.state_dict(),
            "state_dict_ema": self.ema_model.state_dict(),
            "best_score": self.best_score,
            "best_val_loss": self.best_val_loss,
            "best_psds": self.best_psds,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(state, filename)
        if is_best_loss:
            best_path = os.path.join(os.path.dirname(filename), "model_best_loss.pth")
            shutil.copyfile(filename, best_path)
        if is_best_score:
            best_path = os.path.join(os.path.dirname(filename), "model_best_score.pth")
            shutil.copyfile(filename, best_path)
        if is_best_psds:
            best_path = os.path.join(os.path.dirname(filename), "model_best_psds.pth")
            shutil.copyfile(filename, best_path)

    def load(self, filename):
        logging.info(f"=> loading pretrained '{filename}'")
        checkpoint = torch.load(str(filename))
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        self.ema_model.load_state_dict(checkpoint["state_dict_ema"], strict=False)

    def resume(self, filename):
        logging.info(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(str(filename))
        self.forward_count = checkpoint["iteration"]
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        self.ema_model.load_state_dict(checkpoint["state_dict_ema"], strict=False)
        self.best_score = checkpoint["best_score"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        logging.info(f"=> loaded checkpoint {self.forward_count} iterations")

    def average_checkpoint(self, snapshots_dir, out="average.pth", num=10, last_from_best=True):
        snapshots = glob.glob(f"{snapshots_dir}/*.pth")
        last = sorted(snapshots, key=os.path.getmtime)
        if last_from_best:
            best_idx = last.index(f"{snapshots_dir}/model_best_score.pth")
            last = last[best_idx - num : best_idx]
        else:
            last = last[-num:]
        avg = None

        # sum
        for path in last:
            print(path)
            states = torch.load(path, map_location=torch.device("cpu"))["state_dict"]
            if avg is None:
                avg = states
            else:
                for k in avg.keys():
                    avg[k] += states[k]

        # average
        for k in avg.keys():
            if avg[k] is not None:
                avg[k] /= num

        torch.save(avg, os.path.join(snapshots_dir, out))