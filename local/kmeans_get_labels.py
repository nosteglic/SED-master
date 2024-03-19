import os

import librosa
from sklearn.cluster import KMeans
import joblib
import numpy as np
from tqdm import tqdm

def train_kmeans(n_clusters, n_init, data_path, save_path):
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=n_init,
        max_iter=100,
        tol=0.0
    )
    # kmeans = MiniBatchKMeans(
    #     n_clusters=100,
    #     init='k-means++',
    #     batch_size=10000,
    #     verbose=1,
    #     compute_labels=False,
    #     max_iter=100,
    #     max_no_improvement=100,
    #     init_size=None,
    #     tol=0.0,
    #     n_init=20,
    #     reassignment_ratio=0.0,
    # )
    print("----------train kmeans---------------")
    print(f"feature root: {data_path}")
    file_list = []
    for root, _, files in os.walk(data_path):
        for file in tqdm(files):
            file_path = os.path.join(root, file)
            file_data = np.load(file_path).astype(np.float32)
            # file_data = librosa.amplitude_to_db(file_data)
            mfcc = librosa.feature.mfcc(
                S=file_data.squeeze().T,
                n_mfcc=13
            )
            delta1 = librosa.feature.delta(mfcc, order=1)
            delta2 = librosa.feature.delta(mfcc, order=2)
            file_list.append(np.concatenate([mfcc, delta1, delta2], axis=0).transpose(1, 0))
    files = np.squeeze(np.concatenate(file_list, axis=-2))
    print("start kmeans fit...")
    kmeans.fit(files)
    joblib.dump(kmeans, save_path)
    print(f"{save_path} Kmeans fit Done.")
    return kmeans

def get_labels(km_path, data_path):
    print("----------get kmeans labels---------------")
    print(f"feature root: {data_path}")
    print(f"kmeans root: {km_path}")
    if os.path.exists(km_path):
        km=joblib.load(km_path)
    else:
        raise ValueError("No such kmeans!")
    for root, _, files in os.walk(data_path):
        for file in tqdm(files):
            file_path = os.path.join(root, file)
            file_data = np.load(file_path).astype(np.float32)
            # file_data = librosa.amplitude_to_db(file_data)
            mfcc = librosa.feature.mfcc(
                S=file_data.squeeze().T,
                n_mfcc=13
            )
            delta1 = librosa.feature.delta(mfcc, order=1)
            delta2 = librosa.feature.delta(mfcc, order=2)
            data = np.concatenate([mfcc, delta1, delta2], axis=0)
            label = km.predict(data.T.astype(np.float32))
            np.save(file_path.replace("features", "labels"), label)
    print(f"{file_path.replace('features', 'labels')} labels Done.")

if __name__ == '__main__':
    feat_path = "/data2/syx/DCASE2021/features_clean"
    # save_path = "/data2/syx/DCASE2021/kmeans/mfcc_Conformer.pkl"
    # save_path = "/data2/syx/DCASE2021/kmeans/mfcc_CRNN.pkl"
    # save_path = "/data2/syx/DCASE2021/kmeans/mfcc_Conformer_ptr.pkl"
    # save_path = "/data2/syx/DCASE2021/kmeans/mfcc_CRNN_ptr.pkl"
    # data_path = "/data2/syx/DCASE2021/features_clean/sr16000_n_mels64_n_fft1024_hop_size323"
    # data_path = "/data2/syx/DCASE2021/features_clean/sr16000_n_mels128_n_fft2048_hop_size256"
    # data_path = "/data2/syx/DCASE2021/features_clean/sr16000_n_mels64_n_fft1024_hop_size323_ptr"
    # data_path = "/data2/syx/DCASE2021/features_clean/sr16000_n_mels128_n_fft2048_hop_size256_ptr"
    # km = train_kmeans(n_clusters=50, n_init=10, save_path=save_path, data_path=data_path)

    # get_labels(km_path=save_path, data_path=data_path)

    for root, dirs, _ in os.walk(feat_path):
        root_pardir = os.path.abspath(os.path.join(feat_path, os.pardir))
        for dir in dirs:
            data_path = os.path.join(root, dir)
            save_path = f"{root_pardir}/kmeans/mfcc"
            if "ptr" in dir:
                save_path += "_ptr"
            if "64" in dir:
                save_path += "_Conformer"
            else:
                save_path += "_CRNN"
            km = train_kmeans(n_clusters=50, n_init=10, save_path=save_path, data_path=data_path)
            get_labels(km_path=save_path, data_path=data_path)

    # temp = np.load("/data2/syx/DCASE2021/labels_clean/sr16000_n_mels64_n_fft1024_hop_size323_ptr/557.npy")
    # print(temp.shape)
