# feature_extractor.py

import numpy as np
import librosa
import librosa.display 
import traceback

# --- 特征提取函数 ---
def extract_features(audio_data, samplerate,
                     n_mfcc=19,
                     n_fft=512,
                     hop_length=160,
                     win_length=320,
                     n_mels=20
                    ):
    """
    从内存中的音频数据提取 MFCC、Delta 和 Delta-Delta 特征。
    Args:
        audio_data (np.ndarray): 输入音频数据 (NumPy 数组)。
        samplerate (int): 音频的采样率。
        n_mfcc (int): 要提取的 MFCC 数量。
        n_fft (int): FFT 窗口大小。
        hop_length (int): 帧之间的步长。
        win_length (int): 分析窗口大小。
        n_mels (int): 梅尔滤波器组的数量。
    Returns:
        np.ndarray: 拼接后的特征向量 (MFCCs + Deltas + Delta-Deltas)。
                    如果输入数据无效，则返回空数组。
    """
    if audio_data is None or (isinstance(audio_data, np.ndarray) and audio_data.size == 0):
        print("没有有效的音频数据进行特征提取。")
        return np.array([])

    if audio_data.dtype in [np.int16, np.int32, np.int64]:
         iinfo = np.iinfo(audio_data.dtype)
         audio_data = audio_data.astype(np.float32) / max(abs(iinfo.min), abs(iinfo.max))
    elif audio_data.dtype != np.float32 and np.issubdtype(audio_data.dtype, np.floating):
         audio_data = audio_data.astype(np.float32)

    if audio_data.ndim > 1:
        print(f"警告: 特征提取输入数据是多声道 ({audio_data.shape})，只取第一个声道。")
        audio_data = audio_data[:, 0]


    print(f"开始提取特征，数据形状: {audio_data.shape}, 采样率: {samplerate}")
    # print(f"使用特征参数: n_mfcc={n_mfcc}, n_fft={n_fft}, hop_length={hop_length}, win_length={win_length}, n_mels={n_mels}") # Removed detailed print

    try:
        mfccs = librosa.feature.mfcc(y=audio_data,
                                     sr=samplerate,
                                     n_mfcc=n_mfcc,
                                     n_fft=n_fft,
                                     hop_length=hop_length,
                                     win_length=win_length,
                                     n_mels=n_mels)
        mfccs = librosa.util.normalize(mfccs)

        delta_mfccs = librosa.feature.delta(mfccs, order=1)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=0)
        features = features.T

        print(f"特征提取完成，特征形状: {features.shape}")
        return features

    except Exception as e:
        print(f"特征提取失败: {e}")
        print(f"错误信息: {e}") # Simplified error output
        return np.array([])