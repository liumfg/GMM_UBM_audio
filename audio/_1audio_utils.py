# audio_utils.py

import sounddevice as sd
import numpy as np
import traceback
import sys
# --- 音频录制类 ---
class AudioRecorder:
    def __init__(self, samplerate=16000, channels=1, blocksize=1024):
        """
        初始化音频录制器。
        Args:
            samplerate (int): 采样率，默认为 16000 Hz。
            channels (int): 声道数，默认为 1 (单声道)。
            blocksize (int): 每次回调处理的音频帧数。
        """
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self._is_recording = False
        self._audio_data = [] # 存储录制的音频数据块
        self._stream = None
        print(f"初始化录音器，采样率: {self.samplerate}, 声道: {self.channels}")

    def _callback(self, indata, frames, time_info, status):
        """
        录音流的回调函数。每当有新的音频数据块可用时被调用。
        """
        if status:
            print(f"录音状态警告: {status}")
        if self._is_recording:
            if indata.dtype == np.float32 and indata.ndim == 2 and indata.shape[1] == self.channels:
                 self._audio_data.append(indata.copy())
            else:
                 print(f"警告: 录音回调接收到非预期的 indata 类型/形状: {indata.dtype}, {indata.shape}")
                 try:
                     converted_data = indata.astype(np.float32)
                     if converted_data.ndim > 1:
                          converted_data = converted_data[:, 0]
                     self._audio_data.append(converted_data[:, np.newaxis])
                 except Exception as e:
                      print(f"警告: 录音回调数据转换失败: {e}")


    def start_recording(self):
        """
        开始录音。
        """
        if self._is_recording:
            print("正在录音中...")
            return

        print("开始录音...")
        self._is_recording = True
        self._audio_data = []
        try:
            input_device_index = sd.default.device[0]
            device_info = sd.query_devices(input_device_index, 'input')
            print(f"使用输入设备: {device_info.get('name', '默认输入设备')}")

            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                blocksize=self.blocksize,
                callback=self._callback,
                device=input_device_index
            )
            self._stream.start()
            print(f"录音已开始 (流模式)，尝试使用采样率: {self.samplerate} Hz...")

        except Exception as e:
            print(f"启动录音失败: {e}")
            print(f"错误信息: {e}")
            self._is_recording = False
            self._stream = None


    def stop_recording(self):
        """
        停止录音并返回录制的音频数据和采样率。
        """
        if not self._is_recording:
            print("未在录音中...")
            return np.array([]), None

        print("停止录音...")
        self._is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            print("录音已停止，流已关闭。")

        if not self._audio_data:
            print("没有录制到音频数据。")
            return np.array([]), self.samplerate

        try:
            if self.channels > 1:
                recorded_data = np.vstack(self._audio_data)
            else:
                 recorded_data = np.concatenate(self._audio_data, axis=0)

            print(f"录制完成，数据形状: {recorded_data.shape}, 采样率: {self.samplerate}")
            return recorded_data, self.samplerate
        except Exception as e:
            print(f"拼接音频数据失败: {e}")
            print(f"错误信息: {e}")
            return np.array([]), self.samplerate

# --- 音频播放函数 ---
def play_audio(audio_data, samplerate):
    """
    播放音频数据。
    Args:
        audio_data (np.ndarray): 要播放的音频数据 (NumPy 数组)。
        samplerate (int): 音频的采样率。
    """
    if audio_data is None or audio_data.size == 0:
        print("没有音频数据可播放。")
        return

    if audio_data.dtype != np.float32 and np.issubdtype(audio_data.dtype, np.floating):
         audio_data = audio_data.astype(np.float32)

    if np.issubdtype(audio_data.dtype, np.integer):
         iinfo = np.iinfo(audio_data.dtype)
         audio_data = audio_data.astype(np.float32) / max(abs(iinfo.min), abs(iinfo.max))
         audio_data = np.clip(audio_data, -1.0, 1.0)

    if audio_data.ndim == 1:
        audio_data = audio_data[:, np.newaxis]
    elif audio_data.ndim > 2:
        print(f"警告: 播放音频数据维度大于2，形状为 {audio_data.shape}")

    print(f"开始播放音频，数据形状: {audio_data.shape}, 采样率: {samplerate}")
    try:
        sd.play(audio_data, samplerate)
        sd.wait()
        print("音频播放结束。")
    except Exception as e:
        print(f"播放音频失败: {e}")
        print(f"错误信息: {e}")
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.mixture import GaussianMixture
# import joblib
#
# # 加载训练好的GMM模型，这里假设加载UBM模型，你可以根据需要修改模型路径
# model_path = 'models/ubm.model'
# gmm = joblib.load(model_path)
#
# # 加载特征数据，这里假设你有一个示例特征文件
# feature_file = 'liumingfang_lmf1.npy'
# data = np.load(feature_file).T  # 转置数据以符合模型输入格式
#
# # 创建网格点
# x = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
# y = np.linspace(data[:, 1].min(), data[:, 1].max(), 100)
# X_grid, Y_grid = np.meshgrid(x, y)
# XX = np.array([X_grid.ravel(), Y_grid.ravel()]).T
#
# # 计算GMM在网格点上的概率密度
# Z = -gmm.score_samples(XX)
# Z = Z.reshape(X_grid.shape)
#
# # 绘制数据点和GMM的等高线图
# plt.scatter(data[:, 0], data[:, 1], s=10, c='b', alpha=0.5)
# CS = plt.contour(X_grid, Y_grid, Z, levels=np.logspace(0, 3, 20), cmap='viridis')
# plt.clabel(CS, inline=1, fontsize=10)
# plt.title('GMM Visualization')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()