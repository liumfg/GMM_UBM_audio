import librosa
import os
import numpy as np

# 提取 UBM特征

fea_train_path = "fea/TRAIN"
os.makedirs(fea_train_path,exist_ok=True)

file_lines = np.loadtxt('ubm_wav.scp',dtype='str',delimiter=" ")
files= file_lines[:,0]
spk_ids = file_lines[:,1]
utt_ids = file_lines[:,2]

for file,spk,utt in zip(files,spk_ids,utt_ids):
    # 读取音频文件
    y,fs = librosa.load(file,sr=None, mono=True)

    # 进行MFCC特征的提取
    raw_mfcc = librosa.feature.mfcc(y=y,
                                   sr=fs,
                                   n_mfcc=19,
                                   n_fft=512,
                                   hop_length=160,
                                   win_length=320,
                                   n_mels=20
                                   )
    # 增加动态特征
    raw_mfcc = librosa.util.normalize(raw_mfcc)
    mfcc_delta = librosa.feature.delta(raw_mfcc)
    mfcc_delta2 = librosa.feature.delta(raw_mfcc, order=2)

    # 拼接生成最终的MFCC特征
    fea_mfcc = np.concatenate([raw_mfcc,mfcc_delta,mfcc_delta2],axis=0)

    # fea_mean = np.mean(fea_mfcc,axis=1,keepdims=True)
    # fea_std = np.std(fea_mfcc,axis=1,keepdims=True)
    # fea_mfcc = fea_mfcc-

    file_fea = os.path.join(fea_train_path,spk+"_"+utt+".npy")
    np.save(file=file_fea,arr=fea_mfcc)
    print("save_file ",file_fea)


# 提取 test数据特征

fea_train_path = "fea/TEST"
os.makedirs(fea_train_path,exist_ok=True)

file_lines = np.loadtxt('test.scp',dtype='str',delimiter=" ")
files= file_lines[:,0]
spk_ids = file_lines[:,1]
utt_ids = file_lines[:,2]

for file,spk,utt in zip(files,spk_ids,utt_ids):
    # 读取音频文件
    y,fs = librosa.load(file,sr=None, mono=True)

    # 进行MFCC特征的提取
    raw_mfcc = librosa.feature.mfcc(y=y,
                                   sr=fs,
                                   n_mfcc=19,
                                   n_fft=512,
                                   hop_length=160,
                                   win_length=320,
                                   n_mels=20
                                   )
    # 增加动态特征
    raw_mfcc = librosa.util.normalize(raw_mfcc)
    mfcc_delta = librosa.feature.delta(raw_mfcc)
    mfcc_delta2 = librosa.feature.delta(raw_mfcc, order=2)

    # 拼接生成最终的MFCC特征
    fea_mfcc = np.concatenate([raw_mfcc,mfcc_delta,mfcc_delta2],axis=0)
    file_fea = os.path.join(fea_train_path,spk+"_"+utt+".npy")
    np.save(file=file_fea,arr=fea_mfcc)
    print("save_file ",file_fea)
#
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')  # 可以尝试 'Qt5Agg' 等其他后端，如 'Qt5Agg' 适用于基于Qt的界面
#
# # 假设你已经提取了MFCC特征并保存为.npy文件
# # 这里以加载一个.npy文件为例
# file_path = "fea/TEST/wuwei_ww1.npy"
# fea_mfcc = np.load(file_path)
#
# # 只取前20个时间帧的数据（假设时间帧是沿着第一个维度）
# fea_mfcc = fea_mfcc[:, :10]
#
# # 可视化MFCC特征
# plt.figure(figsize=(10, 4))
# # 使用imshow函数绘制MFCC特征，通过extent参数指定范围
# extent = [0, 20, 0, fea_mfcc.shape[0]]  # 这里x轴范围是0到20（对应时间帧），y轴范围根据MFCC系数数量确定
# plt.imshow(fea_mfcc, aspect='auto', origin='lower', cmap='viridis', extent=extent)
# plt.colorbar(format='%+2.0f dB')
# plt.title('wuwei MFCC Features')
# plt.xlabel('Time Frames')
# plt.ylabel('MFCC Coefficients')
# plt.tight_layout()
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')  # 可以尝试 'Qt5Agg' 等其他后端，如 'Qt5Agg' 适用于基于Qt的界面
# import matplotlib.pyplot as plt
# # 假设你已经提取了MFCC特征并保存为.npy文件
# # 这里以加载一个.npy文件为例
# file_path = "fea/TEST/zhanglixuan_zly5.npy"
# fea_mfcc = np.load(file_path)
#
# # 可视化MFCC特征
# plt.figure(figsize=(10, 4))
# # 使用imshow函数绘制MFCC特征
# plt.imshow(fea_mfcc, aspect='auto', origin='lower', cmap='viridis')
# plt.colorbar(format='%+2.0f dB')
# plt.title('zhanglixuan MFCC Features')
# plt.xlabel('Time Frames')
# plt.ylabel('MFCC Coefficients')
# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')  # 可以尝试 'Qt5Agg' 等其他后端，如 'Qt5Agg' 适用于基于Qt的界面
# import matplotlib.pyplot as plt
# # 假设参数
# sample_rate = 16000
# frame_shift = 0.01
# frames_per_second = 1 / frame_shift
# num_frames_2s = int(2 * frames_per_second)
#
# file_path = "fea/TEST/zhanglixuan_zly6.npy"
# fea_mfcc = np.load(file_path)
# fea_mfcc_2s = fea_mfcc[:num_frames_2s, :]
#
# plt.figure(figsize=(10, 4))
# plt.imshow(fea_mfcc_2s.T, aspect='auto', origin='lower', cmap='viridis')
# plt.colorbar(format='%+2.0f dB')
# plt.title('liumingfang MFCC Features (First 2 seconds)')
# plt.xlabel('Time Frames')
# plt.ylabel('MFCC Coefficients')
# plt.tight_layout()
# plt.show()
#
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')  # 可以尝试 'Qt5Agg' 等其他后端，如 'Qt5Agg' 适用于基于Qt的界面
# import matplotlib.pyplot as plt
# # 加载MFCC特征
# file_path = "fea/TEST/zhanglixuan_zly6.npy"
# fea_mfcc = np.load(file_path)
#
# # 假设参数 (根据你的特征提取设置调整)
# sample_rate = 16000  # 采样率 Hz
# frame_length = 25  # 帧长 ms
# frame_shift = 10   # 帧移 ms
#
# # 计算每秒的帧数
# frames_per_second = 1000 / frame_shift  # 10ms帧移 = 100帧/秒
#
# # 计算前两秒的帧数
# frames_in_two_seconds = int(frames_per_second * 2)
#
# # 截取前两秒的特征
# fea_mfcc_2s = fea_mfcc[:frames_in_two_seconds, :]
#
# # 创建时间轴(秒)
# time_axis = np.arange(fea_mfcc_2s.shape[0]) * (frame_shift / 1000)
#
# # 可视化MFCC特征
# plt.figure(figsize=(10, 4))
# # 使用imshow函数绘制MFCC特征，转置矩阵使时间在x轴上
# plt.imshow(fea_mfcc_2s.T,
#            aspect='auto',
#            origin='lower',
#            cmap='viridis',
#            extent=[0, time_axis[-1], 0, fea_mfcc_2s.shape[1]])
# plt.colorbar(format='%+2.0f dB')
# plt.title('zhanglixuan MFCC Features (First 2 Seconds)')
# plt.xlabel('Time (s)')
# plt.ylabel('MFCC Coefficients')
# plt.tight_layout()
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
#
# # 假设你已经提取了MFCC特征并保存为.npy文件
# # 这里以加载一个.npy文件为例
# file_path = "fea/TEST/zhanglixuan_zly1.npy"
# fea_mfcc = np.load(file_path)
#
# # 假设参数
# sample_rate = 16000
# frame_shift = 0.01
# frames_per_second = 1 / frame_shift
# num_frames_2s = int(2 * frames_per_second)
#
# # 截取前两秒的MFCC特征
# fea_mfcc_2s = fea_mfcc[:num_frames_2s, :]
#
# # 可视化MFCC特征
# plt.figure(figsize=(10, 4))
# # 使用imshow函数绘制MFCC特征
# plt.imshow(fea_mfcc_2s, aspect='auto', origin='lower', cmap='viridis')
# plt.colorbar(format='%+2.0f dB')
# plt.title('zhanglixuan MFCC Features (First 2 seconds)')
# plt.xlabel('Time Frames')
# plt.ylabel('MFCC Coefficients')
# plt.tight_layout()
# plt.show()