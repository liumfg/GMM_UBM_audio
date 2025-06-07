from platform import python_branch
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import os
import joblib
import pickle
path_fea = 'fea/Train'
file_lines = np.loadtxt('ubm_wav.scp',dtype='str',delimiter=' ')
datas_all = []

spks = file_lines[:,1]
utt_ids = file_lines[:,2]

for spk,utt in zip(spks,utt_ids):
    file_fea = os.path.join(path_fea,spk+"_"+utt+".npy")
    print("load fea",file_fea)
    data = np.load(file_fea)
    datas_all.append(data)

datas = np.concatenate(datas_all,axis=1).T
print(datas.shape)

# 构造UBM 模型
N_mix = 128
ubm =  GMM(n_components = N_mix, covariance_type='diag', max_iter=50)
ubm.fit(datas)
model_path = 'models'
os.makedirs(model_path,exist_ok=True)
joblib.dump(ubm, os.path.join(model_path,'ubm.model'))

