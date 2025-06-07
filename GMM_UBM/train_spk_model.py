import numpy as np
import os
import joblib


# 利用 MAP自适应从UBM中
# 学习说话人GMM
def GMM_MAP(ubm_model, data):
    # 获取特征维度
    xdim = data.shape[1]
    T = data.shape[0]

    # 获取UBM参数
    M = ubm_model.n_components
    ubm_weights = ubm_model.weights_
    ubm_means = ubm_model.means_
    ubm_covars = ubm_model.covariances_

    # 计算特征在每个高斯成分上的概率
    posterior_prob = ubm_model.predict_proba(data)
    pr_i_xt = (ubm_weights * posterior_prob) / np.asmatrix(np.sum(ubm_weights \
                                                                  * posterior_prob, axis=1)).T
    # 0阶统计量
    n_i = np.asarray(np.sum(pr_i_xt, axis=0)).flatten()  # [M, ]

    # 1阶统计量
    E_x = np.asarray([(np.asarray(pr_i_xt[:, i]) * data).sum(axis=0) / n_i[i] for i in range(M)])  # [M x xdim]

    # 2阶统计量
    E_x2 = np.asarray([(np.asarray(pr_i_xt[:, i]) * (data ** 2)).sum(axis=0) / n_i[i] for i in range(M)])  # [M x xdim]

    # 计算融合参数
    relevance_factor = 16
    scaleparam = 1
    alpha_i = n_i / (n_i + relevance_factor)

    # 计算 GMM的参数
    new_weights = (alpha_i * n_i / T + (1.0 - alpha_i) * ubm_weights) * scaleparam
    alpha_i = np.asarray(np.asmatrix(alpha_i).T)
    new_means = (alpha_i * E_x + (1. - alpha_i) * ubm_means)
    new_covars = alpha_i * E_x2 + (1. - alpha_i) * (ubm_covars + (ubm_means ** 2)) - (new_means ** 2)

    # 返回 GMM
    ubm_model.means_ = new_means
    ubm_model.weights_ = new_weights
    ubm_model.covariances_ = new_covars

    return ubm_model


if __name__ == "__main__":

    model_path = 'models'

    file_lines = np.loadtxt("enrollment.scp", dtype='str', delimiter=" ")
    spks = file_lines[:, 1]
    utts = file_lines[:, 2]

    path_fea = 'fea/TEST'
    path_model = 'models'

    unique_spks = np.unique(spks)

    for spk in unique_spks:
        index = np.where(spks == spk)[0]
        datas = []

        for i in index:
            utt = utts[i]
            data = np.load(os.path.join(path_fea, spk + "_" + utt + ".npy"))
            datas.append(data)
        datas = np.concatenate(datas, axis=1).T
        ubm = joblib.load(os.path.join(model_path, 'ubm.model'))
        gmm = GMM_MAP(ubm, datas)

        joblib.dump(gmm, os.path.join(model_path, spk + '.model'))
        print("save model of spk:", spk)













