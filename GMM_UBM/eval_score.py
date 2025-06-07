import numpy as np
import os
import joblib
import sklearn


def getscore(ubm, gmm, data):
    score_ubm = ubm.score(data)
    score_gmm = gmm.score(data)
    return score_gmm - score_ubm


def compute_eer(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y_true=label, y_score=pred,pos_label= positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer, eer_threshold


if __name__ == "__main__":

    # 加载UBM
    path_model = 'models'
    ubm = joblib.load(os.path.join(path_model, 'ubm.model'))

    # 加载验证数据
    paht_fea = 'fea/TEST'
    file_lines = np.loadtxt("var.scp", dtype='str', delimiter=" ")
    spks_true = file_lines[:, 1]
    utts = file_lines[:, 2]
    spks_var = file_lines[:, 3]
    labs = file_lines[:, 4]
    labs = [int(lab) for lab in labs]
    scores = []
    for spk_ture, utt, spk_var, lab in zip(spks_true, utts, spks_var, labs):
        file_fea = os.path.join(paht_fea, spk_ture + '_' + utt + '.npy')
        data = np.load(file_fea).T

        gmm = joblib.load(os.path.join(path_model, spk_var + '.model'))
        score = getscore(ubm, gmm, data)
        scores.append(score)
        print(spk_ture, ' ', spk_var, ' ', "%.3f" % (score))

    eer, thred = compute_eer(labs, scores, positive_label=1)
    print(eer)
    print(thred)


