# speaker_id.py

import numpy as np
import joblib # 用于加载模型
import os
import traceback
import sys

try:
    from sklearn.mixture import GaussianMixture as GMM
except ImportError:
    print("Error: scikit-learn is not installed. Please install it using 'pip install scikit-learn'.")
    # Exit handled in main app if essential component fails to load
    # sys.exit(1)

class SpeakerIdentifier:
    def __init__(self, model_dir, ubm_model_file, user_models_files, identification_threshold):
        """
        初始化声纹识别器，加载模型。

        Args:
            model_dir (str): 模型文件所在的目录。
            ubm_model_file (str): UBM 模型的文件名。
            user_models_files (dict): 字典，键是用户ID (str)，值是该用户 GMM 模型的文件名 (str)。
            identification_threshold (float): 用于判定的得分阈值。
        """
        self.model_dir = model_dir
        self.ubm_model_file = ubm_model_file
        self.user_models_files = user_models_files
        self.identification_threshold = identification_threshold
        self.ubm_model = None
        self.user_models = {}
        self.users = list(user_models_files.keys())
        self.min_frames_for_inference = 10

        self._load_models()

    def _load_models(self):
        """
        加载 UBM 模型和所有用户的 GMM 模型。
        使用 joblib.load 加载 sklearn 模型。
        """
        print("开始加载声纹识别模型...")
        ubm_path = os.path.join(self.model_dir, self.ubm_model_file)
        try:
            self.ubm_model = joblib.load(ubm_path)
            print(f"成功加载 UBM 模型: {ubm_path}")
            if not isinstance(self.ubm_model, GMM):
                print(f"警告: 加载的 UBM 模型对象类型不是 sklearn GaussianMixture: {type(self.ubm_model)}")
        except FileNotFoundError:
            print(f"错误: 未找到 UBM 模型文件: {ubm_path}")
            self.ubm_model = None
        except Exception as e:
            print(f"加载 UBM 模型失败: {ubm_path} - {e}")
            print(f"错误信息: {e}") # Simplified error output
            self.ubm_model = None

        for user_id, model_file in self.user_models_files.items():
            model_path = os.path.join(self.model_dir, model_file)
            try:
                self.user_models[user_id] = joblib.load(model_path)
                print(f"成功加载用户 {user_id} 的模型: {model_path}")
                if not isinstance(self.user_models[user_id], GMM):
                     print(f"警告: 加载的用户 {user_id} 模型对象类型不是 sklearn GaussianMixture: {type(self.user_models[user_id])}")
            except FileNotFoundError:
                print(f"错误: 未找到用户 {user_id} 的模型文件: {model_path}")
                self.user_models[user_id] = None
            except Exception as e:
                print(f"加载用户 {user_id} 的模型失败: {model_path} - {e}")
                print(f"错误信息: {e}") # Simplified error output
                self.user_models[user_id] = None

        if self.ubm_model is None or not any(self.user_models.values()):
             print("警告: 并非所有必需的模型都加载成功，识别功能可能受限或无法使用。")


    def _calculate_gmm_score(self, features, model):
        """
        计算特征向量在给定 GMM/UBM 模型下的平均对数似然得分。
        使用 sklearn GMM 对象的 .score(features) 方法。
        """
        if model is None:
            return -float('inf')

        if features is None or features.size == 0:
             return -float('inf')

        if not isinstance(features, np.ndarray) or features.ndim != 2:
             print(f"错误: 输入 _calculate_gmm_score 的 features 不是二维 numpy 数组，形状为 {features.shape if isinstance(features, np.ndarray) else 'N/A'}")
             return -float('inf')

        try:
            score = model.score(features)
            return score
        except AttributeError:
            print("错误: 加载的模型对象没有 .score() 方法。请检查您的模型类型和加载方式是否正确 (应为 sklearn GMM)。")
            return -float('inf')
        except Exception as e:
            print(f"计算 GMM 得分失败: {e}")
            print(f"错误信息: {e}") # Simplified error output
            return -float('inf')


    def identify_speaker(self, features):
        """
        对提取的特征进行声纹识别推理。
        """
        if self.ubm_model is None or not any(self.user_models.values()):
            print("模型未完全加载，无法进行识别。")
            return "模型未加载"

        if features is None or features.size == 0:
            return "特征为空"

        if features.shape[0] < self.min_frames_for_inference:
            print(f"警告: 特征帧数不足 ({features.shape[0]} 帧)，需要至少 {self.min_frames_for_inference} 帧进行推理。")
            return "特征太短"

        print(f"\n开始进行声纹识别推理...")

        score_ubm = self._calculate_gmm_score(features, self.ubm_model)
        if score_ubm == -float('inf'):
             print("计算 UBM 得分失败，推理中止。")
             return "推理失败"

        score_diffs = {}
        for user_id, model in self.user_models.items():
            if model is None:
                 print(f"跳过用户 {user_id}，因为模型未加载或加载失败。")
                 continue

            score_gmm = self._calculate_gmm_score(features, model)
            if score_gmm == -float('inf'):
                 print(f"计算用户 {user_id} GMM 得分失败。")
                 score_diffs[user_id] = -float('inf')
                 continue

            score_diff = score_gmm - score_ubm
            score_diffs[user_id] = score_diff
            print(f"用户 {user_id} 得分差 (GMM - UBM): {score_diff:.4f}")

        if not score_diffs or all(score == -float('inf') for score in score_diffs.values()):
            print("所有用户得分差计算失败或无效，无法进行比较。")
            return "无有效得分"

        valid_scores = {uid: score for uid, score in score_diffs.items() if score > -float('inf')}
        if not valid_scores:
             print("所有用户得分差无效。")
             return "无有效得分"

        highest_user = max(valid_scores, key=valid_scores.get)
        highest_score_diff = valid_scores[highest_user]

        print(f"最高得分差属于用户 {highest_user}: {highest_score_diff:.4f}")
        print(f"识别阈值: {self.identification_threshold:.4f}")

        if highest_score_diff > self.identification_threshold:
            print(f"判定结果: 用户 {highest_user} (得分差高于阈值)")
            return highest_user
        else:
            print(f"判定结果: 未知用户 (最高得分差低于阈值)")
            return "未知用户"