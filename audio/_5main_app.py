# main

import sys
import time
import numpy as np # 用于处理音频数据 (numpy array)
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QTextEdit
from PySide6.QtCore import Qt, QThread, Signal, QObject, Slot # 导入 Slot 装饰器


from _1audio_utils import AudioRecorder, play_audio
from _2feature_extractor import extract_features
from _3speaker_id import SpeakerIdentifier
from _4baidu_api_client import BaiduAPIClient


# --- Configuration ---
MODEL_DIR = "./models" 

UBM_MODEL_FILE = "ubm.model"
USER_MODELS_FILES = {
    "liumingfang": "liumingfang.model",
    "wanghongyi": "wanghongyi.model",
    "wuwei": "wuwei.model",
    "zhanglixuan": "zhanglixuan.model",

}
IDENTIFICATION_THRESHOLD = 0.5 # 示例阈值

# --- 配置百度 API Key 和 Secret Key ---
ASR_TTS_API_KEY = "WBqiP1LlQ9XTktDjuu5CtcHz"
ASR_TTS_SECRET_KEY = "lYhmf87vnO2RlhOxKW171DayEydgCsXj"
LLM_API_KEY = "bce-v3/ALTAK-ob15ceNLd3CtHoYoi60yv/bb3bb09da19a074fbd764afa5e32240b56947a4b"

# --- 其他配置 ---
SAMPLE_RATE = 16000 # 采样率

# --- 工作线程类 ---
class Worker(QObject):
    finished = Signal() # 任务完成信号
    progress = Signal(str) # 状态更新信号，发送字符串信息
    speaker_identified = Signal(str) # 声纹识别结果信号
    asr_recognized = Signal(str) # ASR 识别结果信号
    tts_audio_ready = Signal(bytes) # TTS 合成音频数据信号 
    error_occurred = Signal(str) # 错误发生信号
    welcome_user = Signal(str) # 用于发送欢迎信息


    def __init__(self, samplerate, model_dir, ubm_model_file, user_models_files, identification_threshold, baidu_api_key, baidu_secret_key, llm_api_key, parent=None):
        super().__init__(parent)
        self._is_running = True
        self._is_recording_active = False
        self.samplerate = samplerate

        # 对话历史列表
        self.message_history = []
        self.message_history.append({"role": "system", "content": "你是一个有帮助的助手，请简洁明了地回答问题。"})


        try:
            self.recorder = AudioRecorder(samplerate=self.samplerate)
            self.speaker_identifier = SpeakerIdentifier(model_dir, ubm_model_file, user_models_files, identification_threshold)
            self.baidu_client = BaiduAPIClient(baidu_api_key,
                                               baidu_secret_key,
                                               llm_api_key)

            if (self.speaker_identifier.ubm_model is None or not any(self.speaker_identifier.user_models.values())) or self.baidu_client.get_asr_tts_access_token() is None:
                 self.error_occurred.emit("模型或API客户端初始化失败，语音功能受限。请检查模型文件和API Key。")
                 self._is_running = False

        except Exception as e:
            self.error_occurred.emit(f"工作线程初始化失败: {e}")
            self._is_running = False



    @Slot()
    def run(self):
        if not self._is_running:
             self.progress.emit("工作线程因初始化失败而停止。")
             self.finished.emit()
             return

        self.progress.emit("工作线程已启动，等待开始信号...")
        print("工作线程启动，等待开始信号...")

        self.thread().exec() 


    @Slot()
    def start_voice_processing(self):
        if not self._is_running:
            self.error_occurred.emit("工作线程未运行，无法开始处理。")
            return
        if self._is_recording_active:
            print("工作线程：已在录音中。")
            return

        self._is_recording_active = True
        self.progress.emit("开始录音...")
        print("工作线程：开始录音...")
        try:
            self.recorder.start_recording()
        except Exception as e:
            self.error_occurred.emit(f"启动录音失败: {e}")
            print(f"工作线程：启动录音失败: {e}")
            self._is_recording_active = False
            self.progress.emit("录音启动失败。")


    @Slot()
    def stop_recording_task(self):
        if not self._is_running:
             self.error_occurred.emit("工作线程未运行，无法停止处理。")
             return

        if not self._is_recording_active:
            print("工作线程：未在录音中，无法停止。")
            self.progress.emit("未在录音中。")
            return

        self._is_recording_active = False
        self.progress.emit("停止录音，处理中...")
        print("工作线程：停止录音...")

        recorded_audio_data, recorded_samplerate = None, None
        try:
            recorded_audio_data, recorded_samplerate = self.recorder.stop_recording()

            if recorded_audio_data is None or recorded_audio_data.size == 0:
                self.progress.emit("没有录制到有效音频数据。")
                print("工作线程：没有录制到有效音频数据。")
                return

            self.progress.emit("提取特征...")
            print("工作线程：提取特征...")
            features = extract_features(recorded_audio_data, recorded_samplerate)

            if features.size == 0:
                self.progress.emit("特征提取失败。")
                print("工作线程：特征提取失败。")
                return

            self.progress.emit("进行声纹识别...")
            print("工作线程：进行声纹识别...")
            recognition_result = self.speaker_identifier.identify_speaker(features)
            self.speaker_identified.emit(recognition_result)
            # self.progress.emit(f"声纹识别结果: {recognition_result}") # 这条信息由 speaker_identified 信号处理

            print(f"工作线程：声纹识别结果: {recognition_result}")

            if recognition_result in self.speaker_identifier.users:
                # self.progress.emit(f"欢迎回来，{recognition_result}！") # 这条信息由新增的 welcome_user 信号处理
                self.welcome_user.emit(recognition_result) # **发送欢迎信息信号**
                print(f"工作线程：欢迎回来，{recognition_result}！")

                if recorded_audio_data.dtype == np.float32:
                    audio_bytes = (np.clip(recorded_audio_data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                elif recorded_audio_data.dtype == np.int16:
                    audio_bytes = recorded_audio_data.tobytes()
                else:
                    print(f"警告: 未知音频数据类型 {recorded_audio_data.dtype}，尝试转换为 int16。")
                    audio_bytes = recorded_audio_data.astype(np.int16).tobytes()

                self.progress.emit("调用 ASR API...")
                print("工作线程：调用 ASR API...")
                recognized_text = self.baidu_client.asr(audio_bytes, audio_format="pcm", sample_rate=self.samplerate)

                if recognized_text:
                    self.asr_recognized.emit(recognized_text)
                    print(f"工作线程：识别文本: {recognized_text}")

                    # 将用户的识别文本添加到对话历史
                    self.message_history.append({"role": "user", "content": recognized_text})

                    self.progress.emit("进行问答...")
                    print("工作线程：进行问答...")
                    answer_text = self.baidu_client.chat_with_llm(self.message_history)

                    if answer_text:
                        
                        if not answer_text.startswith("LLM 调用失败:") and \
                           not answer_text.startswith("LLM API 错误:") and \
                           not answer_text.startswith("LLM API 响应格式未知.") and \
                           not answer_text.startswith("LLM 请求失败:"):
                            self.message_history.append({"role": "assistant", "content": answer_text})

                        self.progress.emit("调用 TTS API...")
                        print("工作线程：调用 TTS API...")
                        # TTS 合成的是 LLM 的回答文本（或错误信息）
                        tts_audio_bytes = self.baidu_client.tts(answer_text, speaker=4, audio_format="pcm", sample_rate=self.samplerate)
                        
                        if tts_audio_bytes:
                            self.progress.emit("TTS 合成完成，准备播放。")
                            print("工作线程：TTS 合成完成。")
                            try:
                                tts_audio_np = np.frombuffer(tts_audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                                self.progress.emit("播放回答语音...")
                                print("工作线程：播放回答语音...")
                                play_audio(tts_audio_np, self.samplerate)
                                self.progress.emit("播放完成。")
                                print("工作线程：播放完成。")
                            except Exception as e:
                                self.error_occurred.emit(f"播放 TTS 音频失败: {e}")
                                print(f"工作线程：播放 TTS 音频失败: {e}")
                        else:
                            self.progress.emit("TTS 合成回答失败。")
                            print("工作线程：TTS 合成回答失败。")
                    else:
                         self.progress.emit("问答逻辑没有生成回答。")
                         print("工作线程：问答逻辑没有生成回答。")

                else:
                    self.progress.emit("ASR 识别失败。")
                    print("工作线程：ASR 识别失败。")

            else:
                self.progress.emit("未识别到已知用户。")
                print("工作线程：未识别到已知用户。")

        except Exception as e:
            self.error_occurred.emit(f"语音处理流程中发生错误: {e}")
            print(f"工作线程：语音处理流程中发生错误: {e}")
        finally:
            self.progress.emit("处理流程结束。")
            print("工作线程：处理流程结束。")
            self.finished.emit()


# --- GUI 主窗口类 ---
class VoiceInteractionGUI(QMainWindow):

    start_processing_signal = Signal() #启动处理 (开始录音)
    stop_processing_signal = Signal() # 停止录音并处理
    quit_worker_signal = Signal() # GUI 发送信号请求 worker 的 run 方法退出循环 (如果run中有循环)


    def __init__(self):
        super().__init__()

        self.setWindowTitle("语音交互应用")
        self.setGeometry(100, 100, 500, 400)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.record_button = QPushButton("开始录音")
        layout.addWidget(self.record_button)

        self.status_label = QLabel("准备就绪")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        self.info_text_edit = QTextEdit()
        self.info_text_edit.setReadOnly(True)
        layout.addWidget(self.info_text_edit)

        self.is_processing = False # 标志，指示当前是否正在进行语音处理流程 (从开始录音到播放完成)

        self.worker_thread = QThread()
        self.worker = Worker(
            samplerate=SAMPLE_RATE,
            model_dir=MODEL_DIR,
            ubm_model_file=UBM_MODEL_FILE,
            user_models_files=USER_MODELS_FILES,
            identification_threshold=IDENTIFICATION_THRESHOLD,
            baidu_api_key=ASR_TTS_API_KEY,
            baidu_secret_key=ASR_TTS_SECRET_KEY,
            llm_api_key=LLM_API_KEY
        )
        self.worker.moveToThread(self.worker_thread)


        # 连接信号与槽
        self.record_button.clicked.connect(self.manage_processing_flow)

        # 连接 worker 发送的信号到 GUI 更新槽
        self.worker.progress.connect(self.update_status)
        self.worker.speaker_identified.connect(self.display_speaker_result)
        self.worker.asr_recognized.connect(self.display_asr_result)
        self.worker.error_occurred.connect(self.display_error)
        self.worker.welcome_user.connect(self.display_welcome_message) # **新增连接：处理欢迎信息信号**


        # 连接 worker 任务流程结束信号，用于重置 GUI 状态
        self.worker.finished.connect(self.reset_gui_state)

        # **连接 GUI 发送的信号到 worker 槽**
        self.start_processing_signal.connect(self.worker.start_voice_processing)
        self.stop_processing_signal.connect(self.worker.stop_recording_task)
        self.quit_worker_signal.connect(self.worker_thread.quit)


        # 启动工作线程 (线程在后台进入事件循环，等待信号)
        self.worker_thread.start()
        # 在 Worker 的 run 方法进入事件循环后，它就会停在那里等待信号


    @Slot()
    def manage_processing_flow(self):
        if not self.worker_thread.isRunning():
             self.update_status("错误：工作线程未运行。")
             return

        if not self.is_processing:
            # 开始流程
            self.is_processing = True
            self.record_button.setText("停止录音")
            self.status_label.setText("准备录音...")
            self.info_text_edit.clear()
            self.info_text_edit.append("请开始说话...")
            self.info_text_edit.append("点击停止按钮结束录音...")

            # 发送信号给 worker 启动录音
            self.start_processing_signal.emit()

        else:
            # 停止流程
            self.is_processing = False
            self.record_button.setText("处理中...")
            # 发送信号给 worker 停止录音并开始处理
            self.stop_processing_signal.emit()


    @Slot(str)
    def update_status(self, message):
        self.status_label.setText(message)
        # self.info_text_edit.append(f"状态: {message}") # 避免重复显示状态信息


    @Slot(str)
    def display_speaker_result(self, result):
        self.info_text_edit.append(f"声纹识别结果: {result}")

    @Slot(str) # 欢迎信息
    def display_welcome_message(self, user_id):
        self.info_text_edit.append(f"欢迎回来，{user_id}！")
        self.status_label.setText(f"欢迎回来，{user_id}！") # 状态标签也显示欢迎信息


    @Slot(str)
    def display_asr_result(self, result):
        self.info_text_edit.append(f"ASR 识别文本: {result}")
        self.status_label.setText("ASR 识别完成") # ASR 完成后更新状态标签


    @Slot(str)
    def display_error(self, message):
        self.info_text_edit.append(f"错误: {message}")
        self.status_label.setText("发生错误")


    @Slot()
    def reset_gui_state(self):
        self.is_processing = False
        self.record_button.setText("开始录音")
        self.status_label.setText("准备就绪")


    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            print("GUI 关闭，正在请求工作线程退出...")
            # 在退出前通知 worker 停止当前任务（如果正在录音）
            if self.worker and self.worker._is_recording_active:
                 self.stop_processing_signal.emit() # 发送停止处理信号

            # 发送信号请求 worker 线程退出其事件循环
            self.quit_worker_signal.emit()

            # 等待线程退出
            if not self.worker_thread.wait(3000):
                 print("工作线程未在规定时间内退出。")

        print("主应用关闭。")
        super().closeEvent(event)


# --- 主程序入口 ---
if __name__ == "__main__":
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    window = VoiceInteractionGUI()
    window.show()

    sys.exit(app.exec())