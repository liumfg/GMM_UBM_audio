# baidu_api_client.py

import requests
import json
import base64
import time
import uuid
import traceback
import socket 

class BaiduAPIClient:
    def __init__(self, asr_tts_api_key, asr_tts_secret_key, llm_api_key):
        """
        初始化百度 API 客户端，接收 ASR/TTS 的密钥对和 LLM 的 API Key。

        Args:
            asr_tts_api_key (str): ASR/TTS 服务的 API Key。
            asr_tts_secret_key (str): ASR/TTS 服务的 Secret Key。
            llm_api_key (str): LLM 服务的 API Key (只有 Key)。
        """
        self._asr_tts_api_key = asr_tts_api_key
        self._asr_tts_secret_key = asr_tts_secret_key
        self._llm_api_key = llm_api_key # 只存储 LLM API Key

        self._asr_tts_access_token = None
        self._asr_tts_token_expiry_time = 0

        self.cuid = '123456PYTHON' # 硬编码 CUID


    def _get_oauth_token(self, api_key, secret_key):
        """
        从百度 OAuth 服务获取指定密钥对的 Access Token (用于 ASR/TTS)。
        LLM 服务可能不使用此方法获取 Token。
        """
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": api_key,
            "client_secret": secret_key
        }
        try:
            print(f"正在获取 OAuth Access Token (Key: {api_key[:8]}...)...")
            response = requests.post(url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()

            if "access_token" in result:
                token = result["access_token"]
                expiry_time = time.time() + result.get("expires_in", 2592000) - 60 # 提前 60 秒刷新
                print("成功获取 OAuth Access Token。")
                return token, expiry_time
            else:
                print(f"获取 OAuth Access Token 失败: {result}")
                error_code = result.get("error", "未知错误码")
                error_msg = result.get("error_description", "未知错误信息")
                print(f"获取 OAuth Access Token 失败: 错误码 {error_code}, 信息: {error_msg}")
                return None, 0

        except requests.exceptions.RequestException as e:
            print(f"获取 OAuth Access Token 请求失败: {e}")
            print(f"错误信息: {e}")
            return None, 0
        except Exception as e:
            print(f"获取 OAuth Access Token 时发生未知错误: {e}")
            print(f"错误信息: {e}")
            return None, 0


    # 获取 ASR/TTS Access Token 的方法 (使用 OAuth)
    def get_asr_tts_access_token(self):
        """
        获取 ASR/TTS 服务当前有效的 Access Token。
        """
        if self._asr_tts_access_token and time.time() < self._asr_tts_token_expiry_time:
            return self._asr_tts_access_token
        else:
            print("ASR/TTS Access Token 已过期或未获取，尝试刷新。")
            token, expiry_time = self._get_oauth_token(self._asr_tts_api_key, self._asr_tts_secret_key)
            self._asr_tts_access_token = token
            self._asr_tts_token_expiry_time = expiry_time
            return self._asr_tts_access_token


    def asr(self, audio_data_bytes, audio_format="pcm", sample_rate=16000):
        token = self.get_asr_tts_access_token() # 使用 ASR/TTS 的 Access Token
        if not token:
            print("ASR 失败: 无法获取 Access Token。")
            return ""

        url = "https://vop.baidu.com/server_api"
        audio_len = len(audio_data_bytes)
        if audio_len == 0:
            print("ASR 失败: 音频数据为空。")
            return ""

        params = {
            "format": audio_format,
            "rate": sample_rate,
            "channel": 1,
            "cuid": self.cuid,
            "token": token,
            "len": audio_len,
            "speech": base64.b64encode(audio_data_bytes).decode('utf-8'),
            "dev_pid": 1537
        }

        post_data = json.dumps(params, sort_keys=False)
        post_data_bytes = post_data.encode('utf-8')
        headers = {'Content-Type': 'application/json'}

        try:
            print("正在调用百度 ASR API...")
            response = requests.post(url, data=post_data_bytes, headers=headers, timeout=20)
            response.raise_for_status()
            result = response.json()

            if result.get("err_no") == 0 and "result" in result and result["result"]:
                recognized_text = "".join(result["result"])
                print(f"ASR 成功，识别文本: {recognized_text}")
                return recognized_text
            else:
                error_code = result.get("err_no", "未知错误码")
                error_msg = result.get("err_msg", "未知错误信息")
                print(f"ASR 失败: 错误码 {error_code}, 信息: {error_msg}")
                return ""

        except requests.exceptions.RequestException as e:
            print(f"ASR 请求失败: {e}")
            print(f"错误信息: {e}")
            return ""
        except json.JSONDecodeError:
            print(f"ASR 响应不是有效的 JSON。")
            return ""
        except Exception as e:
            print(f"ASR 调用时发生未知错误: {e}")
            print(f"错误信息: {e}")
            return ""

    def tts(self, text, speaker=0, audio_format="pcm", sample_rate=16000):
        token = self.get_asr_tts_access_token() # 使用 ASR/TTS 的 Access Token
        if not token:
            print("TTS 失败: 无法获取 Access Token。")
            return None

        url = f"https://tsn.baidu.com/text2audio?access_token={token}"
        params = {
            "tex": text.encode('utf-8'),
            "tok": token,
            "cuid": self.cuid,
            "ctp": 1,
            "lan": "zh",
            "spd": 5,
            "pit": 5,
            "vol": 5,
            "per": speaker,
            "aue": 6 if audio_format == "pcm" and sample_rate == 16000 else (3 if audio_format == "pcm" and sample_rate == 8000 else (4 if audio_format == "mp3" else 3)),
            "fmt": audio_format
        }

        try:
            print(f"正在调用百度 TTS API 合成文本: '{text}'...")
            response = requests.post(url, data=params, timeout=10)
            if 'audio' in response.headers.get('Content-Type', ''):
                print("TTS 成功，收到音频数据。")
                return response.content
            else:
                try:
                    result = response.json()
                    error_code = result.get("err_no", "未知错误码")
                    error_msg = result.get("err_msg", "未知错误信息")
                    print(f"TTS 失败: 错误码 {error_code}, 信息: {error_msg}")
                except json.JSONDecodeError:
                     print(f"TTS 失败: 收到非音频数据且无法解析为 JSON 错误信息。状态码: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"TTS 请求失败: {e}")
            print(f"错误信息: {e}")
            return None
        except Exception as e:
            print(f"TTS 调用时发生未知错误: {e}")
            print(f"错误信息: {e}")
            return None

    # --- 调用百度 LLM API 进行问答的方法 (使用 LLM API Key 直接认证) ---
    def chat_with_llm(self, message_history):
        llm_api_key = self._llm_api_key
        if not llm_api_key:
            print("LLM 调用失败: 未提供 LLM API Key。")
            return "LLM 调用失败: 未提供 API Key。"

        # 打印即将发送的整个对话历史
        print(f"工作线程：调用百度 LLM API 进行问答，发送对话历史: {message_history}")

        LLM_HOSTNAME = "qianfan.baidubce.com" # 确保这里是正确的域名
        LLM_PATH = "/v2/chat/completions"
        LLM_API_URL = f"https://{LLM_HOSTNAME}{LLM_PATH}"

        try:
            print(f"工作线程：尝试解析主机名 '{LLM_HOSTNAME}'...")
            ip_address = socket.gethostbyname(LLM_HOSTNAME)
            print(f"工作线程：主机名 '{LLM_HOSTNAME}' 解析到 IP: {ip_address}")
        except socket.gaierror as e:
            print(f"工作线程：DNS 解析失败: {e}")
            return f"LLM 调用失败: DNS 解析 '{LLM_HOSTNAME}' 失败 - {e}"
        except Exception as e:
            print(f"工作线程：DNS 解析时发生未知错误: {e}")
            return f"LLM 调用失败: DNS 解析时发生未知错误 - {e}"
        # *** DNS 解析检查结束 ***


        # API 请求体结构
        payload = {
            "model": "ernie-4.5-8k-preview", 
            "messages": message_history
        }

        headers = { 'Content-Type': 'application/json',
                   'Authorization': f'Bearer {llm_api_key}'}

        try:
            print(f"工作线程：向 LLM API 发送请求 (URL: {LLM_API_URL})...")
            print(f"工作线程：请求 Header 中的 Authorization: Bearer {llm_api_key[:8]}...")
            print(f"工作线程：请求 Body: {json.dumps(payload)}") # 打印整个 payload

            response = requests.post(LLM_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()

            result = response.json()
            print(f"工作线程：收到 LLM API 响应: {result}")

            # 解析 API 响应，提取回答文本
            if 'choices' in result and result['choices'] and 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                 llm_answer = result['choices'][0]['message']['content']
                 print(f"工作线程：LLM 调用成功，回答: {llm_answer}")
                 # 成功时返回模型的回答文本
                 return llm_answer
            elif 'error_code' in result:
                 error_code = result.get("error_code", "未知错误码")
                 error_msg = result.get("error_msg", "未知错误信息")
                 print(f"工作线程：LLM API 返回错误: 错误码 {error_code}, 信息: {error_msg}")
                 # API 返回错误时，返回错误信息字符串
                 return f"LLM API 错误: {error_msg}"
            else:
                print(f"工作线程：LLM API 响应格式未知: {result}")
                # 响应格式未知时，返回提示字符串
                return "LLM API 响应格式未知。"


        except requests.exceptions.RequestException as e:
            print(f"工作线程：LLM 请求失败: {e}")
            print(f"错误信息: {e}")
            # 将捕获到的请求异常信息作为回答返回给用户
            return f"LLM 请求失败: {e}"
        except json.JSONDecodeError:
            print(f"工作线程：LLM API 响应不是有效的 JSON。")
            return "LLM API 响应格式错误。"
        except Exception as e:
            print(f"工作线程：LLM 调用时发生未知错误: {e}")
            print(f"错误信息: {e}")
            return f"LLM 调用未知错误: {e}"