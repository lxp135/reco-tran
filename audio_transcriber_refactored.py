import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pyaudiowpatch as pyaudio
import wave
import speech_recognition as sr
from datetime import datetime
from pydub import AudioSegment
from pydub.utils import which
import tempfile
import queue
import io
import subprocess
import whisper
import logging
from logging.handlers import QueueHandler
import torch
import numpy as np
import psutil
import gc
from scipy import signal
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class AudioSource(Enum):
    """音频源类型枚举"""
    MICROPHONE = "microphone"
    SYSTEM_AUDIO = "system_audio"
    # COMBINED = "combined"  # 已移除合并音频功能


@dataclass
class TranscriptionConfig:
    """转写配置"""
    engine_type: str = "google"
    language: str = "zh-CN"
    chunk_size: int = 1024
    sample_rate: int = 16000
    channels: int = 1
    format: int = pyaudio.paInt16
    buffer_duration: int = 5
    error_log_interval: int = 10
    cleanup_log_interval: int = 20


class QueueLogHandler(QueueHandler):
    """队列日志处理器"""
    def __init__(self, log_queue):
        super().__init__(log_queue)
        
    def emit(self, record):
        super().emit(record)


class AudioTranscriptionEngine:
    """音频转写引擎 - 统一处理所有转写逻辑"""
    
    def __init__(self, config: TranscriptionConfig, logger_func: Callable[[str, str], None]):
        self.config = config
        self.log = logger_func
        self.recognizer = sr.Recognizer()
        self.whisper_model = None
        self.belle_pipeline = None  # BELLE模型管道
        self.model_type = "belle"  # 默认使用BELLE模型
        self.audio = pyaudio.PyAudio()
        
    def transcribe_audio_data(self, audio_data: list, source_type: AudioSource) -> Optional[str]:
        """转写音频数据的通用方法"""
        try:
            audio_bytes = b''.join(audio_data)
            temp_file_path = self._create_temp_audio_file(audio_bytes, source_type)
            
            if not temp_file_path:
                return None
                
            try:
                text = self._perform_recognition(temp_file_path)
                if text and text.strip():
                    return self._format_transcription_text(text)
                return None
            finally:
                self._cleanup_temp_file(temp_file_path)
                
        except Exception as e:
            self.log("error", f"{source_type.value}转写处理错误: {str(e)}")
            return None
    
    def _create_temp_audio_file(self, audio_bytes: bytes, source_type: AudioSource) -> Optional[str]:
        """创建临时音频文件"""
        try:
            suffix = f"_{source_type.value}.wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file_path = temp_file.name
                
            with wave.open(temp_file_path, 'wb') as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.config.format))
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(audio_bytes)
                
            return temp_file_path
        except Exception as e:
            self.log("error", f"创建临时音频文件失败: {str(e)}")
            return None
    
    def _perform_recognition(self, temp_file_path: str) -> Optional[str]:
        """执行语音识别"""
        if self.config.engine_type == "whisper":
            return self._transcribe_with_whisper(temp_file_path)
        else:
            return self._transcribe_with_google(temp_file_path)
    
    def _transcribe_with_whisper(self, temp_file_path: str) -> Optional[str]:
        """使用Whisper进行转写"""
        try:
            if self.belle_pipeline is None and self.whisper_model is None:
                self.load_whisper_model()
            
            # 优先使用BELLE模型
            if self.belle_pipeline is not None:
                self.log("info", "开始BELLE模型转写，专为中文优化...")
                start_time = time.time()
                
                # 使用BELLE模型进行转写
                result = self.belle_pipeline(
                    temp_file_path,
                    generate_kwargs={"language": "chinese", "task": "transcribe"}
                )
                
                transcribe_time = time.time() - start_time
                
                # 提取转写文本
                if isinstance(result, dict) and "text" in result:
                    text = result["text"]
                elif isinstance(result, list) and len(result) > 0 and "text" in result[0]:
                    text = result[0]["text"]
                else:
                    text = str(result)
                
                self.log("info", f"BELLE模型转写完成，耗时: {transcribe_time:.1f}秒")
                return text
                
            # 如果BELLE模型不可用，使用原生Whisper模型
            elif self.whisper_model is not None:
                self.log("info", "开始原生Whisper转写，使用中文语言...")
                start_time = time.time()
                # 使用中文语言，不进行自动检测
                result = self.whisper_model.transcribe(
                    temp_file_path, 
                    language='zh',
                    initial_prompt="以下是普通话的句子。"
                )
                transcribe_time = time.time() - start_time
                
                detected_language = result.get('language', '未知')
                self.log("info", f"原生Whisper转写完成，耗时: {transcribe_time:.1f}秒, 检测语言: {detected_language}")
                
                return result["text"]
            else:
                raise Exception("Whisper模型未加载")
        except Exception as e:
            self.log("error", f"Whisper转写失败: {str(e)}")
            return None
    
    def load_whisper_model(self):
        """加载Whisper模型"""
        if self.belle_pipeline is None and self.whisper_model is None:
            try:
                # 检测GPU可用性
                device = "cuda" if torch.cuda.is_available() else "cpu"
                gpu_info = ""
                if device == "cuda":
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    gpu_info = f"（GPU: {gpu_name}, 显存: {gpu_memory:.1f}GB）"
                    self.log("info", f"检测到GPU设备: {gpu_name}，将使用GPU加速")
                else:
                    self.log("info", "未检测到GPU设备，将使用CPU运行")
                
                # 优先尝试加载BELLE模型
                if TRANSFORMERS_AVAILABLE and self.model_type == "belle":
                    try:
                        self.log("info", f"开始加载BELLE-2/Belle-whisper-large-v3-turbo-zh模型，设备: {device} {gpu_info}")
                        
                        start_time = time.time()
                        
                        # 加载BELLE模型
                        model_id = "BELLE-2/Belle-whisper-large-v3-turbo-zh"
                        
                        # 设置torch数据类型
                        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                        
                        # 加载模型
                        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
                        model = AutoModelForSpeechSeq2Seq.from_pretrained(
                            model_id, 
                            torch_dtype=torch_dtype, 
                            low_cpu_mem_usage=True, 
                            use_safetensors=True
                        )
                        model.to(device)
                        
                        # 加载处理器
                        processor = AutoProcessor.from_pretrained(model_id)
                        
                        # 创建管道
                        self.belle_pipeline = pipeline(
                            "automatic-speech-recognition",
                            model=model,
                            tokenizer=processor.tokenizer,
                            feature_extractor=processor.feature_extractor,
                            max_new_tokens=128,
                            chunk_length_s=30,
                            batch_size=16,
                            return_timestamps=True,
                            torch_dtype=torch_dtype,
                            device=device,
                        )
                        
                        load_time = time.time() - start_time
                        self.log("info", f"BELLE-2/Belle-whisper-large-v3-turbo-zh模型加载成功，耗时: {load_time:.1f}秒，设备: {device}")
                        return
                        
                    except Exception as e:
                        self.log("warning", f"BELLE模型加载失败: {str(e)}")
                        self.log("info", "回退到原生Whisper模型...")
                        self.model_type = "whisper"
                
                # 如果BELLE模型加载失败或不可用，使用原生Whisper模型
                self.log("info", f"开始加载原生Whisper模型，设备: {device} {gpu_info}")
                
                # 优先使用turbo模型（最新最快模型）
                try:
                    self.log("info", "尝试加载turbo模型...")
                    start_time = time.time()
                    self.whisper_model = whisper.load_model("turbo", device=device)
                    load_time = time.time() - start_time
                    self.log("info", f"turbo模型加载成功，耗时: {load_time:.1f}秒，设备: {device}")
                except Exception as e1:
                    self.log("warning", f"turbo模型加载失败: {str(e1)}")
                    # 如果turbo模型失败，尝试small模型
                    try:
                        self.log("info", "尝试加载small模型...")
                        start_time = time.time()
                        self.whisper_model = whisper.load_model("small", device=device)
                        load_time = time.time() - start_time
                        self.log("info", f"small模型加载成功，耗时: {load_time:.1f}秒，设备: {device}")
                    except Exception as e2:
                        self.log("warning", f"small模型加载失败: {str(e2)}")
                        # 如果small模型失败，尝试base模型
                        try:
                            self.log("info", "尝试加载base模型...")
                            start_time = time.time()
                            self.whisper_model = whisper.load_model("base", device=device)
                            load_time = time.time() - start_time
                            self.log("info", f"base模型加载成功，耗时: {load_time:.1f}秒，设备: {device}")
                        except Exception as e3:
                            self.log("warning", f"base模型加载失败: {str(e3)}")
                            # 最后尝试tiny模型作为备选
                            try:
                                self.log("info", "尝试加载tiny模型...")
                                start_time = time.time()
                                self.whisper_model = whisper.load_model("tiny", device=device)
                                load_time = time.time() - start_time
                                self.log("warning", f"tiny模型加载成功，耗时: {load_time:.1f}秒，设备: {device}（注意：准确率较低）")
                            except Exception as e4:
                                error_msg = f"所有模型下载失败。Turbo: {str(e1)}, Small: {str(e2)}, Base: {str(e3)}, Tiny: {str(e4)}"
                                self.log("error", error_msg)
                                raise Exception(error_msg)
                        
            except Exception as e:
                self.log("error", f"Whisper模型加载失败: {str(e)}")
                raise e  # 重新抛出异常
    
    def _transcribe_with_google(self, temp_file_path: str) -> Optional[str]:
        """使用Google进行转写"""
        try:
            with sr.AudioFile(temp_file_path) as source:
                audio_for_recognition = self.recognizer.record(source)
                
            try:
                return self.recognizer.recognize_google(
                    audio_for_recognition, 
                    language=self.config.language
                )
            except sr.UnknownValueError:
                return None
            except sr.RequestError as e:
                self.log("warning", f"Google转写网络错误: {str(e)}")
                return None
        except Exception as e:
            self.log("error", f"Google转写处理错误: {str(e)}")
            return None
    
    def _format_transcription_text(self, text: str) -> str:
        """格式化转写文本"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        sentences = text.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n')
        return f"[{timestamp}] {sentences}\n"
    
    def _cleanup_temp_file(self, temp_file_path: str):
        """清理临时文件"""
        try:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except Exception as e:
            self.log("warning", f"清理临时文件失败: {str(e)}")


class TranscriptionWorker:
    """转写工作线程 - 统一处理所有转写线程逻辑"""
    
    def __init__(self, source_type: AudioSource, transcription_queue: queue.Queue, 
                 engine: AudioTranscriptionEngine, ui_callback: Callable[[str], None],
                 status_callback: Callable[[str], None]):
        self.source_type = source_type
        self.queue = transcription_queue
        self.engine = engine
        self.ui_callback = ui_callback
        self.status_callback = status_callback
        self.running = False
        self.thread = None
        self.transcription_count = 0
        
    def start(self):
        """启动转写线程"""
        if self.running:
            return
            
        self.running = True
        self.transcription_count = 0
        self.thread = threading.Thread(target=self._transcription_loop, daemon=True)
        self.thread.start()
        
        self.status_callback(f"{self.source_type.value}转写: 运行中")
        self.engine.log("info", f"{self.source_type.value}转写线程启动")
        
    def stop(self):
        """停止转写线程"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
            
        self.status_callback(f"{self.source_type.value}转写: 已停止")
        self.engine.log("info", f"{self.source_type.value}转写线程结束，共处理 {self.transcription_count} 个音频片段")
        
    def _transcription_loop(self):
        """转写循环"""
        while self.running:
            try:
                if not self.queue.empty():
                    audio_data = self.queue.get(timeout=1)
                    self.transcription_count += 1
                    
                    # 错误日志频率控制
                    should_log_error = (self.transcription_count % self.engine.config.error_log_interval == 1)
                    
                    text = self.engine.transcribe_audio_data(audio_data, self.source_type)
                    if text:
                        self.ui_callback(text)
                        self.engine.log("info", 
                            f"{self.source_type.value}转写成功 #{self.transcription_count}: "
                            f"{text[:50]}{'...' if len(text) > 50 else ''}")
                else:
                    time.sleep(0.1)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.engine.log("error", f"{self.source_type.value}转写线程异常: {str(e)}")
                continue


class UIComponentFactory:
    """UI组件工厂 - 统一创建UI组件"""
    
    @staticmethod
    def create_labeled_frame(parent, text: str, padding: str = "6") -> ttk.LabelFrame:
        """创建带标签的框架"""
        frame = ttk.LabelFrame(parent, text=text, padding=padding)
        return frame
    
    @staticmethod
    def create_button_row(parent, buttons_config: list, row: int = 0, start_col: int = 0):
        """创建按钮行"""
        for i, config in enumerate(buttons_config):
            btn = ttk.Button(parent, **config)
            btn.grid(row=row, column=start_col + i, padx=(0, 8) if i < len(buttons_config) - 1 else 0)
            
    @staticmethod
    def create_text_area_with_status(parent, title: str, height: int = 8) -> tuple:
        """创建带状态的文本区域"""
        frame = UIComponentFactory.create_labeled_frame(parent, title, "3")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        
        status_label = ttk.Label(frame, text=f"{title}: 未启动", font=("Arial", 8))
        status_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 2))
        
        text_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=height, font=("Arial", 9))
        text_area.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        return frame, status_label, text_area


class LoggerMixin:
    """日志混入类 - 统一日志处理"""
    
    def setup_logging(self):
        """设置日志系统"""
        self.log_queue = queue.Queue()
        
        # 配置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # 添加队列处理器
        queue_handler = QueueLogHandler(self.log_queue)
        queue_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        queue_handler.setFormatter(formatter)
        root_logger.addHandler(queue_handler)
        
    def log(self, level: str, message: str):
        """统一日志方法"""
        getattr(logging, level)(message)
        
    def start_log_updater(self):
        """启动日志更新线程"""
        def update_log():
            while True:
                try:
                    if hasattr(self, 'log_queue') and not self.log_queue.empty():
                        log_record = self.log_queue.get(timeout=0.1)
                        if hasattr(self, 'root'):
                            self.root.after(0, lambda record=log_record: self.append_log(record))
                    time.sleep(0.1)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"日志更新错误: {e}")
                    
        log_thread = threading.Thread(target=update_log, daemon=True)
        log_thread.start()


class AudioTranscriber(LoggerMixin):
    """重构后的音频转写器主类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("录音转写工具")
        self.root.geometry("1200x650")
        self.root.resizable(True, True)
        
        # 初始化配置
        self.config = TranscriptionConfig()
        
        # 录音状态
        self.recording = False
        self.start_time = None
        
        # 音频相关
        self.audio = pyaudio.PyAudio()
        self.microphone_frames = []
        self.system_audio_frames = []
        
        # 音频流
        self.microphone_stream = None
        self.system_audio_stream = None
        
        # 设备相关
        self.audio_devices = []
        self.microphone_device_index = None
        self.system_audio_device_index = None
        self.microphone_enabled = True
        self.system_audio_enabled = True
        
        # 音频泄漏检测
        self.mic_audio_samples = []
        self.leakage_detection_interval = 50
        self.audio_block_count = 0
        self.last_leakage_warning_time = 0
        self.leakage_warning_interval = 30
        
        # 转写相关
        self.real_time_transcription = False
        self.engine_type = self.config.engine_type  # 使用配置中的引擎类型
        self.model_type = "belle"
        
        # 队列
        self.microphone_transcription_queue = queue.Queue()
        self.system_audio_transcription_queue = queue.Queue()
        
        # 缓冲区
        self.microphone_buffer = []
        self.system_audio_buffer = []
        
        # 文件管理
        self.current_audio_file = None
        self.current_audio_files = []
        
        # 初始化组件
        self.setup_logging()
        self.start_log_updater()
        
        # 创建转写引擎
        self.transcription_engine = AudioTranscriptionEngine(self.config, self.log)
        
        # 创建转写工作器
        self.microphone_worker = None
        self.system_audio_worker = None
        
        # 设置UI
        self.setup_ui()
        
        # 初始化音频设备
        self.initialize_audio_devices()
        
        # 确保引擎配置正确同步
        self.on_engine_change()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        for i in range(3):
            main_frame.columnconfigure(i, weight=1, minsize=300)
        main_frame.rowconfigure(2, weight=1)
        
        # 录音控制区域
        self._setup_control_frame(main_frame)
        
        # 文件操作区域
        self._setup_file_frame(main_frame)
        
        # 主内容区域
        self._setup_content_frames(main_frame)
        
        # 进度条和状态栏
        self._setup_status_components(main_frame)
        
    def _setup_control_frame(self, parent):
        """设置控制框架"""
        control_frame = UIComponentFactory.create_labeled_frame(parent, "录音控制", "8")
        control_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 8))
        control_frame.columnconfigure(5, weight=1)
        
        # 录音按钮
        self.record_button = ttk.Button(control_frame, text="开始录音", command=self.toggle_recording)
        self.record_button.grid(row=0, column=0, padx=(0, 8))
        
        # 引擎选择
        ttk.Label(control_frame, text="引擎:").grid(row=0, column=1, padx=(0, 3), sticky=tk.W)
        self.engine_var = tk.StringVar(value=self.config.engine_type)  # 使用配置中的引擎类型
        self.engine_combo = ttk.Combobox(control_frame, textvariable=self.engine_var, 
                                        values=["google", "whisper"], state="readonly", width=8)
        self.engine_combo.grid(row=0, column=2, padx=(0, 8))
        self.engine_combo.bind("<<ComboboxSelected>>", self.on_engine_change)
        
        # 开关控件
        self.realtime_var = tk.BooleanVar(value=True)
        self.realtime_checkbox = ttk.Checkbutton(control_frame, text="实时转写", variable=self.realtime_var)
        self.realtime_checkbox.grid(row=0, column=3, padx=(0, 8))
        
        self.microphone_var = tk.BooleanVar(value=self.microphone_enabled)
        self.microphone_checkbox = ttk.Checkbutton(control_frame, text="麦克风", 
                                                  variable=self.microphone_var, command=self.toggle_microphone)
        self.microphone_checkbox.grid(row=0, column=4, padx=(0, 8))
        
        self.system_audio_var = tk.BooleanVar(value=self.system_audio_enabled)
        self.system_audio_checkbox = ttk.Checkbutton(control_frame, text="系统音频", 
                                                    variable=self.system_audio_var, command=self.toggle_system_audio)
        self.system_audio_checkbox.grid(row=0, column=5, padx=(0, 8))
        
        # 状态信息
        self._setup_status_info(control_frame)
        
    def _setup_status_info(self, parent):
        """设置状态信息"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=1, column=0, columnspan=9, sticky=(tk.W, tk.E), pady=(5, 0))
        status_frame.columnconfigure(1, weight=1)
        
        self.status_label = ttk.Label(status_frame, text="准备就绪", font=("Arial", 9))
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.duration_label = ttk.Label(status_frame, text="时长: 00:00", font=("Arial", 9))
        self.duration_label.grid(row=0, column=2, sticky=tk.E)
        
    def _setup_file_frame(self, parent):
        """设置文件操作框架"""
        file_frame = UIComponentFactory.create_labeled_frame(parent, "文件操作", "8")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 8))
        
        # 第一行按钮
        buttons_config_row1 = [
            {"text": "打开音频文件", "command": self.open_audio_file},
            {"text": "开始转写", "command": self.transcribe_audio, "state": "disabled"},
            {"text": "清空文本", "command": self.clear_text}
        ]
        
        for i, config in enumerate(buttons_config_row1):
            btn = ttk.Button(file_frame, **config)
            btn.grid(row=0, column=i, padx=(0, 8) if i < len(buttons_config_row1) - 1 else 0)
            if i == 1:  # 转写按钮
                self.transcribe_button = btn
                
        # 第二行按钮 - 保存功能
        buttons_config_row2 = [
            {"text": "保存麦克风转写", "command": self.save_mic_text, "state": "disabled"},
            {"text": "保存系统音频转写", "command": self.save_sys_text, "state": "disabled"},
            {"text": "保存全部转写", "command": self.save_all_text, "state": "disabled"}
        ]
        
        for i, config in enumerate(buttons_config_row2):
            btn = ttk.Button(file_frame, **config)
            btn.grid(row=1, column=i, padx=(0, 8) if i < len(buttons_config_row2) - 1 else 0, pady=(5, 0))
            if i == 0:  # 保存麦克风按钮
                self.save_mic_button = btn
            elif i == 1:  # 保存系统音频按钮
                self.save_sys_button = btn
            elif i == 2:  # 保存全部按钮
                self.save_all_button = btn
                
        # 设置默认保存按钮为保存全部
        self.save_button = self.save_all_button
                
    def _setup_content_frames(self, parent):
        """设置主内容框架"""
        # 音频文件管理区域
        self._setup_audio_files_frame(parent)
        
        # 转写结果区域
        self._setup_transcription_frame(parent)
        
        # 日志区域
        self._setup_log_frame(parent)
        
    def _setup_audio_files_frame(self, parent):
        """设置音频文件管理框架"""
        audio_files_frame = UIComponentFactory.create_labeled_frame(parent, "音频文件管理")
        audio_files_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 3))
        audio_files_frame.columnconfigure(0, weight=1, minsize=300)
        audio_files_frame.rowconfigure(1, weight=1)
        
        # 历史文件区域
        history_files_frame = UIComponentFactory.create_labeled_frame(audio_files_frame, "历史文件", "3")
        history_files_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        history_files_frame.columnconfigure(0, weight=1)
        history_files_frame.rowconfigure(1, weight=1)
        
        # 历史文件控制按钮
        history_control_frame = ttk.Frame(history_files_frame)
        history_control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 3))
        
        UIComponentFactory.create_button_row(history_control_frame, [
            {"text": "刷新", "command": self.refresh_history_files, "width": 6},
            {"text": "清理", "command": self.clean_history_files, "width": 6}
        ])
        
        # 历史文件列表
        self.history_files_listbox = tk.Listbox(history_files_frame, height=20, font=("Arial", 8), width=30)
        history_files_scrollbar = ttk.Scrollbar(history_files_frame, orient="vertical", 
                                               command=self.history_files_listbox.yview)
        self.history_files_listbox.configure(yscrollcommand=history_files_scrollbar.set)
        self.history_files_listbox.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        history_files_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        
        # 历史文件操作按钮
        history_files_buttons = ttk.Frame(history_files_frame)
        history_files_buttons.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(3, 0))
        
        UIComponentFactory.create_button_row(history_files_buttons, [
            {"text": "播放", "command": self.play_history_file, "width": 6},
            {"text": "删除", "command": self.delete_history_file, "width": 6},
            {"text": "文件夹", "command": self.open_history_folder, "width": 6}
        ])
        
        self.refresh_history_files()
        
    def _setup_transcription_frame(self, parent):
        """设置转写结果框架"""
        result_frame = UIComponentFactory.create_labeled_frame(parent, "转写结果")
        result_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(3, 3))
        result_frame.columnconfigure(0, weight=1, minsize=300)
        result_frame.rowconfigure(0, weight=1)
        result_frame.rowconfigure(1, weight=1)
        
        # 麦克风转写区域
        mic_frame, self.mic_status, self.mic_text_area = UIComponentFactory.create_text_area_with_status(
            result_frame, "麦克风转写", 8)
        mic_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 3))
        
        # 系统音频转写区域
        sys_frame, self.sys_status, self.sys_text_area = UIComponentFactory.create_text_area_with_status(
            result_frame, "系统音频转写", 8)
        sys_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 移除兼容性代码，不再需要统一的text_area
        
    def _setup_log_frame(self, parent):
        """设置日志框架"""
        log_frame = UIComponentFactory.create_labeled_frame(parent, "执行日志")
        log_frame.grid(row=2, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(3, 0))
        log_frame.columnconfigure(0, weight=1, minsize=300)
        log_frame.rowconfigure(1, weight=1)
        
        # 日志控制
        log_control_frame = ttk.Frame(log_frame)
        log_control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 3))
        
        ttk.Button(log_control_frame, text="清空日志", command=self.clear_log).grid(row=0, column=0, padx=(0, 5))
        
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_control_frame, text="自动滚动", variable=self.auto_scroll_var).grid(row=0, column=1)
        
        # 日志显示区域
        self.log_area = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=18, font=("Consolas", 8))
        self.log_area.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.log_area.config(state=tk.DISABLED)
        
    def _setup_status_components(self, parent):
        """设置状态组件"""
        # 进度条
        self.progress = ttk.Progressbar(parent, mode='indeterminate')
        self.progress.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 10))
        
        # 状态栏
        self.status_bar = ttk.Label(parent, text="就绪", relief=tk.SUNKEN)
        self.status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E))

    # 以下是简化后的核心方法，保持原有功能但减少重复代码
    
    def toggle_recording(self):
        """切换录音状态"""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()
            
    def start_recording(self):
        """开始录音"""
        try:
            self.recording = True
            self.start_time = time.time()
            
            # 清空缓冲区
            self.microphone_frames.clear()
            self.system_audio_frames.clear()
            
            # 启动录音线程
            self.record_thread = threading.Thread(target=self.record_audio, daemon=True)
            self.record_thread.start()
            
            # 启动转写工作器
            if self.realtime_var.get():
                self.real_time_transcription = True
                
                if self.microphone_enabled:
                    self.microphone_worker = TranscriptionWorker(
                        AudioSource.MICROPHONE,
                        self.microphone_transcription_queue,
                        self.transcription_engine,
                        lambda text: self.root.after(0, lambda: self.append_mic_text(text)),
                        lambda status: self.root.after(0, lambda: self.mic_status.config(text=status))
                    )
                    self.microphone_worker.start()
                    
                if self.system_audio_enabled:
                    self.system_audio_worker = TranscriptionWorker(
                        AudioSource.SYSTEM_AUDIO,
                        self.system_audio_transcription_queue,
                        self.transcription_engine,
                        lambda text: self.root.after(0, lambda: self.append_sys_text(text)),
                        lambda status: self.root.after(0, lambda: self.sys_status.config(text=status))
                    )
                    self.system_audio_worker.start()
            
            # 更新UI
            self.record_button.config(text="停止录音")
            self.status_label.config(text="录音中...")
            
            # 启动计时器
            self.update_timer()
            
            self.log("info", "录音已开始")
            
        except Exception as e:
            self.log("error", f"开始录音失败: {str(e)}")
            self.recording = False
        
    def stop_recording(self):
        """停止录音"""
        try:
            self.recording = False
            self.real_time_transcription = False
            
            # 停止转写工作器
            if self.microphone_worker:
                self.microphone_worker.stop()
                self.microphone_worker = None
                
            if self.system_audio_worker:
                self.system_audio_worker.stop()
                self.system_audio_worker = None
            
            # 停止音频流
            if self.microphone_stream:
                self.microphone_stream.stop_stream()
                self.microphone_stream.close()
                self.microphone_stream = None
                
            if self.system_audio_stream:
                self.system_audio_stream.stop_stream()
                self.system_audio_stream.close()
                self.system_audio_stream = None
            
            # 更新UI
            self.record_button.config(text="开始录音")
            self.status_label.config(text="准备就绪")
            
            # 保存录音文件
            if self.microphone_frames or self.system_audio_frames:
                self._save_recording_files()
            
            self.log("info", "录音已停止")
            
        except Exception as e:
            self.log("error", f"停止录音失败: {str(e)}")
        
    def on_engine_change(self, event=None):
        """引擎变更处理"""
        self.engine_type = self.engine_var.get()
        self.config.engine_type = self.engine_type
        self.transcription_engine.config = self.config
        if self.engine_type == "whisper":
            self.load_whisper_model_async()
        else:
            self.status_label.config(text=f"已切换到{self.engine_type}引擎")
    
    def load_whisper_model_async(self):
        """异步加载Whisper模型"""
        # 显示进度条并禁用相关控件
        self.progress.start()
        self.engine_combo.config(state="disabled")
        self.record_button.config(state="disabled")
        self.transcribe_button.config(state="disabled")
        self.status_label.config(text="正在加载Whisper模型...")
        self.status_bar.config(text="正在加载模型，请稍候...")
        
        # 在后台线程中加载模型
        def load_model_thread():
            try:
                self.transcription_engine.load_whisper_model()
                # 加载完成后更新UI
                self.root.after(0, self._on_model_loaded_success)
            except Exception as e:
                # 加载失败后更新UI
                self.root.after(0, lambda: self._on_model_loaded_error(str(e)))
        
        threading.Thread(target=load_model_thread, daemon=True).start()
    
    def _on_model_loaded_success(self):
        """模型加载成功的UI更新"""
        self.progress.stop()
        self.engine_combo.config(state="readonly")
        self.record_button.config(state="normal")
        self.transcribe_button.config(state="normal" if hasattr(self, 'audio_file_path') and self.audio_file_path else "disabled")
        self.status_label.config(text=f"已切换到{self.engine_type}引擎")
        self.status_bar.config(text="Whisper模型加载完成")
    
    def _on_model_loaded_error(self, error_msg):
        """模型加载失败的UI更新"""
        self.progress.stop()
        self.engine_combo.config(state="readonly")
        self.record_button.config(state="normal")
        self.transcribe_button.config(state="normal" if hasattr(self, 'audio_file_path') and self.audio_file_path else "disabled")
        self.status_label.config(text="模型加载失败，已回退到Google引擎")
        self.status_bar.config(text="模型加载失败")
        self.engine_var.set("google")
        self.engine_type = "google"
        self.config.engine_type = "google"
        self.transcription_engine.config = self.config
        messagebox.showerror("错误", f"加载Whisper模型失败: {error_msg}\n\n建议：\n1. 检查网络连接\n2. 确保有足够的磁盘空间\n3. 安装transformers库: pip install transformers\n4. 尝试重新启动程序")
        
    def toggle_microphone(self):
        """切换麦克风状态"""
        self.microphone_enabled = self.microphone_var.get()
        
    def toggle_system_audio(self):
        """切换系统音频状态"""
        self.system_audio_enabled = self.system_audio_var.get()
        
    def append_mic_text(self, text: str):
        """添加麦克风转写文本"""
        self.mic_text_area.config(state=tk.NORMAL)
        self.mic_text_area.insert(tk.END, text)
        self.mic_text_area.see(tk.END)
        self.mic_text_area.config(state=tk.DISABLED)
        
    def append_sys_text(self, text: str):
        """添加系统音频转写文本"""
        self.sys_text_area.config(state=tk.NORMAL)
        self.sys_text_area.insert(tk.END, text)
        self.sys_text_area.see(tk.END)
        self.sys_text_area.config(state=tk.DISABLED)
        
    def append_log(self, log_record):
        """添加日志"""
        try:
            if hasattr(self, 'log_area'):
                self.log_area.config(state=tk.NORMAL)
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_text = f"[{timestamp}] {log_record.levelname}: {log_record.getMessage()}\n"
                self.log_area.insert(tk.END, log_text)
                
                if self.auto_scroll_var.get():
                    self.log_area.see(tk.END)
                    
                self.log_area.config(state=tk.DISABLED)
        except Exception as e:
            print(f"日志显示错误: {e}")
            
    def clear_log(self):
        """清空日志"""
        self.log_area.config(state=tk.NORMAL)
        self.log_area.delete(1.0, tk.END)
        self.log_area.config(state=tk.DISABLED)
        
    def record_audio(self):
        """录音线程函数"""
        try:
            # 初始化音频流
            if self.microphone_enabled and self.microphone_device_index is not None:
                self.microphone_stream = self.audio.open(
                    format=self.config.format,
                    channels=self.config.channels,
                    rate=self.config.sample_rate,
                    input=True,
                    input_device_index=self.microphone_device_index,
                    frames_per_buffer=self.config.chunk_size
                )
                
            if self.system_audio_enabled and self.system_audio_device_index is not None:
                device_info = self.audio.get_device_info_by_index(self.system_audio_device_index)
                self.system_audio_stream = self.audio.open(
                    format=self.config.format,
                    channels=int(device_info['maxInputChannels']),
                    rate=int(device_info['defaultSampleRate']),
                    input=True,
                    input_device_index=self.system_audio_device_index,
                    frames_per_buffer=self.config.chunk_size
                )
            
            buffer_count = 0
            while self.recording:
                # 录制麦克风音频
                if self.microphone_stream:
                    try:
                        mic_data = self.microphone_stream.read(self.config.chunk_size, exception_on_overflow=False)
                        self.microphone_frames.append(mic_data)
                        self.microphone_buffer.append(mic_data)
                    except Exception as e:
                        self.log("warning", f"麦克风录音错误: {str(e)}")
                
                # 录制系统音频
                if self.system_audio_stream:
                    try:
                        sys_data = self.system_audio_stream.read(self.config.chunk_size, exception_on_overflow=False)
                        # 处理多声道数据
                        device_info = self.audio.get_device_info_by_index(self.system_audio_device_index)
                        processed_data = self.analyze_channel_data(sys_data, int(device_info['maxInputChannels']))
                        self.system_audio_frames.append(processed_data)
                        self.system_audio_buffer.append(processed_data)
                    except Exception as e:
                        self.log("warning", f"系统音频录音错误: {str(e)}")
                
                # 实时转写处理
                if self.real_time_transcription:
                    buffer_count += 1
                    if buffer_count >= self.config.sample_rate // self.config.chunk_size * self.config.buffer_duration:
                        # 发送音频数据到转写队列
                        if self.microphone_buffer and self.microphone_enabled:
                            self.microphone_transcription_queue.put(self.microphone_buffer.copy())
                            self.microphone_buffer.clear()
                            
                        if self.system_audio_buffer and self.system_audio_enabled:
                            self.system_audio_transcription_queue.put(self.system_audio_buffer.copy())
                            self.system_audio_buffer.clear()
                            
                        buffer_count = 0
                        
        except Exception as e:
            self.log("error", f"录音线程错误: {str(e)}")
            
    def analyze_channel_data(self, data, channels):
        """分析和处理多声道音频数据"""
        try:
            if channels == 1:
                return data
                
            # 转换为numpy数组进行处理
            audio_array = np.frombuffer(data, dtype=np.int16)
            
            if len(audio_array) == 0:
                return b'\x00\x00' * (len(data) // (channels * 2))
                
            # 重塑为多声道格式
            if len(audio_array) % channels != 0:
                # 填充数据以匹配声道数
                padding_needed = channels - (len(audio_array) % channels)
                audio_array = np.pad(audio_array, (0, padding_needed), mode='constant')
                
            reshaped = audio_array.reshape(-1, channels)
            
            # 混合到单声道
            if channels == 2:
                # 立体声处理
                mono_audio = np.mean(reshaped, axis=1, dtype=np.int16)
            else:
                # 多声道处理
                mono_audio = np.mean(reshaped, axis=1, dtype=np.int16)
                
            return mono_audio.tobytes()
            
        except Exception as e:
            self.log("warning", f"声道数据处理错误: {str(e)}")
            return b'\x00\x00' * (len(data) // (channels * 2))
            
    def _save_recording_files(self):
        """保存录音文件 - 独立保存麦克风和系统音频"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_dir = "audio"
            os.makedirs(audio_dir, exist_ok=True)
            
            saved_files = []
            
            # 保存麦克风音频
            if self.microphone_frames:
                mic_file = os.path.join(audio_dir, f"microphone_{timestamp}.wav")
                self._save_wav_file(mic_file, self.microphone_frames, self.config.sample_rate, self.config.channels)
                saved_files.append(mic_file)
                self.log("info", f"麦克风音频已保存: {mic_file}")
                
            # 保存系统音频
            if self.system_audio_frames:
                sys_file = os.path.join(audio_dir, f"system_audio_{timestamp}.wav")
                self._save_wav_file(sys_file, self.system_audio_frames, self.config.sample_rate, 1)
                saved_files.append(sys_file)
                self.log("info", f"系统音频已保存: {sys_file}")
                
            # 不再合并音频文件，保持独立
            self.current_audio_files = saved_files
            self.refresh_history_files()
            
        except Exception as e:
            self.log("error", f"保存录音文件失败: {str(e)}")
            
    def _save_wav_file(self, filename, frames, sample_rate, channels):
        """保存WAV文件"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(self.audio.get_sample_size(self.config.format))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
            
    # 移除音频合并功能 - 保持麦克风和系统音频完全独立
            
    def update_timer(self):
        """更新录音时长显示"""
        if self.recording and self.start_time:
            elapsed = time.time() - self.start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.duration_label.config(text=f"时长: {minutes:02d}:{seconds:02d}")
            self.root.after(1000, self.update_timer)
            
    def initialize_audio_devices(self):
        """初始化音频设备"""
        try:
            self.audio_devices.clear()
            
            # 扫描音频设备
            for i in range(self.audio.get_device_count()):
                try:
                    device_info = self.audio.get_device_info_by_index(i)
                    if device_info['maxInputChannels'] > 0:
                        self.audio_devices.append({
                            'index': i,
                            'name': device_info['name'],
                            'channels': device_info['maxInputChannels'],
                            'sample_rate': device_info['defaultSampleRate'],
                            'is_loopback': 'loopback' in device_info['name'].lower()
                        })
                except Exception:
                    continue
                    
            # 自动选择设备
            self._auto_select_devices()
            
            self.log("info", f"发现 {len(self.audio_devices)} 个音频输入设备")
            
        except Exception as e:
            self.log("error", f"初始化音频设备失败: {str(e)}")
            
    def _auto_select_devices(self):
        """自动选择音频设备"""
        # 选择默认麦克风
        try:
            default_input = self.audio.get_default_input_device_info()
            self.microphone_device_index = default_input['index']
            self.log("info", f"选择麦克风设备: {default_input['name']}")
        except Exception:
            # 选择第一个非回环设备
            for device in self.audio_devices:
                if not device['is_loopback']:
                    self.microphone_device_index = device['index']
                    self.log("info", f"选择麦克风设备: {device['name']}")
                    break
                    
        # 选择系统音频设备（回环设备）
        for device in self.audio_devices:
            if device['is_loopback']:
                self.system_audio_device_index = device['index']
                self.log("info", f"选择系统音频设备: {device['name']}")
                break
                
    # 文件操作相关方法
    def open_audio_file(self):
        """打开音频文件"""
        file_path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[("音频文件", "*.wav *.mp3 *.flac *.m4a"), ("所有文件", "*.*")]
        )
        if file_path:
            self.current_audio_file = file_path
            self.transcribe_button.config(state="normal")
            self.log("info", f"已选择音频文件: {os.path.basename(file_path)}")
            
    def transcribe_audio(self):
        """转写音频文件"""
        if not self.current_audio_file:
            messagebox.showwarning("警告", "请先选择音频文件")
            return
            
        try:
            self.progress.start()
            self.transcribe_button.config(state="disabled")
            
            # 在后台线程中执行转写
            thread = threading.Thread(target=self._perform_file_transcription, daemon=True)
            thread.start()
            
        except Exception as e:
            self.log("error", f"转写失败: {str(e)}")
            self.progress.stop()
            self.transcribe_button.config(state="normal")
            
    def _perform_file_transcription(self):
        """执行文件转写"""
        try:
            # 文件转写默认使用麦克风源类型
            text = self.transcription_engine.transcribe_audio_data(
                [open(self.current_audio_file, 'rb').read()], 
                AudioSource.MICROPHONE
            )
            
            if text:
                self.root.after(0, lambda: self._update_transcription_result(text))
            else:
                self.root.after(0, lambda: messagebox.showinfo("提示", "未识别到语音内容"))
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"转写失败: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.transcribe_button.config(state="normal"))
            
    def _update_transcription_result(self, text):
        """更新转写结果"""
        self.mic_text_area.config(state=tk.NORMAL)
        self.mic_text_area.delete(1.0, tk.END)
        self.mic_text_area.insert(tk.END, text)
        self.mic_text_area.config(state=tk.DISABLED)
        self.save_all_button.config(state="normal")
        
    def save_mic_text(self):
        """保存麦克风转写文本"""
        text = self.mic_text_area.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("警告", "没有可保存的麦克风转写文本")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="保存麦克风转写文本",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
            initialvalue="麦克风转写.txt"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                messagebox.showinfo("成功", "麦克风转写文本已保存")
                self.log("info", f"麦克风转写文本已保存: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")
                
    def save_sys_text(self):
        """保存系统音频转写文本"""
        text = self.sys_text_area.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("警告", "没有可保存的系统音频转写文本")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="保存系统音频转写文本",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
            initialvalue="系统音频转写.txt"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                messagebox.showinfo("成功", "系统音频转写文本已保存")
                self.log("info", f"系统音频转写文本已保存: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")
                
    def save_all_text(self):
        """保存全部转写文本"""
        mic_text = self.mic_text_area.get(1.0, tk.END).strip()
        sys_text = self.sys_text_area.get(1.0, tk.END).strip()
        
        if not mic_text and not sys_text:
            messagebox.showwarning("警告", "没有可保存的转写文本")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="保存全部转写文本",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
            initialvalue="全部转写.txt"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    if mic_text:
                        f.write("=== 麦克风转写 ===\n")
                        f.write(mic_text)
                        f.write("\n\n")
                    if sys_text:
                        f.write("=== 系统音频转写 ===\n")
                        f.write(sys_text)
                        f.write("\n")
                messagebox.showinfo("成功", "全部转写文本已保存")
                self.log("info", f"全部转写文本已保存: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")
                
    def save_text(self):
        """保存转写文本（向后兼容方法）"""
        self.save_all_text()
                
    def clear_text(self):
        """清空转写文本"""
        self.mic_text_area.config(state=tk.NORMAL)
        self.mic_text_area.delete(1.0, tk.END)
        self.mic_text_area.config(state=tk.DISABLED)
        
        self.sys_text_area.config(state=tk.NORMAL)
        self.sys_text_area.delete(1.0, tk.END)
        self.sys_text_area.config(state=tk.DISABLED)
        
        self.save_mic_button.config(state="disabled")
        self.save_sys_button.config(state="disabled")
        self.save_all_button.config(state="disabled")
        
    def refresh_history_files(self):
        """刷新历史文件列表"""
        try:
            self.history_files_listbox.delete(0, tk.END)
            
            audio_dir = "audio"
            if os.path.exists(audio_dir):
                files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
                files.sort(reverse=True)  # 最新的在前
                
                for file in files:
                    self.history_files_listbox.insert(tk.END, file)
                    
        except Exception as e:
            self.log("error", f"刷新历史文件失败: {str(e)}")
            
    def clean_history_files(self):
        """清理历史文件"""
        if messagebox.askyesno("确认", "确定要删除所有历史音频文件吗？"):
            try:
                audio_dir = "audio"
                if os.path.exists(audio_dir):
                    for file in os.listdir(audio_dir):
                        if file.endswith('.wav'):
                            os.remove(os.path.join(audio_dir, file))
                            
                self.refresh_history_files()
                messagebox.showinfo("成功", "历史文件已清理")
                self.log("info", "历史音频文件已清理")
                
            except Exception as e:
                messagebox.showerror("错误", f"清理失败: {str(e)}")
                
    def play_history_file(self):
        """播放历史文件"""
        selection = self.history_files_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要播放的文件")
            return
            
        filename = self.history_files_listbox.get(selection[0])
        file_path = os.path.join("audio", filename)
        
        if os.path.exists(file_path):
            self.play_audio_file(file_path)
        else:
            messagebox.showerror("错误", "文件不存在")
            
    def delete_history_file(self):
        """删除历史文件"""
        selection = self.history_files_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要删除的文件")
            return
            
        filename = self.history_files_listbox.get(selection[0])
        
        if messagebox.askyesno("确认", f"确定要删除文件 {filename} 吗？"):
            try:
                file_path = os.path.join("audio", filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.refresh_history_files()
                    messagebox.showinfo("成功", "文件已删除")
                    self.log("info", f"已删除文件: {filename}")
                else:
                    messagebox.showerror("错误", "文件不存在")
            except Exception as e:
                messagebox.showerror("错误", f"删除失败: {str(e)}")
                
    def open_history_folder(self):
        """打开历史文件夹"""
        audio_dir = os.path.abspath("audio")
        if os.path.exists(audio_dir):
            self.open_folder(audio_dir)
        else:
            messagebox.showinfo("提示", "音频文件夹不存在")
            
    def play_audio_file(self, file_path):
        """播放音频文件"""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(file_path)
            else:  # macOS/Linux
                subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', file_path])
            self.log("info", f"正在播放: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"播放失败: {str(e)}")
            
    def open_folder(self, folder_path):
        """打开文件夹"""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(folder_path)
            else:  # macOS/Linux
                subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', folder_path])
        except Exception as e:
            messagebox.showerror("错误", f"打开文件夹失败: {str(e)}")
            
    def append_mic_text(self, text: str):
        """添加麦克风转写文本"""
        if text and text.strip():
            self.mic_text_area.config(state=tk.NORMAL)
            self.mic_text_area.insert(tk.END, text + "\n")
            self.mic_text_area.see(tk.END)
            self.mic_text_area.config(state=tk.DISABLED)
            self.save_mic_button.config(state="normal")
            self.save_all_button.config(state="normal")
            
    def append_sys_text(self, text: str):
        """添加系统音频转写文本"""
        if text and text.strip():
            self.sys_text_area.config(state=tk.NORMAL)
            self.sys_text_area.insert(tk.END, text + "\n")
            self.sys_text_area.see(tk.END)
            self.sys_text_area.config(state=tk.DISABLED)
            self.save_sys_button.config(state="normal")
            self.save_all_button.config(state="normal")
            
    def toggle_recording(self):
        """切换录音状态"""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()
            
    def toggle_microphone(self):
        """切换麦克风开关"""
        self.microphone_enabled = self.mic_var.get()
        status = "启用" if self.microphone_enabled else "禁用"
        self.log("info", f"麦克风已{status}")
        
    def toggle_system_audio(self):
        """切换系统音频开关"""
        self.system_audio_enabled = self.sys_var.get()
        status = "启用" if self.system_audio_enabled else "禁用"
        self.log("info", f"系统音频已{status}")
        
    def on_closing(self):
        """窗口关闭事件处理"""
        try:
            # 停止录音
            if self.recording:
                self.stop_recording()
                
            # 停止转写工作器
            if self.microphone_worker:
                self.microphone_worker.stop()
            if self.system_audio_worker:
                self.system_audio_worker.stop()
                
            # 关闭音频
            if hasattr(self, 'audio'):
                self.audio.terminate()
                
            # 销毁窗口
            self.root.destroy()
            
        except Exception as e:
            self.log("error", f"关闭程序时出错: {str(e)}")
            self.root.destroy()
    
    def __del__(self):
        """析构函数"""
        try:
            if hasattr(self, 'audio'):
                self.audio.terminate()
        except Exception:
            pass


def main():
    """主函数"""
    try:
        # 创建主窗口
        root = tk.Tk()
        
        # 创建应用实例
        app = AudioTranscriber(root)
        
        # 设置窗口关闭事件
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # 启动主循环
        root.mainloop()
        
    except Exception as e:
        print(f"程序启动失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()