import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pyaudio
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
try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class QueueLogHandler(QueueHandler):
    """自定义队列日志处理器"""
    def __init__(self, log_queue):
        super().__init__(log_queue)
    
    def emit(self, record):
        """发送日志记录到队列"""
        try:
            self.queue.put_nowait(record)
        except Exception:
            self.handleError(record)

class AudioTranscriber:
    def __init__(self, root):
        self.root = root
        self.root.title("录音转写工具")
        self.root.geometry("1400x700")
        self.root.resizable(True, True)
        
        # ffmpeg会自动从系统PATH中查找，无需手动设置路径
        
        # 录音参数
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # 使用16kHz采样率，更适合语音识别
        self.recording = False
        
        # 独立的音频数据存储
        self.microphone_frames = []  # 麦克风音频数据
        self.system_audio_frames = []  # 系统音频数据
        self.frames = []  # 合并后的音频数据（保持兼容性）
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.start_time = None
        
        # 音频设备相关
        self.audio_devices = []
        self.selected_device_index = None
        self.microphone_device_index = None  # 麦克风设备索引
        self.system_audio_device_index = None  # 系统音频设备索引
        
        # 麦克风控制
        self.microphone_enabled = True  # 麦克风启用状态
        self.system_audio_enabled = True  # 系统音频启用状态
        self.audio_gain = 1.0  # 音频增益倍数，默认为1.0（无增益）
        
        # 音频流
        self.microphone_stream = None
        self.system_audio_stream = None
        
        # 语音识别器
        self.recognizer = sr.Recognizer()
        
        # Whisper模型
        self.whisper_model = None
        self.belle_pipeline = None  # BELLE模型管道
        self.engine_type = "google"  # 默认使用Google引擎
        self.model_type = "belle"  # 默认使用BELLE模型
        
        # 当前录音文件路径
        self.current_audio_file = None
        
        # 实时转写相关
        self.real_time_transcription = False
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        
        # 独立的音频缓冲区和转写队列
        self.microphone_buffer = []  # 麦克风音频缓冲区
        self.system_audio_buffer = []  # 系统音频缓冲区
        self.microphone_transcription_queue = queue.Queue()  # 麦克风转写队列
        self.system_audio_transcription_queue = queue.Queue()  # 系统音频转写队列
        
        # 日志系统
        self.log_queue = queue.Queue()
        self.setup_logging()
        self.transcription_thread = None
        self.microphone_transcription_thread = None  # 麦克风转写线程
        self.system_audio_transcription_thread = None  # 系统音频转写线程
        self.audio_buffer = []  # 保持兼容性
        self.buffer_duration = 5  # 每5秒进行一次转写
        self.last_transcription_time = 0
        self.last_microphone_transcription_time = 0
        self.last_system_audio_transcription_time = 0
        
        self.setup_ui()
        
        # 启动日志更新线程
        self.start_log_updater()
        
        # 初始化音频设备
        self.initialize_audio_devices()
        
    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)  # 转写结果列
        main_frame.columnconfigure(1, weight=1)  # 日志列
        main_frame.columnconfigure(2, weight=1)  # 音频源列
        main_frame.rowconfigure(3, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="录音转写工具", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 录音控制区域
        control_frame = ttk.LabelFrame(main_frame, text="录音控制", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(2, weight=1)
        
        # 录音按钮
        self.record_button = ttk.Button(control_frame, text="开始录音", command=self.toggle_recording)
        self.record_button.grid(row=0, column=0, padx=(0, 10))
        
        # 引擎选择
        ttk.Label(control_frame, text="识别引擎:").grid(row=0, column=1, padx=(0, 5), sticky=tk.W)
        self.engine_var = tk.StringVar(value="google")
        self.engine_combo = ttk.Combobox(control_frame, textvariable=self.engine_var, 
                                        values=["google", "whisper"], state="readonly", width=10)
        self.engine_combo.grid(row=0, column=2, padx=(0, 10))
        self.engine_combo.bind("<<ComboboxSelected>>", self.on_engine_change)
        
        # 实时转写开关
        self.realtime_var = tk.BooleanVar(value=True)
        self.realtime_checkbox = ttk.Checkbutton(control_frame, text="实时转写", variable=self.realtime_var)
        self.realtime_checkbox.grid(row=0, column=3, padx=(0, 10))
        
        # 麦克风开关
        self.microphone_var = tk.BooleanVar(value=self.microphone_enabled)
        self.microphone_checkbox = ttk.Checkbutton(control_frame, text="麦克风", variable=self.microphone_var, command=self.toggle_microphone)
        self.microphone_checkbox.grid(row=0, column=4, padx=(0, 10))
        
        # 系统音频开关
        self.system_audio_var = tk.BooleanVar(value=self.system_audio_enabled)
        self.system_audio_checkbox = ttk.Checkbutton(control_frame, text="系统音频", variable=self.system_audio_var, command=self.toggle_system_audio)
        self.system_audio_checkbox.grid(row=0, column=5, padx=(0, 10))
        
        # 音频增益控制
        ttk.Label(control_frame, text="增益:").grid(row=0, column=6, padx=(0, 5), sticky=tk.W)
        self.gain_var = tk.DoubleVar(value=1.0)
        self.gain_scale = ttk.Scale(control_frame, from_=0.1, to=5.0, variable=self.gain_var, 
                                   orient=tk.HORIZONTAL, length=80, command=self.on_gain_change)
        self.gain_scale.grid(row=0, column=7, padx=(0, 5))
        self.gain_label = ttk.Label(control_frame, text="1.0x")
        self.gain_label.grid(row=0, column=8, padx=(0, 10))
        
        # 录音状态标签
        self.status_label = ttk.Label(control_frame, text="准备就绪")
        self.status_label.grid(row=0, column=9, sticky=tk.W)
        
        # 录音时长标签
        self.duration_label = ttk.Label(control_frame, text="时长: 00:00")
        self.duration_label.grid(row=0, column=9)
        
        # 文件操作区域
        file_frame = ttk.LabelFrame(main_frame, text="文件操作", padding="10")
        file_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 打开音频文件按钮
        ttk.Button(file_frame, text="打开音频文件", command=self.open_audio_file).grid(row=0, column=0, padx=(0, 10))
        
        # 转写按钮
        self.transcribe_button = ttk.Button(file_frame, text="开始转写", command=self.transcribe_audio, state="disabled")
        self.transcribe_button.grid(row=0, column=1, padx=(0, 10))
        
        # 保存文本按钮
        self.save_button = ttk.Button(file_frame, text="保存文本", command=self.save_text, state="disabled")
        self.save_button.grid(row=0, column=2, padx=(0, 10))
        
        # 清空文本按钮
        ttk.Button(file_frame, text="清空文本", command=self.clear_text).grid(row=0, column=3)
        
        # 三列主内容区域
        # 转写结果区域（左列）
        result_frame = ttk.LabelFrame(main_frame, text="转写结果", padding="10")
        result_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(1, weight=1)
        
        # 实时转写状态
        self.realtime_status = ttk.Label(result_frame, text="实时转写: 未启动", font=("Arial", 9))
        self.realtime_status.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # 文本显示区域
        self.text_area = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, height=20)
        self.text_area.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 日志区域（中列）
        log_frame = ttk.LabelFrame(main_frame, text="执行日志", padding="10")
        log_frame.grid(row=3, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 5))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)
        
        # 日志控制按钮
        log_control_frame = ttk.Frame(log_frame)
        log_control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Button(log_control_frame, text="清空日志", command=self.clear_log).grid(row=0, column=0, padx=(0, 5))
        
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_control_frame, text="自动滚动", variable=self.auto_scroll_var).grid(row=0, column=1)
        
        # 日志显示区域
        self.log_area = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20, font=("Consolas", 9))
        self.log_area.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.log_area.config(state=tk.DISABLED)  # 设置为只读
        

        
        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 10))
        
        # 状态栏
        self.status_bar = ttk.Label(main_frame, text="就绪", relief=tk.SUNKEN)
        self.status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))
    
    def setup_logging(self):
        """设置日志系统"""
        # 创建自定义日志处理器
        self.log_handler = QueueLogHandler(self.log_queue)
        self.log_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                    datefmt='%H:%M:%S')
        self.log_handler.setFormatter(formatter)
        
        # 获取根日志记录器并添加处理器
        self.logger = logging.getLogger('AudioTranscriber')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.log_handler)
        
        # 记录初始化日志
        self.logger.info("音频转写工具已启动")
    
    def start_log_updater(self):
        """启动日志更新线程"""
        def update_logs():
            while True:
                try:
                    # 从队列中获取日志记录
                    record = self.log_queue.get(timeout=1)
                    if record is None:
                        break
                    
                    # 在主线程中更新UI
                    self.root.after(0, self.append_log, record)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"日志更新线程错误: {e}")
        
        self.log_thread = threading.Thread(target=update_logs, daemon=True)
        self.log_thread.start()
    
    def initialize_audio_devices(self):
        """初始化音频设备列表"""
        try:
            device_count = self.audio.get_device_count()
            self.audio_devices = []
            
            for i in range(device_count):
                try:
                    device_info = self.audio.get_device_info_by_index(i)
                    if device_info['maxInputChannels'] > 0:  # 只考虑输入设备
                        self.audio_devices.append({
                            'index': i,
                            'name': device_info['name'],
                            'channels': device_info['maxInputChannels'],
                            'sample_rate': int(device_info['defaultSampleRate'])
                        })
                        
                        # 尝试识别麦克风和立体声混音设备
                        device_name_lower = device_info['name'].lower()
                        if any(keyword in device_name_lower for keyword in ['麦克风', 'microphone', 'mic']):
                            if self.microphone_device_index is None:
                                self.microphone_device_index = i
                                self.log_info(f"检测到麦克风设备: {device_info['name']}")
                        elif any(keyword in device_name_lower for keyword in ['立体声混音', 'stereo mix', 'what u hear', 'loopback']):
                            if self.system_audio_device_index is None:
                                self.system_audio_device_index = i
                                self.log_info(f"检测到系统音频设备: {device_info['name']}")
                                
                except Exception as e:
                    self.log_warning(f"无法获取设备 {i} 的信息: {e}")
                    
            # 如果没有找到特定设备，使用默认设备
            if self.microphone_device_index is None:
                try:
                    default_device = self.audio.get_default_input_device_info()
                    self.microphone_device_index = default_device['index']
                    self.log_info(f"使用默认输入设备作为麦克风: {default_device['name']}")
                except Exception as e:
                    self.log_error(f"无法获取默认输入设备: {e}")
                    
        except Exception as e:
            self.log_error(f"初始化音频设备失败: {e}")
    
    def refresh_audio_devices(self):
        """刷新音频设备列表"""
        # 音频源控制功能已移除
        pass
    
    def update_devices_display(self):
        """更新设备显示界面"""
        # 音频源控制功能已移除
        pass
    
    def toggle_device_enabled(self, device_index, enabled):
        """切换设备启用状态"""
        # 音频源控制功能已移除
        pass
    
    def toggle_microphone_enabled(self, enabled):
        """切换麦克风启用状态"""
        # 音频源控制功能已移除
        pass
    
    def toggle_device_audio_enabled(self, enabled):
        """切换设备音频启用状态"""
        # 音频源控制功能已移除
        pass
    
    def update_microphone_status(self):
        """更新麦克风状态指示"""
        # 音频源控制功能已移除
        pass
    
    def update_device_audio_status(self):
        """更新设备音频状态指示"""
        # 音频源控制功能已移除
        pass
    
    def update_device_details(self):
        """更新设备详情显示"""
        # 音频源控制功能已移除
        pass
    
    def start_device_monitoring(self):
        """开始监控音频设备"""
        # 音频源控制功能已移除
        pass

    def stop_device_monitoring(self):
        """停止监控音频设备"""
        # 音频源控制功能已移除
        pass
    
    def monitor_device(self, device_index):
        """监控单个设备的音频输入"""
        # 音频源控制功能已移除
        pass
    
    def update_device_status_display(self, device_index):
        """更新设备状态显示"""
        # 音频源控制功能已移除
        pass
    
    def append_log(self, log_record):
        """在日志区域添加日志消息"""
        try:
            # 格式化日志消息
            formatted_message = self.log_handler.format(log_record)
            
            # 同时在终端中打印日志
            print(formatted_message)
            
            # 更新日志显示区域
            self.log_area.config(state=tk.NORMAL)
            self.log_area.insert(tk.END, formatted_message + "\n")
            
            # 限制日志行数（保留最近500行），减少内存使用
            line_count = int(self.log_area.index('end-1c').split('.')[0])
            if line_count > 500:
                # 删除前面的行，保留最近的500行
                delete_lines = line_count - 500
                self.log_area.delete("1.0", f"{delete_lines + 1}.0")
            
            # 自动滚动到底部
            if self.auto_scroll_var.get():
                self.log_area.see(tk.END)
            
            self.log_area.config(state=tk.DISABLED)
        except Exception as e:
            print(f"日志显示错误: {e}")
    
    def clear_log(self):
        """清空日志"""
        self.log_area.config(state=tk.NORMAL)
        self.log_area.delete("1.0", tk.END)
        self.log_area.config(state=tk.DISABLED)
        self.logger.info("日志已清空")
    
    def log_info(self, message):
        """记录信息日志"""
        self.logger.info(message)
    
    def log_warning(self, message):
        """记录警告日志"""
        self.logger.warning(message)
    
    def log_error(self, message):
        """记录错误日志"""
        self.logger.error(message)
        
    def toggle_microphone(self):
        """切换麦克风启用状态"""
        self.microphone_enabled = self.microphone_var.get()
        status = "启用" if self.microphone_enabled else "禁用"
        self.log_info(f"麦克风已{status}")
        
        # 如果正在录音且麦克风被禁用，提示用户
        if self.recording and not self.microphone_enabled:
            self.log_warning("麦克风已禁用，录音将继续但不会接收麦克风音频")
    
    def toggle_system_audio(self):
        """切换系统音频启用状态"""
        self.system_audio_enabled = self.system_audio_var.get()
        status = "启用" if self.system_audio_enabled else "禁用"
        self.log_info(f"系统音频已{status}")
        
        # 如果正在录音，提示用户
        if self.recording:
            if self.system_audio_enabled:
                self.log_info("系统音频重新启用，开始录制系统声音")
            else:
                self.log_warning("系统音频已禁用，录音将继续但不会接收系统音频")
    
    def on_gain_change(self, value):
        """音频增益变化回调"""
        self.audio_gain = float(value)
        self.gain_label.config(text=f"{self.audio_gain:.1f}x")
        if self.recording:
            self.log_info(f"音频增益已调整为 {self.audio_gain:.1f}x")
            
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        try:
            # 获取默认音频设备信息（用于配置音频流）
            try:
                default_device_info = self.audio.get_default_input_device_info()
                selected_device = {
                    'index': default_device_info['index'],
                    'name': default_device_info['name']
                }
            except Exception as e:
                self.log_error(f"无法获取默认音频设备: {e}")
                messagebox.showerror("错误", "无法获取默认音频设备，请检查音频设备配置")
                return
            
            self.log_info("开始录音...")
            self.recording = True
            
            # 清空所有音频数据和缓冲区，防止内存累积
            self.frames = []
            self.microphone_frames = []
            self.system_audio_frames = []
            self.audio_buffer = []
            self.microphone_buffer = []
            self.system_audio_buffer = []
            
            # 清空所有转写队列
            if hasattr(self, 'transcription_queue'):
                while not self.transcription_queue.empty():
                    try:
                        self.transcription_queue.get_nowait()
                    except queue.Empty:
                        break
            if hasattr(self, 'microphone_transcription_queue'):
                while not self.microphone_transcription_queue.empty():
                    try:
                        self.microphone_transcription_queue.get_nowait()
                    except queue.Empty:
                        break
            if hasattr(self, 'system_audio_transcription_queue'):
                while not self.system_audio_transcription_queue.empty():
                    try:
                        self.system_audio_transcription_queue.get_nowait()
                    except queue.Empty:
                        break
            
            # 强制垃圾回收
            gc.collect()
            
            # 获取内存使用情况
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self.log_info(f"音频缓冲区已清空，开始新的录音会话，当前内存使用: {memory_mb:.1f}MB")
            self.record_button.config(text="停止录音")
            self.status_label.config(text="正在录音...")
            self.status_bar.config(text="录音中...")
            
            # 检查是否启用实时转写
            self.real_time_transcription = self.realtime_var.get()
            if self.real_time_transcription:
                self.realtime_status.config(text="实时转写: 启动中...")
                # 启动独立的转写线程
                if self.microphone_enabled:
                    self.microphone_transcription_thread = threading.Thread(target=self.real_time_transcribe_microphone)
                    self.microphone_transcription_thread.daemon = True
                    self.microphone_transcription_thread.start()
                    self.log_info(f"麦克风实时转写已启动，使用引擎: {self.engine_type}")
                
                if self.system_audio_enabled:
                    self.system_audio_transcription_thread = threading.Thread(target=self.real_time_transcribe_system_audio)
                    self.system_audio_transcription_thread.daemon = True
                    self.system_audio_transcription_thread.start()
                    self.log_info(f"系统音频实时转写已启动，使用引擎: {self.engine_type}")
                
                # 保持原有的转写线程作为兼容
                self.transcription_thread = threading.Thread(target=self.real_time_transcribe)
                self.transcription_thread.daemon = True
                self.transcription_thread.start()
            else:
                self.realtime_status.config(text="实时转写: 未启动")
            
            # 开始录音线程
            self.record_thread = threading.Thread(target=self.record_audio)
            self.record_thread.daemon = True
            self.record_thread.start()
            
            # 开始计时线程
            self.timer_thread = threading.Thread(target=self.update_timer)
            self.timer_thread.daemon = True
            self.timer_thread.start()
            
        except Exception as e:
            self.log_error(f"录音启动失败: {str(e)}")
            messagebox.showerror("错误", f"录音启动失败: {str(e)}")
            self.recording = False
            
    def record_audio(self):
        try:
            # 初始化音频流
            streams_info = []
            
            # 设置麦克风流
            if self.microphone_enabled and self.microphone_device_index is not None:
                try:
                    self.microphone_stream = self.audio.open(
                        format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        input_device_index=self.microphone_device_index,
                        frames_per_buffer=self.chunk
                    )
                    mic_device_info = self.audio.get_device_info_by_index(self.microphone_device_index)
                    streams_info.append(f"麦克风: {mic_device_info['name']}")
                    self.log_info(f"麦克风流已启动: {mic_device_info['name']}")
                except Exception as e:
                    self.log_error(f"无法启动麦克风流: {e}")
                    self.microphone_stream = None
            
            # 设置系统音频流
            if self.system_audio_enabled and self.system_audio_device_index is not None:
                try:
                    self.system_audio_stream = self.audio.open(
                        format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        input_device_index=self.system_audio_device_index,
                        frames_per_buffer=self.chunk
                    )
                    sys_device_info = self.audio.get_device_info_by_index(self.system_audio_device_index)
                    streams_info.append(f"系统音频: {sys_device_info['name']}")
                    self.log_info(f"系统音频流已启动: {sys_device_info['name']}")
                except Exception as e:
                    self.log_error(f"无法启动系统音频流: {e}")
                    self.system_audio_stream = None
            
            # 如果没有可用的音频流，使用默认设备
            if self.microphone_stream is None and self.system_audio_stream is None:
                try:
                    default_device_info = self.audio.get_default_input_device_info()
                    self.stream = self.audio.open(
                        format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        input_device_index=default_device_info['index'],
                        frames_per_buffer=self.chunk
                    )
                    streams_info.append(f"默认设备: {default_device_info['name']}")
                    self.log_info(f"使用默认音频设备: {default_device_info['name']}")
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("错误", f"无法获取音频设备: {e}"))
                    return
            
            self.log_info(f"音频流配置: {self.rate}Hz, {self.channels}声道, 缓冲区: {self.chunk}, 活跃流: {', '.join(streams_info)}")
            
            self.start_time = time.time()
            self.last_transcription_time = self.start_time
            
            while self.recording:
                try:
                    # 初始化独立的音频数据
                    mic_data = None
                    sys_data = None
                    mixed_data = np.zeros(self.chunk, dtype=np.int16)
                    has_audio = False
                    
                    # 读取麦克风数据
                    if self.microphone_enabled and self.microphone_stream is not None:
                        try:
                            mic_data = self.microphone_stream.read(self.chunk, exception_on_overflow=False)
                            mic_array = np.frombuffer(mic_data, dtype=np.int16)
                            # 应用增益
                            if self.audio_gain != 1.0:
                                mic_array = mic_array.astype(np.float32) * self.audio_gain
                                mic_array = np.clip(mic_array, -32768, 32767).astype(np.int16)
                            
                            # 存储独立的麦克风数据
                            self.microphone_frames.append(mic_array.tobytes())
                            mixed_data = mixed_data + mic_array
                            has_audio = True
                        except Exception as e:
                            self.log_warning(f"麦克风读取错误: {e}")
                            # 添加静音数据保持同步
                            silent_mic_data = b'\x00' * (self.chunk * 2)
                            self.microphone_frames.append(silent_mic_data)
                    
                    # 读取系统音频数据
                    if self.system_audio_enabled and self.system_audio_stream is not None:
                        try:
                            sys_data = self.system_audio_stream.read(self.chunk, exception_on_overflow=False)
                            sys_array = np.frombuffer(sys_data, dtype=np.int16)
                            # 应用增益
                            if self.audio_gain != 1.0:
                                sys_array = sys_array.astype(np.float32) * self.audio_gain
                                sys_array = np.clip(sys_array, -32768, 32767).astype(np.int16)
                            
                            # 存储独立的系统音频数据
                            self.system_audio_frames.append(sys_array.tobytes())
                            mixed_data = mixed_data + sys_array
                            has_audio = True
                        except Exception as e:
                            self.log_warning(f"系统音频读取错误: {e}")
                            # 添加静音数据保持同步
                            silent_sys_data = b'\x00' * (self.chunk * 2)
                            self.system_audio_frames.append(silent_sys_data)
                    
                    # 如果使用默认设备
                    if hasattr(self, 'stream') and self.stream is not None:
                        try:
                            data = self.stream.read(self.chunk, exception_on_overflow=False)
                            if self.microphone_enabled:  # 只有在麦克风启用时才使用默认设备数据
                                default_array = np.frombuffer(data, dtype=np.int16)
                                if self.audio_gain != 1.0:
                                    default_array = default_array.astype(np.float32) * self.audio_gain
                                    default_array = np.clip(default_array, -32768, 32767).astype(np.int16)
                                
                                # 如果没有独立的麦克风流，将默认设备数据作为麦克风数据
                                if self.microphone_stream is None:
                                    self.microphone_frames.append(default_array.tobytes())
                                
                                mixed_data = mixed_data + default_array
                                has_audio = True
                        except Exception as e:
                            self.log_warning(f"默认设备读取错误: {e}")
                    
                    # 处理混合后的音频数据
                    if has_audio:
                        # 防止溢出，限制在int16范围内
                        mixed_data = np.clip(mixed_data, -32768, 32767)
                        final_data = mixed_data.tobytes()
                    else:
                        # 如果没有音频数据，使用静音
                        final_data = b'\x00' * (self.chunk * 2)
                        # 为独立流也添加静音数据保持同步
                        if self.microphone_enabled and len(self.microphone_frames) == len(self.system_audio_frames):
                            self.microphone_frames.append(final_data)
                        if self.system_audio_enabled and len(self.system_audio_frames) == len(self.microphone_frames):
                            self.system_audio_frames.append(final_data)
                    
                    self.frames.append(final_data)
                    
                    # 独立的实时转写处理
                    current_time = time.time()
                    
                    # 麦克风实时转写
                    if self.real_time_transcription and self.microphone_enabled and mic_data:
                        self.microphone_buffer.append(mic_data)
                        
                        # 限制缓冲区大小
                        max_buffer_size = self.rate * self.buffer_duration * 2
                        if len(self.microphone_buffer) * self.chunk * 2 > max_buffer_size:
                            self.microphone_buffer.pop(0)
                        
                        if current_time - self.last_microphone_transcription_time >= self.buffer_duration:
                            if self.microphone_buffer and self.microphone_transcription_queue.qsize() < 5:
                                buffer_copy = self.microphone_buffer.copy()
                                self.microphone_transcription_queue.put(buffer_copy)
                                self.microphone_buffer.clear()
                                self.last_microphone_transcription_time = current_time
                    
                    # 系统音频实时转写
                    if self.real_time_transcription and self.system_audio_enabled and sys_data:
                        self.system_audio_buffer.append(sys_data)
                        
                        # 限制缓冲区大小
                        max_buffer_size = self.rate * self.buffer_duration * 2
                        if len(self.system_audio_buffer) * self.chunk * 2 > max_buffer_size:
                            self.system_audio_buffer.pop(0)
                        
                        if current_time - self.last_system_audio_transcription_time >= self.buffer_duration:
                            if self.system_audio_buffer and self.system_audio_transcription_queue.qsize() < 5:
                                buffer_copy = self.system_audio_buffer.copy()
                                self.system_audio_transcription_queue.put(buffer_copy)
                                self.system_audio_buffer.clear()
                                self.last_system_audio_transcription_time = current_time
                    
                    # 保持原有的转写处理（兼容性）
                    if self.real_time_transcription and has_audio:
                        self.audio_buffer.append(final_data)
                        
                        # 限制音频缓冲区大小，防止内存溢出
                        max_buffer_size = self.rate * self.buffer_duration * 2
                        if len(self.audio_buffer) * self.chunk * 2 > max_buffer_size:
                            self.audio_buffer.pop(0)
                        
                        if current_time - self.last_transcription_time >= self.buffer_duration:
                            if self.audio_buffer:
                                if self.transcription_queue.qsize() < 5:
                                    buffer_copy = self.audio_buffer.copy()
                                    self.transcription_queue.put(buffer_copy)
                                else:
                                    self.log_warning("转写队列已满，跳过本次转写")
                                self.audio_buffer.clear()
                                self.last_transcription_time = current_time
                                
                except Exception as e:
                    self.log_error(f"音频处理错误: {e}")
                    # 发生错误时添加静音数据保持录音连续性
                    silent_data = b'\x00' * (self.chunk * 2)
                    self.frames.append(silent_data)
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"录音过程中出错: {str(e)}"))
        finally:
            # 清理音频流
            if hasattr(self, 'microphone_stream') and self.microphone_stream is not None:
                try:
                    self.microphone_stream.stop_stream()
                    self.microphone_stream.close()
                    self.microphone_stream = None
                except:
                    pass
            if hasattr(self, 'system_audio_stream') and self.system_audio_stream is not None:
                try:
                    self.system_audio_stream.stop_stream()
                    self.system_audio_stream.close()
                    self.system_audio_stream = None
                except:
                    pass
    
    def get_selected_device(self):
        """获取当前选中的音频设备"""
        # 始终返回系统默认音频输入设备信息
        # 麦克风控制逻辑在record_audio方法中处理
        try:
            default_device_info = self.audio.get_default_input_device_info()
            return {
                'index': default_device_info['index'],
                'name': default_device_info['name'],
                'channels': default_device_info['maxInputChannels'],
                'sample_rate': int(default_device_info['defaultSampleRate']),
                'is_default': True
            }
        except Exception as e:
            self.log_error(f"获取默认音频设备失败: {e}")
            return None
            
    def update_timer(self):
        while self.recording:
            if self.start_time is not None:
                elapsed = time.time() - self.start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                time_str = f"时长: {minutes:02d}:{seconds:02d}"
                self.root.after(0, lambda: self.duration_label.config(text=time_str))
            time.sleep(1)
            
    def stop_recording(self):
        self.log_info("停止录音...")
        self.recording = False
        self.real_time_transcription = False
        
        # 清理所有音频流
        streams_closed = []
        
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                streams_closed.append("默认流")
            except Exception as e:
                self.log_warning(f"关闭默认流时出错: {e}")
        
        if hasattr(self, 'microphone_stream') and self.microphone_stream:
            try:
                self.microphone_stream.stop_stream()
                self.microphone_stream.close()
                self.microphone_stream = None
                streams_closed.append("麦克风流")
            except Exception as e:
                self.log_warning(f"关闭麦克风流时出错: {e}")
        
        if hasattr(self, 'system_audio_stream') and self.system_audio_stream:
            try:
                self.system_audio_stream.stop_stream()
                self.system_audio_stream.close()
                self.system_audio_stream = None
                streams_closed.append("系统音频流")
            except Exception as e:
                self.log_warning(f"关闭系统音频流时出错: {e}")
        
        # 清理内存中的音频数据和队列
        try:
            # 清空音频缓冲区
            if hasattr(self, 'audio_buffer'):
                self.audio_buffer.clear()
            if hasattr(self, 'microphone_buffer'):
                self.microphone_buffer.clear()
            if hasattr(self, 'system_audio_buffer'):
                self.system_audio_buffer.clear()
                
            # 清空转写队列
            if hasattr(self, 'transcription_queue'):
                while not self.transcription_queue.empty():
                    try:
                        self.transcription_queue.get_nowait()
                    except queue.Empty:
                        break
            if hasattr(self, 'microphone_transcription_queue'):
                while not self.microphone_transcription_queue.empty():
                    try:
                        self.microphone_transcription_queue.get_nowait()
                    except queue.Empty:
                        break
            if hasattr(self, 'system_audio_transcription_queue'):
                while not self.system_audio_transcription_queue.empty():
                    try:
                        self.system_audio_transcription_queue.get_nowait()
                    except queue.Empty:
                        break
                        
            # 记录内存清理信息
            frames_count = len(self.frames) if hasattr(self, 'frames') else 0
            mic_frames_count = len(self.microphone_frames) if hasattr(self, 'microphone_frames') else 0
            sys_frames_count = len(self.system_audio_frames) if hasattr(self, 'system_audio_frames') else 0
            self.log_info(f"清理内存数据: 混合音频 {frames_count} 帧, 麦克风 {mic_frames_count} 帧, 系统音频 {sys_frames_count} 帧")
            
        except Exception as e:
            self.log_warning(f"清理内存数据时出错: {e}")
        
        if streams_closed:
            self.log_info(f"音频流已关闭: {', '.join(streams_closed)}")
        else:
            self.log_info("没有活跃的音频流需要关闭")
            
        self.record_button.config(text="开始录音")
        self.status_label.config(text="录音完成")
        self.status_bar.config(text="保存录音文件...")
        self.realtime_status.config(text="实时转写: 已停止")
        
        # 保存录音文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        mic_filename = f"microphone_{timestamp}.wav"
        sys_filename = f"system_audio_{timestamp}.wav"
        
        # 确保audio目录存在
        audio_dir = os.path.join(os.getcwd(), "audio")
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
            self.log_info(f"创建音频目录: {audio_dir}")
        
        self.current_audio_file = os.path.join(audio_dir, filename)
        mic_audio_file = os.path.join(audio_dir, mic_filename)
        sys_audio_file = os.path.join(audio_dir, sys_filename)
        
        saved_files = []
        
        try:
            # 保存混合音频文件（兼容性）
            if hasattr(self, 'frames') and self.frames:
                wf = wave.open(self.current_audio_file, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.frames))
                wf.close()
                saved_files.append(("混合音频", filename, self.current_audio_file))
                self.frames.clear()
            
            # 保存独立的麦克风音频文件
            if hasattr(self, 'microphone_frames') and self.microphone_frames:
                wf_mic = wave.open(mic_audio_file, 'wb')
                wf_mic.setnchannels(self.channels)
                wf_mic.setsampwidth(self.audio.get_sample_size(self.format))
                wf_mic.setframerate(self.rate)
                wf_mic.writeframes(b''.join(self.microphone_frames))
                wf_mic.close()
                saved_files.append(("麦克风", mic_filename, mic_audio_file))
                self.microphone_frames.clear()
            
            # 保存独立的系统音频文件
            if hasattr(self, 'system_audio_frames') and self.system_audio_frames:
                wf_sys = wave.open(sys_audio_file, 'wb')
                wf_sys.setnchannels(self.channels)
                wf_sys.setsampwidth(self.audio.get_sample_size(self.format))
                wf_sys.setframerate(self.rate)
                wf_sys.writeframes(b''.join(self.system_audio_frames))
                wf_sys.close()
                saved_files.append(("系统音频", sys_filename, sys_audio_file))
                self.system_audio_frames.clear()
            
            # 如果同时有麦克风和系统音频，创建合并文件
            if len(saved_files) >= 2 and any("麦克风" in item[0] for item in saved_files) and any("系统音频" in item[0] for item in saved_files):
                try:
                    self.merge_audio_files(mic_audio_file, sys_audio_file, self.current_audio_file, timestamp)
                    saved_files.append(("合并音频", filename, self.current_audio_file))
                except Exception as e:
                    self.log_warning(f"合并音频文件失败: {e}")
            
            # 强制垃圾回收释放内存
            gc.collect()
            
            # 获取内存使用情况
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self.log_info(f"音频帧数据已清空，内存使用: {memory_mb:.1f}MB")
            
            self.transcribe_button.config(state="normal")
            
            # 记录保存的文件信息
            if saved_files:
                duration = time.time() - self.start_time if self.start_time else 0
                self.log_info(f"录音完成，时长: {duration:.1f}秒")
                
                for file_type, file_name, file_path in saved_files:
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path) / 1024  # KB
                        self.log_info(f"{file_type}文件已保存: {file_name}, 大小: {file_size:.1f}KB")
                
                self.status_bar.config(text=f"录音已保存: {len(saved_files)} 个文件")
            else:
                self.status_bar.config(text="录音完成，但没有音频数据")
                self.log_warning("录音完成，但没有检测到音频数据")
            
        except Exception as e:
            self.log_error(f"保存录音文件失败: {str(e)}")
            messagebox.showerror("错误", f"保存录音文件失败: {str(e)}")
            # 即使保存失败也要清空frames避免内存泄漏
            if hasattr(self, 'frames'):
                self.frames.clear()
            if hasattr(self, 'microphone_frames'):
                self.microphone_frames.clear()
            if hasattr(self, 'system_audio_frames'):
                self.system_audio_frames.clear()
            self.log_info("清空音频帧数据以释放内存")
            
    def merge_audio_files(self, mic_file, sys_file, output_file, timestamp):
        """合并麦克风和系统音频文件"""
        try:
            self.log_info("开始合并音频文件...")
            
            # 使用pydub加载音频文件
            mic_audio = AudioSegment.from_wav(mic_file)
            sys_audio = AudioSegment.from_wav(sys_file)
            
            # 确保两个音频文件长度一致
            min_length = min(len(mic_audio), len(sys_audio))
            mic_audio = mic_audio[:min_length]
            sys_audio = sys_audio[:min_length]
            
            # 合并音频（叠加）
            merged_audio = mic_audio.overlay(sys_audio)
            
            # 导出合并后的音频
            merged_audio.export(output_file, format="wav")
            
            self.log_info(f"音频合并完成: {os.path.basename(output_file)}")
            
        except Exception as e:
            self.log_error(f"音频合并失败: {e}")
            # 如果合并失败，尝试简单复制麦克风文件作为备选
            try:
                import shutil
                shutil.copy2(mic_file, output_file)
                self.log_info(f"合并失败，已复制麦克风文件作为主文件: {os.path.basename(output_file)}")
            except Exception as copy_error:
                self.log_error(f"复制备选文件也失败: {copy_error}")
                raise e
    
    def open_audio_file(self):
        # 设置默认目录为audio子目录
        audio_dir = os.path.join(os.getcwd(), "audio")
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
            
        file_path = filedialog.askopenfilename(
            title="选择音频文件",
            initialdir=audio_dir,
            filetypes=[("音频文件", "*.wav *.mp3 *.flac *.m4a"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.log_info(f"选择音频文件: {os.path.basename(file_path)}")
            self.current_audio_file = file_path
            self.transcribe_button.config(state="normal")
            self.status_bar.config(text=f"已选择文件: {os.path.basename(file_path)}")
            self.log_info(f"音频文件加载成功: {os.path.basename(file_path)}")
            
    def prepare_audio_file(self, audio_file_path):
        """准备音频文件，如果需要则转换格式"""
        try:
            # 首先尝试直接使用原文件
            with sr.AudioFile(audio_file_path) as source:
                pass  # 如果能成功打开，说明格式兼容
            return audio_file_path
        except Exception:
            # 如果无法直接使用，尝试转换格式
            file_ext = os.path.splitext(audio_file_path)[1].lower()
            
            # 检查文件格式
            if file_ext in ['.wav', '.flac']:
                # WAV和FLAC格式应该直接支持，如果失败可能是文件损坏
                raise Exception(f"音频文件可能已损坏或格式不正确: {audio_file_path}")
            
            # 对于需要ffmpeg的格式（MP3, M4A等）
            try:
                self.root.after(0, lambda: self.status_bar.config(text="正在转换音频格式..."))
                
                # 尝试使用pydub加载音频文件
                try:
                    audio = AudioSegment.from_file(audio_file_path)
                except Exception as e:
                    if "ffmpeg" in str(e).lower() or "找不到指定的文件" in str(e):
                        raise Exception(f"需要安装ffmpeg来处理{file_ext}格式的音频文件。\n\n解决方案：\n1. 下载ffmpeg: https://ffmpeg.org/download.html\n2. 解压后将bin目录添加到系统PATH\n3. 或者将音频文件转换为WAV格式后重试")
                    else:
                        raise Exception(f"无法读取音频文件: {str(e)}")
                
                # 转换为WAV格式，设置为单声道，16位，16kHz
                audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
                
                # 创建临时文件
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.close()
                
                # 导出为WAV格式
                audio.export(temp_file.name, format="wav")
                
                return temp_file.name
                
            except Exception as e:
                if "ffmpeg" in str(e) or "需要安装ffmpeg" in str(e):
                    raise e  # 直接传递ffmpeg相关的错误
                else:
                    raise Exception(f"音频格式转换失败: {str(e)}")
    
    def transcribe_audio(self):
        if not self.current_audio_file or not os.path.exists(self.current_audio_file):
            messagebox.showerror("错误", "请先录音或选择音频文件")
            return
            
        # 在新线程中进行转写
        self.transcribe_thread = threading.Thread(target=self.perform_transcription)
        self.transcribe_thread.daemon = True
        self.transcribe_thread.start()
        
    def perform_transcription(self):
        try:
            self.root.after(0, lambda: self.progress.start())
            self.root.after(0, lambda: self.status_bar.config(text="正在转写音频..."))
            self.root.after(0, lambda: self.transcribe_button.config(state="disabled"))
            
            # 准备音频文件
            audio_file_to_use = self.prepare_audio_file(self.current_audio_file)
            
            file_size = os.path.getsize(audio_file_to_use) / 1024  # KB
            self.log_info(f"开始转写音频文件，引擎: {self.engine_type}, 文件大小: {file_size:.1f}KB")
            
            start_time = time.time()
            
            # 根据选择的引擎进行转写
            text = ""
            if self.engine_type == "whisper":
                try:
                    self.log_info("使用Whisper引擎进行转写...")
                    text = self.transcribe_with_whisper(audio_file_to_use)
                    self.root.after(0, lambda: self.status_bar.config(text="转写完成（使用Whisper引擎）"))
                except Exception as e:
                    self.log_error(f"Whisper转写失败: {str(e)}")
                    text = f"Whisper转写失败: {str(e)}"
                    self.root.after(0, lambda: self.status_bar.config(text="Whisper转写失败"))
            else:
                # 使用Google语音识别
                try:
                    self.log_info("使用Google引擎进行转写...")
                    with sr.AudioFile(audio_file_to_use) as source:
                        # 调整环境噪音
                        self.recognizer.adjust_for_ambient_noise(source)
                        audio_data = self.recognizer.record(source)
                        
                    # 使用Google识别（需要网络）
                    text = self.recognizer.recognize_google(audio_data, language='zh-CN')
                    self.root.after(0, lambda: self.status_bar.config(text="转写完成（使用Google引擎）"))
                except sr.RequestError:
                    self.log_warning("Google引擎连接失败，尝试离线引擎...")
                    try:
                        # 如果Google不可用，尝试使用离线识别
                        text = self.recognizer.recognize_sphinx(audio_data, language='zh-CN')
                        self.root.after(0, lambda: self.status_bar.config(text="转写完成（使用离线引擎）"))
                    except:
                        text = "转写失败：无法连接到识别服务，且离线引擎不可用"
                        self.log_error(text)
                except sr.UnknownValueError:
                    text = "无法识别音频内容，请确保音频清晰且包含语音"
                    self.log_warning(text)
                except Exception as e:
                    text = f"转写过程中出现错误: {str(e)}"
                    self.log_error(text)
                
            end_time = time.time()
            duration = end_time - start_time
            word_count = len(text.replace(' ', ''))
            self.log_info(f"转写完成，耗时: {duration:.1f}秒, 识别文字: {word_count}字")
            
            # 更新UI
            self.root.after(0, lambda: self.update_transcription_result(text))
            
        except Exception as e:
            error_msg = f"转写失败: {str(e)}"
            self.log_error(error_msg)
            self.root.after(0, lambda: self.update_transcription_result(error_msg))
        finally:
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.transcribe_button.config(state="normal"))
            
    def filter_unwanted_text(self, text):
        """过滤异常的推广文本"""
        if not text:
            return text
            
        # 定义需要过滤的异常文本模式
        unwanted_patterns = [
            "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
            "请不吝点赞",
            "订阅 转发 打赏支持明镜与点点栏目",
            "明镜与点点栏目",
            "点赞 订阅 转发 打赏",
            "支持明镜与点点"
        ]
        
        filtered_text = text
        for pattern in unwanted_patterns:
            if pattern in filtered_text:
                filtered_text = filtered_text.replace(pattern, "")
                self.log_info(f"已过滤异常文本: {pattern}")
        
        # 清理多余的空格和换行
        filtered_text = " ".join(filtered_text.split())
        
        return filtered_text
    
    def update_transcription_result(self, text):
        # 过滤异常文本
        filtered_text = self.filter_unwanted_text(text)
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(1.0, filtered_text)
        self.save_button.config(state="normal")
        
    def save_text(self):
        text = self.text_area.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("警告", "没有文本可保存")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="保存文本文件",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.status_bar.config(text=f"文本已保存: {os.path.basename(file_path)}")
                messagebox.showinfo("成功", "文本保存成功")
            except Exception as e:
                messagebox.showerror("错误", f"保存文件失败: {str(e)}")
                
    def clear_text(self):
        self.text_area.delete(1.0, tk.END)
        self.save_button.config(state="disabled")
        self.status_bar.config(text="文本已清空")
        
    def real_time_transcribe(self):
        """实时转写线程函数"""
        self.root.after(0, lambda: self.realtime_status.config(text="实时转写: 运行中"))
        self.log_info(f"实时转写线程启动，使用引擎: {self.engine_type}")
        
        transcription_count = 0
        
        while self.real_time_transcription and self.recording:
            try:
                # 从队列中获取音频数据
                if not self.transcription_queue.empty():
                    audio_data = self.transcription_queue.get(timeout=1)
                    transcription_count += 1
                    
                    # 将音频数据转换为可识别的格式
                    audio_bytes = b''.join(audio_data)
                    
                    # 创建临时WAV文件
                    temp_file_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                            temp_file_path = temp_file.name
                            wf = wave.open(temp_file_path, 'wb')
                            wf.setnchannels(self.channels)
                            wf.setsampwidth(self.audio.get_sample_size(self.format))
                            wf.setframerate(self.rate)
                            wf.writeframes(audio_bytes)
                            wf.close()
                        
                        # 进行语音识别
                        text = ""
                        if self.engine_type == "whisper":
                            try:
                                text = self.transcribe_with_whisper(temp_file_path)
                            except Exception as e:
                                # Whisper识别失败，记录日志但继续
                                if transcription_count % 10 == 1:  # 每10次记录一次错误，避免日志过多
                                    self.log_warning(f"实时Whisper转写失败: {str(e)}")
                        else:
                            # 使用Google引擎
                            try:
                                with sr.AudioFile(temp_file_path) as source:
                                    audio_for_recognition = self.recognizer.record(source)
                                    
                                try:
                                    text = self.recognizer.recognize_google(audio_for_recognition, language='zh-CN')
                                except sr.UnknownValueError:
                                    # 无法识别的音频，忽略
                                    pass
                                except sr.RequestError as e:
                                    # 网络错误，记录日志但继续
                                    if transcription_count % 10 == 1:
                                        self.log_warning(f"实时Google转写网络错误: {str(e)}")
                            except Exception as e:
                                if transcription_count % 10 == 1:
                                    self.log_error(f"音频文件处理错误: {str(e)}")
                        
                        if text and text.strip():  # 只有当识别到文本时才更新
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            formatted_text = f"[{timestamp}] {text}\n"
                            self.root.after(0, lambda t=formatted_text: self.append_realtime_text(t))
                            self.log_info(f"实时转写成功 #{transcription_count}: {text[:50]}{'...' if len(text) > 50 else ''}")
                            
                    except Exception as e:
                        # 处理音频文件时出错，记录日志但继续
                        if transcription_count % 10 == 1:
                            self.log_error(f"实时转写处理错误: {str(e)}")
                    finally:
                        # 确保临时文件被清理
                        if temp_file_path and os.path.exists(temp_file_path):
                            try:
                                os.unlink(temp_file_path)
                            except Exception as e:
                                if transcription_count % 20 == 1:  # 减少清理错误日志频率
                                    self.log_warning(f"清理临时文件失败: {str(e)}")
                                
                else:
                    time.sleep(0.1)  # 短暂等待
                    
            except queue.Empty:
                continue
            except Exception as e:
                # 处理其他异常
                self.log_error(f"实时转写线程异常: {str(e)}")
                continue
                
        self.root.after(0, lambda: self.realtime_status.config(text="实时转写: 已停止"))
        self.log_info(f"实时转写线程结束，共处理 {transcription_count} 个音频片段")
        
    def microphone_transcribe(self):
        """麦克风音频实时转写线程函数"""
        self.log_info(f"麦克风转写线程启动，使用引擎: {self.engine_type}")
        
        transcription_count = 0
        
        while self.real_time_transcription and self.recording:
            try:
                # 从队列中获取音频数据
                if not self.microphone_transcription_queue.empty():
                    audio_data = self.microphone_transcription_queue.get(timeout=1)
                    transcription_count += 1
                    
                    # 将音频数据转换为可识别的格式
                    audio_bytes = b''.join(audio_data)
                    
                    # 创建临时WAV文件
                    temp_file_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix="_mic.wav", delete=False) as temp_file:
                            temp_file_path = temp_file.name
                            wf = wave.open(temp_file_path, 'wb')
                            wf.setnchannels(self.channels)
                            wf.setsampwidth(self.audio.get_sample_size(self.format))
                            wf.setframerate(self.rate)
                            wf.writeframes(audio_bytes)
                            wf.close()
                        
                        # 进行语音识别
                        text = ""
                        if self.engine_type == "whisper":
                            try:
                                text = self.transcribe_with_whisper(temp_file_path)
                            except Exception as e:
                                if transcription_count % 10 == 1:
                                    self.log_warning(f"麦克风Whisper转写失败: {str(e)}")
                        else:
                            # 使用Google引擎
                            try:
                                with sr.AudioFile(temp_file_path) as source:
                                    audio_for_recognition = self.recognizer.record(source)
                                    
                                try:
                                    text = self.recognizer.recognize_google(audio_for_recognition, language='zh-CN')
                                except sr.UnknownValueError:
                                    pass
                                except sr.RequestError as e:
                                    if transcription_count % 10 == 1:
                                        self.log_warning(f"麦克风Google转写网络错误: {str(e)}")
                            except Exception as e:
                                if transcription_count % 10 == 1:
                                    self.log_error(f"麦克风音频文件处理错误: {str(e)}")
                        
                        if text and text.strip():
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            formatted_text = f"[{timestamp}][麦克风] {text}\n"
                            self.root.after(0, lambda t=formatted_text: self.append_realtime_text(t))
                            self.log_info(f"麦克风转写成功 #{transcription_count}: {text[:50]}{'...' if len(text) > 50 else ''}")
                            
                    except Exception as e:
                        if transcription_count % 10 == 1:
                            self.log_error(f"麦克风转写处理错误: {str(e)}")
                    finally:
                        if temp_file_path and os.path.exists(temp_file_path):
                            try:
                                os.unlink(temp_file_path)
                            except Exception as e:
                                if transcription_count % 20 == 1:
                                    self.log_warning(f"清理麦克风临时文件失败: {str(e)}")
                                
                else:
                    time.sleep(0.1)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.log_error(f"麦克风转写线程异常: {str(e)}")
                continue
                
        self.log_info(f"麦克风转写线程结束，共处理 {transcription_count} 个音频片段")
    
    def system_audio_transcribe(self):
        """系统音频实时转写线程函数"""
        self.log_info(f"系统音频转写线程启动，使用引擎: {self.engine_type}")
        
        transcription_count = 0
        
        while self.real_time_transcription and self.recording:
            try:
                # 从队列中获取音频数据
                if not self.system_audio_transcription_queue.empty():
                    audio_data = self.system_audio_transcription_queue.get(timeout=1)
                    transcription_count += 1
                    
                    # 将音频数据转换为可识别的格式
                    audio_bytes = b''.join(audio_data)
                    
                    # 创建临时WAV文件
                    temp_file_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix="_sys.wav", delete=False) as temp_file:
                            temp_file_path = temp_file.name
                            wf = wave.open(temp_file_path, 'wb')
                            wf.setnchannels(self.channels)
                            wf.setsampwidth(self.audio.get_sample_size(self.format))
                            wf.setframerate(self.rate)
                            wf.writeframes(audio_bytes)
                            wf.close()
                        
                        # 进行语音识别
                        text = ""
                        if self.engine_type == "whisper":
                            try:
                                text = self.transcribe_with_whisper(temp_file_path)
                            except Exception as e:
                                if transcription_count % 10 == 1:
                                    self.log_warning(f"系统音频Whisper转写失败: {str(e)}")
                        else:
                            # 使用Google引擎
                            try:
                                with sr.AudioFile(temp_file_path) as source:
                                    audio_for_recognition = self.recognizer.record(source)
                                    
                                try:
                                    text = self.recognizer.recognize_google(audio_for_recognition, language='zh-CN')
                                except sr.UnknownValueError:
                                    pass
                                except sr.RequestError as e:
                                    if transcription_count % 10 == 1:
                                        self.log_warning(f"系统音频Google转写网络错误: {str(e)}")
                            except Exception as e:
                                if transcription_count % 10 == 1:
                                    self.log_error(f"系统音频文件处理错误: {str(e)}")
                        
                        if text and text.strip():
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            formatted_text = f"[{timestamp}][系统音频] {text}\n"
                            self.root.after(0, lambda t=formatted_text: self.append_realtime_text(t))
                            self.log_info(f"系统音频转写成功 #{transcription_count}: {text[:50]}{'...' if len(text) > 50 else ''}")
                            
                    except Exception as e:
                        if transcription_count % 10 == 1:
                            self.log_error(f"系统音频转写处理错误: {str(e)}")
                    finally:
                        if temp_file_path and os.path.exists(temp_file_path):
                            try:
                                os.unlink(temp_file_path)
                            except Exception as e:
                                if transcription_count % 20 == 1:
                                    self.log_warning(f"清理系统音频临时文件失败: {str(e)}")
                                
                else:
                    time.sleep(0.1)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.log_error(f"系统音频转写线程异常: {str(e)}")
                continue
                
        self.log_info(f"系统音频转写线程结束，共处理 {transcription_count} 个音频片段")
        
    def append_realtime_text(self, text):
        """向文本区域追加实时转写结果"""
        # 过滤异常文本
        filtered_text = self.filter_unwanted_text(text)
        if filtered_text.strip():  # 只有过滤后还有内容才添加
            self.text_area.insert(tk.END, filtered_text)
            self.text_area.see(tk.END)  # 自动滚动到底部
            self.save_button.config(state="normal")
    
    def on_engine_change(self, event=None):
        """引擎切换事件处理"""
        self.engine_type = self.engine_var.get()
        if self.engine_type == "whisper":
            self.load_whisper_model()
        self.status_label.config(text=f"已切换到{self.engine_type}引擎")
    
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
                    self.log_info(f"检测到GPU设备: {gpu_name}，将使用GPU加速")
                else:
                    self.log_info("未检测到GPU设备，将使用CPU运行")
                
                # 优先尝试加载BELLE模型
                if TRANSFORMERS_AVAILABLE and self.model_type == "belle":
                    try:
                        self.log_info(f"开始加载BELLE-2/Belle-whisper-large-v3-turbo-zh模型，设备: {device} {gpu_info}")
                        self.status_label.config(text=f"正在下载并加载BELLE模型（首次使用需要下载）- {device.upper()}模式...")
                        self.root.update()
                        
                        start_time = time.time()
                        
                        # 加载BELLE模型
                        model_id = "BELLE-2/Belle-whisper-large-v3-turbo-zh"
                        
                        # 设置torch数据类型
                        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                        
                        # 加载模型
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
                        self.status_label.config(text=f"BELLE模型加载完成 - {device.upper()}")
                        self.log_info(f"BELLE-2/Belle-whisper-large-v3-turbo-zh模型加载成功，耗时: {load_time:.1f}秒，设备: {device}")
                        return
                        
                    except Exception as e:
                        self.log_warning(f"BELLE模型加载失败: {str(e)}")
                        self.log_info("回退到原生Whisper模型...")
                        self.model_type = "whisper"
                
                # 如果BELLE模型加载失败或不可用，使用原生Whisper模型
                self.log_info(f"开始加载原生Whisper模型，设备: {device} {gpu_info}")
                self.status_label.config(text=f"正在下载并加载Whisper模型（首次使用需要下载）- {device.upper()}模式...")
                self.root.update()
                
                # 优先使用turbo模型（最新最快模型）
                try:
                    self.log_info("尝试加载turbo模型...")
                    start_time = time.time()
                    self.whisper_model = whisper.load_model("turbo", device=device)
                    load_time = time.time() - start_time
                    self.status_label.config(text=f"Whisper模型加载完成（turbo模型 - {device.upper()}）")
                    self.log_info(f"turbo模型加载成功，耗时: {load_time:.1f}秒，设备: {device}")
                except Exception as e1:
                    self.log_warning(f"turbo模型加载失败: {str(e1)}")
                    # 如果turbo模型失败，尝试small模型
                    try:
                        self.log_info("尝试加载small模型...")
                        start_time = time.time()
                        self.whisper_model = whisper.load_model("small", device=device)
                        load_time = time.time() - start_time
                        self.status_label.config(text=f"Whisper模型加载完成（small模型 - {device.upper()}）")
                        self.log_info(f"small模型加载成功，耗时: {load_time:.1f}秒，设备: {device}")
                    except Exception as e2:
                        self.log_warning(f"small模型加载失败: {str(e2)}")
                        # 如果small模型失败，尝试base模型
                        try:
                            self.log_info("尝试加载base模型...")
                            start_time = time.time()
                            self.whisper_model = whisper.load_model("base", device=device)
                            load_time = time.time() - start_time
                            self.status_label.config(text=f"Whisper模型加载完成（base模型 - {device.upper()}）")
                            self.log_info(f"base模型加载成功，耗时: {load_time:.1f}秒，设备: {device}")
                        except Exception as e3:
                            self.log_warning(f"base模型加载失败: {str(e3)}")
                            # 最后尝试tiny模型作为备选
                            try:
                                self.log_info("尝试加载tiny模型...")
                                start_time = time.time()
                                self.whisper_model = whisper.load_model("tiny", device=device)
                                load_time = time.time() - start_time
                                self.status_label.config(text=f"Whisper模型加载完成（tiny模型 - {device.upper()} - 准确率较低）")
                                self.log_warning(f"tiny模型加载成功，耗时: {load_time:.1f}秒，设备: {device}（注意：准确率较低）")
                            except Exception as e4:
                                error_msg = f"所有模型下载失败。Turbo: {str(e1)}, Small: {str(e2)}, Base: {str(e3)}, Tiny: {str(e4)}"
                                self.log_error(error_msg)
                                raise Exception(error_msg)
                        
            except Exception as e:
                self.log_error(f"Whisper模型加载失败: {str(e)}")
                messagebox.showerror("错误", f"加载Whisper模型失败: {str(e)}\n\n建议：\n1. 检查网络连接\n2. 确保有足够的磁盘空间\n3. 安装transformers库: pip install transformers\n4. 尝试重新启动程序")
                self.engine_var.set("google")
                self.engine_type = "google"
                self.status_label.config(text="已回退到Google引擎")
                self.log_info("已回退到Google引擎")
    
    def transcribe_with_whisper(self, audio_file_path):
        """使用Whisper进行转写"""
        try:
            if self.belle_pipeline is None and self.whisper_model is None:
                self.load_whisper_model()
            
            # 优先使用BELLE模型
            if self.belle_pipeline is not None:
                self.log_info("开始BELLE模型转写，专为中文优化...")
                start_time = time.time()
                
                # 使用BELLE模型进行转写
                result = self.belle_pipeline(
                    audio_file_path,
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
                
                self.log_info(f"BELLE模型转写完成，耗时: {transcribe_time:.1f}秒")
                return text
                
            # 如果BELLE模型不可用，使用原生Whisper模型
            elif self.whisper_model is not None:
                self.log_info("开始原生Whisper转写，使用中文语言...")
                start_time = time.time()
                # 使用中文语言，不进行自动检测
                result = self.whisper_model.transcribe(
                    audio_file_path, 
                    language='zh',
                    initial_prompt="以下是普通话的句子。"
                )
                transcribe_time = time.time() - start_time
                
                detected_language = result.get('language', '未知')
                self.log_info(f"原生Whisper转写完成，耗时: {transcribe_time:.1f}秒, 检测语言: {detected_language}")
                
                return result["text"]
            else:
                raise Exception("Whisper模型未加载")
        except Exception as e:
            self.log_error(f"Whisper转写失败: {str(e)}")
            raise Exception(f"Whisper转写失败: {str(e)}")
    
    def __del__(self):
        if hasattr(self, 'audio'):
            self.audio.terminate()

def main():
    root = tk.Tk()
    app = AudioTranscriber(root)
    
    # 设置窗口关闭事件
    def on_closing():
        if hasattr(app, 'recording') and app.recording:
            app.stop_recording()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()