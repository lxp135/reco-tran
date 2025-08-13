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
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # ffmpeg会自动从系统PATH中查找，无需手动设置路径
        
        # 录音参数
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # 使用16kHz采样率，更适合语音识别
        self.recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.start_time = None
        
        # 音频设备相关
        self.audio_devices = []
        self.selected_device_index = None
        self.device_status = {}  # 存储每个设备的状态
        self.device_enabled = {}  # 存储每个设备的启用状态
        self.device_monitors = {}  # 存储设备监控线程
        self.monitoring_active = False
        
        # 语音识别器
        self.recognizer = sr.Recognizer()
        
        # Whisper模型
        self.whisper_model = None
        self.engine_type = "google"  # 默认使用Google引擎
        
        # 当前录音文件路径
        self.current_audio_file = None
        
        # 实时转写相关
        self.real_time_transcription = False
        self.audio_queue = queue.Queue()
        
        # 日志系统
        self.log_queue = queue.Queue()
        self.setup_logging()
        self.transcription_thread = None
        self.audio_buffer = []
        self.buffer_duration = 3  # 每3秒进行一次转写
        self.last_transcription_time = 0
        
        self.setup_ui()
        
        # 初始化音频设备
        self.initialize_audio_devices()
        
        # 启动日志更新线程
        self.start_log_updater()
        
    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
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
        
        # 录音状态标签
        self.status_label = ttk.Label(control_frame, text="准备就绪")
        self.status_label.grid(row=0, column=4, sticky=tk.W)
        
        # 录音时长标签
        self.duration_label = ttk.Label(control_frame, text="时长: 00:00")
        self.duration_label.grid(row=0, column=5)
        
        # 音频源监控区域
        audio_source_frame = ttk.LabelFrame(main_frame, text="音频源监控 (自动启动)", padding="10")
        audio_source_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        audio_source_frame.columnconfigure(0, weight=1)
        

        
        # 音频设备列表框架 - 使用固定布局，不使用滚动条
        self.devices_frame = ttk.Frame(audio_source_frame)
        self.devices_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        self.devices_frame.columnconfigure(0, weight=1)
        self.devices_frame.columnconfigure(1, weight=1)
        
        # 文件操作区域
        file_frame = ttk.LabelFrame(main_frame, text="文件操作", padding="10")
        file_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
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
        
        # 主内容区域（转写结果和日志）
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        content_frame.columnconfigure(0, weight=2)  # 转写结果占更多空间
        content_frame.columnconfigure(1, weight=1)  # 日志窗口
        content_frame.rowconfigure(0, weight=1)
        
        # 转写结果区域
        result_frame = ttk.LabelFrame(content_frame, text="转写结果", padding="10")
        result_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(1, weight=1)
        
        # 实时转写状态
        self.realtime_status = ttk.Label(result_frame, text="实时转写: 未启动", font=("Arial", 9))
        self.realtime_status.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # 文本显示区域
        self.text_area = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, height=15)
        self.text_area.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 日志区域
        log_frame = ttk.LabelFrame(content_frame, text="执行日志", padding="10")
        log_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)
        
        # 日志控制按钮
        log_control_frame = ttk.Frame(log_frame)
        log_control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Button(log_control_frame, text="清空日志", command=self.clear_log).grid(row=0, column=0, padx=(0, 5))
        
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_control_frame, text="自动滚动", variable=self.auto_scroll_var).grid(row=0, column=1)
        
        # 日志显示区域
        self.log_area = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15, font=("Consolas", 9))
        self.log_area.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.log_area.config(state=tk.DISABLED)  # 设置为只读
        
        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 状态栏
        self.status_bar = ttk.Label(main_frame, text="就绪", relief=tk.SUNKEN)
        self.status_bar.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E))
    
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
            self.audio_devices = []
            device_count = self.audio.get_device_count()
            
            for i in range(device_count):
                try:
                    device_info = self.audio.get_device_info_by_index(i)
                    # 只添加输入设备，并过滤扬声器和麦克风
                    if device_info['maxInputChannels'] > 0:
                        device_name = device_info['name'].lower()
                        # 过滤出扬声器和麦克风设备
                        if any(keyword in device_name for keyword in ['麦克风', 'microphone', 'mic', '扬声器', 'speaker', 'headphone', 'headset']):
                            self.audio_devices.append({
                                'index': i,
                                'name': device_info['name'],
                                'channels': device_info['maxInputChannels'],
                                'sample_rate': int(device_info['defaultSampleRate']),
                                'is_default': i == self.audio.get_default_input_device_info()['index']
                            })
                            # 初始化设备状态
                            self.device_status[i] = {'active': False, 'level': 0}
                            self.device_enabled[i] = True  # 默认启用所有设备
                except Exception as e:
                    self.logger.warning(f"获取设备 {i} 信息失败: {e}")
            
            self.logger.info(f"发现 {len(self.audio_devices)} 个音频输入设备")
            self.update_devices_display()
            
            # 默认启动监控
            if self.audio_devices:
                self.start_device_monitoring()
            
        except Exception as e:
            self.logger.error(f"初始化音频设备失败: {e}")
    
    def refresh_audio_devices(self):
        """刷新音频设备列表"""
        self.logger.info("刷新音频设备列表...")
        self.initialize_audio_devices()
    
    def update_devices_display(self):
        """更新设备显示界面"""
        # 清空现有显示
        for widget in self.devices_frame.winfo_children():
            widget.destroy()
        
        if not self.audio_devices:
            ttk.Label(self.devices_frame, text="未发现扬声器或麦克风设备").grid(row=0, column=0, columnspan=2, pady=10)
            return
        
        # 使用网格布局显示设备，每行最多2个设备
        for i, device in enumerate(self.audio_devices):
            row = i // 2
            col = i % 2
            
            # 设备容器框架
            device_container = ttk.LabelFrame(self.devices_frame, text="", padding="5")
            device_container.grid(row=row, column=col, sticky=(tk.W, tk.E), pady=2, padx=5)
            device_container.columnconfigure(1, weight=1)
            
            # 设备启用开关和状态指示灯
            control_frame = ttk.Frame(device_container)
            control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
            
            # 设备启用开关
            device_var = tk.BooleanVar(value=self.device_enabled.get(device['index'], True))
            device_check = ttk.Checkbutton(
                control_frame, 
                variable=device_var,
                command=lambda idx=device['index'], var=device_var: self.toggle_device_enabled(idx, var.get())
            )
            device_check.grid(row=0, column=0, padx=(0, 5))
            
            # 活动状态指示灯
            status_canvas = tk.Canvas(control_frame, width=12, height=12)
            status_canvas.grid(row=0, column=1, padx=(0, 5))
            
            # 绘制状态指示灯
            device_status = self.device_status.get(device['index'], {'active': False, 'level': 0})
            color = 'green' if device_status['active'] else 'gray'
            status_canvas.create_oval(2, 2, 10, 10, fill=color, outline='black')
            
            # 音量级别显示
            level_label = ttk.Label(control_frame, text=f"音量: {device_status['level']:.0f}%")
            level_label.grid(row=0, column=2, padx=(5, 0))
            
            # 设备名称
            device_name = device['name'][:30] + '...' if len(device['name']) > 30 else device['name']
            default_text = " (默认)" if device['is_default'] else ""
            device_label = ttk.Label(
                device_container, 
                text=f"{device_name}{default_text}",
                font=('TkDefaultFont', 9, 'bold')
            )
            device_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 2))
            
            # 设备详细信息
            info_label = ttk.Label(
                device_container, 
                text=f"通道: {device['channels']} | 采样率: {device['sample_rate']}Hz",
                font=('TkDefaultFont', 8)
            )
            info_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
            
            # 存储控件引用以便后续更新
            setattr(self, f'device_status_canvas_{device["index"]}', status_canvas)
            setattr(self, f'device_level_label_{device["index"]}', level_label)
    
    def toggle_device_enabled(self, device_index, enabled):
        """切换设备启用状态"""
        self.device_enabled[device_index] = enabled
        device_name = next((d['name'] for d in self.audio_devices if d['index'] == device_index), f"设备{device_index}")
        status = "启用" if enabled else "禁用"
        self.logger.info(f"{status}音频设备: {device_name}")
    
    def start_device_monitoring(self):
        """开始监控音频设备"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.logger.info("开始监控音频设备状态...")
        
        # 为每个启用的设备创建监控线程
        for device in self.audio_devices:
            if self.device_enabled.get(device['index'], True):
                monitor_thread = threading.Thread(
                    target=self.monitor_device,
                    args=(device['index'],),
                    daemon=True
                )
                monitor_thread.start()
                self.device_monitors[device['index']] = monitor_thread

    def stop_device_monitoring(self):
        """停止监控音频设备"""
        self.monitoring_active = False
        self.logger.info("停止监控音频设备状态")
        
        # 重置所有设备状态
        for device_index in self.device_status:
            self.device_status[device_index] = {'active': False, 'level': 0}
            self.update_device_status_display(device_index)
    
    def monitor_device(self, device_index):
        """监控单个设备的音频输入"""
        try:
            # 创建音频流用于监控
            stream = self.audio.open(
                format=self.format,
                channels=1,
                rate=self.rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk
            )
            
            while self.monitoring_active and self.device_enabled.get(device_index, True):
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    
                    # 计算音频级别
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    level = np.sqrt(np.mean(audio_data**2))
                    level_percent = min(100, (level / 3000) * 100)  # 归一化到0-100%
                    
                    # 更新设备状态
                    is_active = level_percent > 1  # 阈值可调整
                    self.device_status[device_index] = {
                        'active': is_active,
                        'level': level_percent
                    }
                    
                    # 在主线程中更新UI
                    self.root.after(0, self.update_device_status_display, device_index)
                    
                    time.sleep(0.1)  # 100ms更新间隔
                    
                except Exception as e:
                    if self.monitoring_active:
                        self.logger.warning(f"设备 {device_index} 监控错误: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            self.logger.error(f"无法监控设备 {device_index}: {e}")
    
    def update_device_status_display(self, device_index):
        """更新设备状态显示"""
        try:
            status_canvas = getattr(self, f'device_status_canvas_{device_index}', None)
            level_label = getattr(self, f'device_level_label_{device_index}', None)
            
            if status_canvas and level_label:
                device_status = self.device_status.get(device_index, {'active': False, 'level': 0})
                
                # 更新状态指示灯
                status_canvas.delete("all")
                color = 'green' if device_status['active'] else 'gray'
                status_canvas.create_oval(2, 2, 10, 10, fill=color, outline='black')
                
                # 更新音量级别
                level_label.config(text=f"音量: {device_status['level']:.0f}%")
                
        except Exception as e:
            pass  # 忽略UI更新错误
    
    def append_log(self, log_record):
        """在日志区域添加日志消息"""
        try:
            # 格式化日志消息
            formatted_message = self.log_handler.format(log_record)
            
            # 更新日志显示区域
            self.log_area.config(state=tk.NORMAL)
            self.log_area.insert(tk.END, formatted_message + "\n")
            
            # 限制日志行数（保留最近1000行）
            lines = self.log_area.get("1.0", tk.END).split("\n")
            if len(lines) > 1000:
                self.log_area.delete("1.0", f"{len(lines)-1000}.0")
            
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
        
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        try:
            self.log_info("开始录音...")
            self.recording = True
            self.frames = []
            self.audio_buffer = []
            self.record_button.config(text="停止录音")
            self.status_label.config(text="正在录音...")
            self.status_bar.config(text="录音中...")
            
            # 检查是否启用实时转写
            self.real_time_transcription = self.realtime_var.get()
            if self.real_time_transcription:
                self.realtime_status.config(text="实时转写: 启动中...")
                # 启动实时转写线程
                self.transcription_thread = threading.Thread(target=self.real_time_transcribe)
                self.transcription_thread.daemon = True
                self.transcription_thread.start()
                self.log_info(f"实时转写已启动，使用引擎: {self.engine_type}")
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
            # 获取选中的音频设备
            selected_device = self.get_selected_device()
            if selected_device is None:
                self.root.after(0, lambda: messagebox.showwarning("警告", "请先选择一个音频设备"))
                return
            
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=selected_device['index'],
                frames_per_buffer=self.chunk
            )
            
            self.log_info(f"音频流配置: {self.rate}Hz, {self.channels}声道, 缓冲区大小: {self.chunk}, 设备: {selected_device['name']}")
            
            self.start_time = time.time()
            self.last_transcription_time = self.start_time
            
            while self.recording:
                data = self.stream.read(self.chunk)
                self.frames.append(data)
                
                # 如果启用实时转写，将音频数据添加到缓冲区
                if self.real_time_transcription:
                    self.audio_buffer.append(data)
                    
                    # 每隔指定时间进行一次转写
                    current_time = time.time()
                    if current_time - self.last_transcription_time >= self.buffer_duration:
                        # 将缓冲区数据放入队列
                        if self.audio_buffer:
                            buffer_copy = self.audio_buffer.copy()
                            self.audio_queue.put(buffer_copy)
                            self.audio_buffer = []  # 清空缓冲区
                            self.last_transcription_time = current_time
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"录音过程中出错: {str(e)}"))
    
    def get_selected_device(self):
        """获取当前选中的音频设备"""
        # 如果有默认设备且启用，使用默认设备
        for device in self.audio_devices:
            if device['is_default'] and self.device_enabled.get(device['index'], True):
                return device
        
        # 否则使用第一个启用的设备
        for device in self.audio_devices:
            if self.device_enabled.get(device['index'], True):
                return device
        
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
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.log_info("音频流已关闭")
            
        self.record_button.config(text="开始录音")
        self.status_label.config(text="录音完成")
        self.status_bar.config(text="保存录音文件...")
        self.realtime_status.config(text="实时转写: 已停止")
        
        # 保存录音文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        
        # 确保audio目录存在
        audio_dir = os.path.join(os.getcwd(), "audio")
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
            self.log_info(f"创建音频目录: {audio_dir}")
        
        self.current_audio_file = os.path.join(audio_dir, filename)
        
        try:
            wf = wave.open(self.current_audio_file, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            
            self.transcribe_button.config(state="normal")
            self.status_bar.config(text=f"录音已保存: {filename}")
            
            # 计算录音时长和文件大小
            duration = time.time() - self.start_time if self.start_time else 0
            file_size = os.path.getsize(self.current_audio_file) / 1024  # KB
            self.log_info(f"录音文件已保存: {filename}, 时长: {duration:.1f}秒, 大小: {file_size:.1f}KB")
            
        except Exception as e:
            self.log_error(f"保存录音文件失败: {str(e)}")
            messagebox.showerror("错误", f"保存录音文件失败: {str(e)}")
            
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
            
    def update_transcription_result(self, text):
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(1.0, text)
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
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get(timeout=1)
                    transcription_count += 1
                    
                    # 将音频数据转换为可识别的格式
                    audio_bytes = b''.join(audio_data)
                    
                    # 创建临时WAV文件
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        wf = wave.open(temp_file.name, 'wb')
                        wf.setnchannels(self.channels)
                        wf.setsampwidth(self.audio.get_sample_size(self.format))
                        wf.setframerate(self.rate)
                        wf.writeframes(audio_bytes)
                        wf.close()
                        
                        # 进行语音识别
                        try:
                            text = ""
                            if self.engine_type == "whisper":
                                try:
                                    text = self.transcribe_with_whisper(temp_file.name)
                                except Exception as e:
                                    # Whisper识别失败，记录日志但继续
                                    if transcription_count % 10 == 1:  # 每10次记录一次错误，避免日志过多
                                        self.log_warning(f"实时Whisper转写失败: {str(e)}")
                            else:
                                # 使用Google引擎
                                with sr.AudioFile(temp_file.name) as source:
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
                            # 清理临时文件
                            try:
                                os.unlink(temp_file.name)
                            except:
                                pass
                                
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
        
    def append_realtime_text(self, text):
        """向文本区域追加实时转写结果"""
        self.text_area.insert(tk.END, text)
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
        if self.whisper_model is None:
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
                
                self.log_info(f"开始加载Whisper模型，设备: {device} {gpu_info}")
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
                messagebox.showerror("错误", f"加载Whisper模型失败: {str(e)}\n\n建议：\n1. 检查网络连接\n2. 确保有足够的磁盘空间\n3. 尝试重新启动程序")
                self.engine_var.set("google")
                self.engine_type = "google"
                self.status_label.config(text="已回退到Google引擎")
                self.log_info("已回退到Google引擎")
    
    def transcribe_with_whisper(self, audio_file_path):
        """使用Whisper进行转写"""
        try:
            if self.whisper_model is None:
                self.load_whisper_model()
            
            if self.whisper_model is not None:
                self.log_info("开始Whisper转写，使用中文语言...")
                start_time = time.time()
                # 使用中文语言，不进行自动检测
                result = self.whisper_model.transcribe(
                    audio_file_path, 
                    language='zh',
                    initial_prompt="以下是普通话的句子。"
                )
                transcribe_time = time.time() - start_time
                
                detected_language = result.get('language', '未知')
                self.log_info(f"Whisper转写完成，耗时: {transcribe_time:.1f}秒, 检测语言: {detected_language}")
                
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