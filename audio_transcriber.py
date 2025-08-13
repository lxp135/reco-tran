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
import imageio_ffmpeg as ffmpeg

class AudioTranscriber:
    def __init__(self, root):
        self.root = root
        self.root.title("录音转写工具")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 设置ffmpeg路径
        try:
            ffmpeg_path = ffmpeg.get_ffmpeg_exe()
            # 直接设置pydub的工具路径
            from pydub.utils import which
            AudioSegment.converter = ffmpeg_path
            AudioSegment.ffmpeg = ffmpeg_path
            
            # 尝试找到ffprobe
            ffmpeg_dir = os.path.dirname(ffmpeg_path)
            # 检查同目录下是否有ffprobe
            possible_ffprobe_names = ['ffprobe.exe', 'ffprobe']
            ffprobe_found = False
            
            for probe_name in possible_ffprobe_names:
                ffprobe_path = os.path.join(ffmpeg_dir, probe_name)
                if os.path.exists(ffprobe_path):
                    AudioSegment.ffprobe = ffprobe_path
                    ffprobe_found = True
                    break
            
            # 如果找不到ffprobe，使用ffmpeg作为替代
            if not ffprobe_found:
                AudioSegment.ffprobe = ffmpeg_path
            
            print(f"✅ 成功设置ffmpeg路径: {ffmpeg_path}")
            if ffprobe_found:
                print(f"✅ 成功设置ffprobe路径: {AudioSegment.ffprobe}")
            else:
                print(f"⚠️  使用ffmpeg作为ffprobe替代")
                
        except Exception as e:
            print(f"警告: 无法设置ffmpeg路径: {e}")
        
        # 录音参数
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # 使用16kHz采样率，更适合语音识别
        self.recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # 语音识别器
        self.recognizer = sr.Recognizer()
        
        # 当前录音文件路径
        self.current_audio_file = None
        
        self.setup_ui()
        
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
        control_frame.columnconfigure(1, weight=1)
        
        # 录音按钮
        self.record_button = ttk.Button(control_frame, text="开始录音", command=self.toggle_recording)
        self.record_button.grid(row=0, column=0, padx=(0, 10))
        
        # 录音状态标签
        self.status_label = ttk.Label(control_frame, text="准备就绪")
        self.status_label.grid(row=0, column=1, sticky=tk.W)
        
        # 录音时长标签
        self.duration_label = ttk.Label(control_frame, text="时长: 00:00")
        self.duration_label.grid(row=0, column=2)
        
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
        
        # 转写结果区域
        result_frame = ttk.LabelFrame(main_frame, text="转写结果", padding="10")
        result_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        # 文本显示区域
        self.text_area = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, height=15)
        self.text_area.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 状态栏
        self.status_bar = ttk.Label(main_frame, text="就绪", relief=tk.SUNKEN)
        self.status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        try:
            self.recording = True
            self.frames = []
            self.record_button.config(text="停止录音")
            self.status_label.config(text="正在录音...")
            self.status_bar.config(text="录音中...")
            
            # 开始录音线程
            self.record_thread = threading.Thread(target=self.record_audio)
            self.record_thread.daemon = True
            self.record_thread.start()
            
            # 开始计时线程
            self.timer_thread = threading.Thread(target=self.update_timer)
            self.timer_thread.daemon = True
            self.timer_thread.start()
            
        except Exception as e:
            messagebox.showerror("错误", f"录音启动失败: {str(e)}")
            self.recording = False
            
    def record_audio(self):
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            self.start_time = time.time()
            
            while self.recording:
                data = self.stream.read(self.chunk)
                self.frames.append(data)
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"录音过程中出错: {str(e)}"))
            
    def update_timer(self):
        while self.recording:
            elapsed = time.time() - self.start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"时长: {minutes:02d}:{seconds:02d}"
            self.root.after(0, lambda: self.duration_label.config(text=time_str))
            time.sleep(1)
            
    def stop_recording(self):
        self.recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        self.record_button.config(text="开始录音")
        self.status_label.config(text="录音完成")
        self.status_bar.config(text="保存录音文件...")
        
        # 保存录音文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        
        # 确保audio目录存在
        audio_dir = os.path.join(os.getcwd(), "audio")
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
        
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
            
        except Exception as e:
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
            self.current_audio_file = file_path
            self.transcribe_button.config(state="normal")
            self.status_bar.config(text=f"已选择文件: {os.path.basename(file_path)}")
            
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
            
            # 使用语音识别
            with sr.AudioFile(audio_file_to_use) as source:
                # 调整环境噪音
                self.recognizer.adjust_for_ambient_noise(source)
                audio_data = self.recognizer.record(source)
                
            # 尝试多种识别引擎
            text = ""
            try:
                # 首先尝试使用Google识别（需要网络）
                text = self.recognizer.recognize_google(audio_data, language='zh-CN')
                self.root.after(0, lambda: self.status_bar.config(text="转写完成（使用Google引擎）"))
            except sr.RequestError:
                try:
                    # 如果Google不可用，尝试使用离线识别
                    text = self.recognizer.recognize_sphinx(audio_data, language='zh-CN')
                    self.root.after(0, lambda: self.status_bar.config(text="转写完成（使用离线引擎）"))
                except:
                    text = "转写失败：无法连接到识别服务，且离线引擎不可用"
                    
            except sr.UnknownValueError:
                text = "无法识别音频内容，请确保音频清晰且包含语音"
            except Exception as e:
                text = f"转写过程中出现错误: {str(e)}"
                
            # 更新UI
            self.root.after(0, lambda: self.update_transcription_result(text))
            
        except Exception as e:
            error_msg = f"转写失败: {str(e)}"
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