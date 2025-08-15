#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WASAPI Loopback音频测试程序
用于录制系统音频并进行可视化分析
"""

import pyaudiowpatch as pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
import queue
import sys
from datetime import datetime

class LoopbackAudioTester:
    def __init__(self):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1  # 输出单声道
        self.RATE = 48000  # 使用设备默认采样率
        self.RECORD_SECONDS = 10
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()
        self.recording = False
        
        # 用于存储音频数据
        self.audio_data = []
        self.time_data = []
        
    def find_loopback_device(self):
        """查找WASAPI Loopback设备"""
        print("\n=== 扫描WASAPI Loopback设备 ===")
        
        loopback_devices = []
        device_count = self.audio.get_device_count()
        
        for i in range(device_count):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                if device_info.get('name', '').find('Loopback') != -1:
                    loopback_devices.append((i, device_info))
                    print(f"设备 {i}: {device_info['name']}")
                    print(f"  通道数: {device_info['maxInputChannels']}")
                    print(f"  采样率: {device_info['defaultSampleRate']}Hz")
                    print()
            except Exception as e:
                continue
                
        if not loopback_devices:
            print("❌ 未找到WASAPI Loopback设备")
            return None
            
        # 优先选择HECATE声卡设备
        hecate_device = None
        for device_index, device_info in loopback_devices:
            if 'HECATE' in device_info['name'].upper():
                hecate_device = (device_index, device_info)
                break
        
        if hecate_device:
            device_index, device_info = hecate_device
            print(f"✅ 选择HECATE设备: {device_info['name']}")
        else:
            # 如果没有HECATE设备，选择第一个可用的loopback设备
            device_index, device_info = loopback_devices[0]
            print(f"✅ 选择设备: {device_info['name']}")
        
        # 使用设备的默认采样率
        self.RATE = int(device_info['defaultSampleRate'])
        print(f"📊 使用采样率: {self.RATE}Hz")
        
        return device_index, device_info
    
    def analyze_channel_data(self, data, channels):
        """分析多通道音频数据"""
        if channels == 1:
            return data
            
        # 转换为numpy数组
        audio_array = np.frombuffer(data, dtype=np.int16)
        
        if len(audio_array) < self.CHUNK * channels:
            return np.zeros(self.CHUNK, dtype=np.int16)
            
        # 重新整形为(samples, channels)
        try:
            reshaped = audio_array[:self.CHUNK * channels].reshape(self.CHUNK, channels)
            
            # 分析各通道特征
            print(f"\n--- 通道分析 (时间: {datetime.now().strftime('%H:%M:%S')}) ---")
            channel_stats = []
            for ch in range(min(channels, 8)):
                ch_data = reshaped[:, ch]
                ch_std = np.std(ch_data)
                ch_max = np.max(np.abs(ch_data))
                ch_rms = np.sqrt(np.mean(ch_data.astype(np.float32)**2))
                channel_stats.append({
                    'channel': ch,
                    'std': ch_std,
                    'max': ch_max,
                    'rms': ch_rms,
                    'data': ch_data
                })
                print(f"CH{ch}: std={ch_std:.1f}, max={ch_max}, rms={ch_rms:.1f}")
            
            # 使用改进的downmix算法
            if channels == 2:
                # 立体声
                mono_data = ((reshaped[:, 0].astype(np.float32) + reshaped[:, 1].astype(np.float32)) / 2).astype(np.int16)
            elif channels <= 8:
                # 多通道 - 主要使用前两个通道
                left = reshaped[:, 0].astype(np.float32)
                right = reshaped[:, 1].astype(np.float32) if channels > 1 else left
                
                # 检查其他通道是否有有效数据
                other_channels = np.zeros_like(left)
                active_other_channels = 0
                
                if channels > 2:
                    for ch in range(2, min(channels, 8)):
                        ch_data = reshaped[:, ch].astype(np.float32)
                        # 检查通道是否有有效且不同的数据
                        if (np.std(ch_data) > 100 and 
                            not np.allclose(ch_data, left, atol=1000) and 
                            not np.allclose(ch_data, right, atol=1000)):
                            other_channels += ch_data * 0.1
                            active_other_channels += 1
                            print(f"  -> CH{ch} 包含有效数据，已混入")
                
                print(f"  -> 活跃的其他通道数: {active_other_channels}")
                
                # 主要基于立体声，少量混入其他通道
                mono_data = ((left + right) / 2 + other_channels * 0.2).astype(np.int16)
                mono_data = np.clip(mono_data, -32768, 32767)
            else:
                # 超过8通道，只使用前两个
                left = reshaped[:, 0].astype(np.float32)
                right = reshaped[:, 1].astype(np.float32) if channels > 1 else left
                mono_data = ((left + right) / 2).astype(np.int16)
            
            return mono_data
            
        except Exception as e:
            print(f"❌ 通道分析错误: {e}")
            return np.zeros(self.CHUNK, dtype=np.int16)
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        if status:
            print(f"音频状态: {status}")
            
        # 将音频数据放入队列
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def start_recording(self, device_index, device_info):
        """开始录制音频"""
        try:
            # 获取设备的实际通道数
            device_channels = int(device_info['maxInputChannels'])
            print(f"\n🎙️ 开始录制音频...")
            print(f"设备通道数: {device_channels}")
            print(f"输出通道数: {self.CHANNELS}")
            print(f"采样率: {self.RATE}Hz")
            print(f"录制时长: {self.RECORD_SECONDS}秒")
            
            # 创建音频流
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=device_channels,  # 使用设备的实际通道数
                rate=self.RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.audio_callback
            )
            
            self.recording = True
            self.stream.start_stream()
            
            # 录制音频数据
            start_time = time.time()
            frame_count = 0
            
            while self.recording and (time.time() - start_time) < self.RECORD_SECONDS:
                try:
                    # 从队列获取音频数据
                    data = self.audio_queue.get(timeout=1.0)
                    
                    # 分析并转换音频数据
                    mono_data = self.analyze_channel_data(data, device_channels)
                    
                    # 存储音频数据用于分析
                    self.audio_data.extend(mono_data)
                    current_time = time.time() - start_time
                    self.time_data.extend([current_time + i/self.RATE for i in range(len(mono_data))])
                    
                    frame_count += 1
                    if frame_count % 20 == 0:  # 每20帧打印一次状态
                        rms = np.sqrt(np.mean(mono_data.astype(np.float32)**2))
                        print(f"帧 {frame_count}: RMS={rms:.1f}, 最大值={np.max(np.abs(mono_data))}")
                        
                except queue.Empty:
                    print("⚠️ 音频队列超时")
                    continue
                except Exception as e:
                    print(f"❌ 处理音频数据错误: {e}")
                    break
            
            print(f"\n✅ 录制完成，共录制 {len(self.audio_data)} 个采样点")
            
        except Exception as e:
            print(f"❌ 录制错误: {e}")
        finally:
            self.stop_recording()
    
    def stop_recording(self):
        """停止录制"""
        self.recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
    def analyze_audio(self):
        """分析录制的音频数据"""
        if not self.audio_data:
            print("❌ 没有音频数据可分析")
            return
            
        audio_array = np.array(self.audio_data, dtype=np.float32)
        time_array = np.array(self.time_data)
        
        print(f"\n=== 音频分析结果 ===")
        print(f"总采样点数: {len(audio_array)}")
        print(f"录制时长: {time_array[-1]:.2f}秒")
        print(f"RMS值: {np.sqrt(np.mean(audio_array**2)):.2f}")
        print(f"最大值: {np.max(np.abs(audio_array)):.2f}")
        print(f"标准差: {np.std(audio_array):.2f}")
        
        # 检查是否有明显的音频信号
        rms_threshold = 100  # RMS阈值
        max_threshold = 1000  # 最大值阈值
        
        rms_value = np.sqrt(np.mean(audio_array**2))
        max_value = np.max(np.abs(audio_array))
        
        if rms_value > rms_threshold and max_value > max_threshold:
            print("✅ 检测到有效音频信号")
        else:
            print("⚠️ 音频信号较弱或可能存在问题")
            
        # 检查是否有电流声特征（高频噪音）
        # 计算频谱
        fft = np.fft.fft(audio_array)
        freqs = np.fft.fftfreq(len(audio_array), 1/self.RATE)
        magnitude = np.abs(fft)
        
        # 检查高频部分的能量
        high_freq_mask = freqs > 8000  # 8kHz以上
        high_freq_energy = np.mean(magnitude[high_freq_mask])
        total_energy = np.mean(magnitude)
        
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        print(f"高频能量比例: {high_freq_ratio:.3f}")
        if high_freq_ratio > 0.3:
            print("⚠️ 检测到可能的高频噪音（电流声）")
        else:
            print("✅ 高频噪音水平正常")
    
    def plot_audio(self):
        """绘制音频波形和频谱图"""
        if not self.audio_data:
            print("❌ 没有音频数据可绘制")
            return
            
        audio_array = np.array(self.audio_data, dtype=np.float32)
        time_array = np.array(self.time_data)
        
        # 创建图形
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('WASAPI Loopback音频分析', fontsize=16)
        
        # 1. 完整波形图
        ax1.plot(time_array, audio_array, 'b-', linewidth=0.5)
        ax1.set_title('完整音频波形')
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('幅度')
        ax1.grid(True, alpha=0.3)
        
        # 2. 局部波形图（前1秒）
        if len(time_array) > 0:
            mask = time_array <= 1.0
            if np.any(mask):
                ax2.plot(time_array[mask], audio_array[mask], 'r-', linewidth=1)
                ax2.set_title('前1秒波形详图')
                ax2.set_xlabel('时间 (秒)')
                ax2.set_ylabel('幅度')
                ax2.grid(True, alpha=0.3)
        
        # 3. 频谱图
        fft = np.fft.fft(audio_array)
        freqs = np.fft.fftfreq(len(audio_array), 1/self.RATE)
        magnitude = np.abs(fft)
        
        # 只显示正频率部分
        positive_freq_mask = freqs >= 0
        ax3.semilogx(freqs[positive_freq_mask], 20*np.log10(magnitude[positive_freq_mask] + 1e-10), 'g-')
        ax3.set_title('频谱图')
        ax3.set_xlabel('频率 (Hz)')
        ax3.set_ylabel('幅度 (dB)')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(20, self.RATE//2)
        
        # 4. RMS随时间变化
        window_size = self.RATE // 10  # 0.1秒窗口
        rms_values = []
        rms_times = []
        
        for i in range(0, len(audio_array) - window_size, window_size//2):
            window = audio_array[i:i+window_size]
            rms = np.sqrt(np.mean(window**2))
            rms_values.append(rms)
            rms_times.append(time_array[i + window_size//2] if i + window_size//2 < len(time_array) else time_array[-1])
        
        ax4.plot(rms_times, rms_values, 'm-', linewidth=2)
        ax4.set_title('RMS随时间变化')
        ax4.set_xlabel('时间 (秒)')
        ax4.set_ylabel('RMS值')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'loopback_audio_analysis_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 音频分析图已保存: {filename}")
        
        plt.show()
    
    def run_test(self):
        """运行完整测试"""
        print("🎵 WASAPI Loopback音频测试程序")
        print("=" * 50)
        
        try:
            # 查找loopback设备
            device_result = self.find_loopback_device()
            if not device_result:
                return
                
            device_index, device_info = device_result
            
            # 开始录制
            self.start_recording(device_index, device_info)
            
            # 分析音频
            self.analyze_audio()
            
            # 绘制图形
            self.plot_audio()
            
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断测试")
        except Exception as e:
            print(f"❌ 测试错误: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.stop_recording()
        if self.audio:
            self.audio.terminate()
        print("\n🧹 资源清理完成")

def main():
    """主函数"""
    tester = LoopbackAudioTester()
    tester.run_test()

if __name__ == "__main__":
    main()