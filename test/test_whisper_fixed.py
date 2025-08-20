#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的Whisper功能
"""

import sys
import os
import numpy as np
import wave
import tempfile
from audio_transcriber_refactored import AudioTranscriptionEngine, TranscriptionConfig

def create_test_audio_with_speech():
    """创建包含简单语音的测试音频文件"""
    # 创建一个包含简单音调的音频文件（模拟语音）
    sample_rate = 16000
    duration = 2.0  # 2秒
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # 创建多个频率的正弦波混合，模拟语音
    frequencies = [200, 400, 800, 1600]  # 模拟语音的基频和谐波
    audio_data = np.zeros_like(t)
    
    for freq in frequencies:
        audio_data += 0.1 * np.sin(2 * np.pi * freq * t)
    
    # 添加一些随机噪声使其更像语音
    noise = 0.05 * np.random.randn(len(t))
    audio_data += noise
    
    # 归一化到16位整数范围
    audio_data = np.clip(audio_data, -1, 1)
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # 保存为临时WAV文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    with wave.open(temp_file.name, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return temp_file.name

def test_logger(level, message):
    """测试日志函数"""
    print(f"[{level.upper()}] {message}")

def main():
    print("=== 测试修复后的Whisper功能 ===")
    
    try:
        # 创建配置
        config = TranscriptionConfig()
        config.engine_type = "whisper"
        print(f"引擎类型: {config.engine_type}")
        
        # 创建转写引擎
        engine = AudioTranscriptionEngine(config, test_logger)
        print("✓ AudioTranscriptionEngine创建成功")
        
        # 创建测试音频文件
        test_audio_file = create_test_audio_with_speech()
        print(f"创建测试音频文件: {test_audio_file}")
        
        try:
            print("\n--- 测试Whisper模型加载和转写 ---")
            print("调用_transcribe_with_whisper方法...")
            result = engine._transcribe_with_whisper(test_audio_file)
            
            print(f"返回结果: {repr(result)}")
            
            if result:
                print(f"✓ Whisper转写成功: '{result}'")
            elif result == "":
                print("⚠ Whisper转写返回空字符串（这对于测试音频是正常的）")
            else:
                print("⚠ Whisper转写返回None")
                
            # 检查模型是否已加载
            if hasattr(engine, 'whisper_model') and engine.whisper_model:
                print(f"✓ Whisper模型已加载: {type(engine.whisper_model)}")
            elif hasattr(engine, 'belle_pipeline') and engine.belle_pipeline:
                print(f"✓ BELLE模型已加载: {type(engine.belle_pipeline)}")
            else:
                print("✗ 没有模型被加载")
                
        except Exception as e:
            print(f"✗ Whisper转写失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理测试文件
            if os.path.exists(test_audio_file):
                os.unlink(test_audio_file)
                print("清理测试音频文件")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()