#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试重构后的AudioTranscriptionEngine
"""

import sys
import os
import tempfile
import wave
import numpy as np
from audio_transcriber_refactored import AudioTranscriptionEngine, TranscriptionConfig, AudioSource
import whisper

def create_test_audio_file():
    """创建一个测试音频文件"""
    # 生成1秒的440Hz正弦波（A音符）
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # 创建临时WAV文件
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    with wave.open(temp_file.name, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return temp_file.name

def test_logger(level, message):
    """测试日志函数"""
    print(f"[{level.upper()}] {message}")

def test_whisper_directly():
    """直接测试whisper模块"""
    print("\n=== 直接测试Whisper模块 ===")
    try:
        model = whisper.load_model("base")
        print("✓ Whisper模型加载成功")
        
        # 创建测试音频
        test_audio_file = create_test_audio_file()
        print(f"创建测试音频文件: {test_audio_file}")
        
        # 直接使用whisper转写
        result = model.transcribe(test_audio_file)
        print(f"Whisper原始结果: {result}")
        print(f"转写文本: '{result['text'].strip()}'")
        print(f"文本长度: {len(result['text'].strip())}")
        
        # 清理
        os.unlink(test_audio_file)
        return True
        
    except Exception as e:
        print(f"✗ 直接测试Whisper失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== 测试AudioTranscriptionEngine ===")
    
    # 首先直接测试whisper
    if not test_whisper_directly():
        print("直接测试whisper失败，退出")
        return
    
    try:
        # 创建配置
        config = TranscriptionConfig()
        print(f"\n默认引擎类型: {config.engine_type}")
        
        # 创建转写引擎
        engine = AudioTranscriptionEngine(config, test_logger)
        print("✓ AudioTranscriptionEngine创建成功")
        
        # 测试Whisper引擎配置
        print("\n--- 测试Whisper引擎 ---")
        config.engine_type = "whisper"
        engine.config = config
        print(f"当前引擎类型: {engine.config.engine_type}")
        
        # 测试Whisper模型加载
        print("\n--- 测试AudioTranscriptionEngine中的Whisper ---")
        test_audio_file = create_test_audio_file()
        print(f"创建测试音频文件: {test_audio_file}")
        
        try:
            print("调用_transcribe_with_whisper方法...")
            result = engine._transcribe_with_whisper(test_audio_file)
            print(f"返回结果: {repr(result)}")
            
            if result:
                print(f"✓ Whisper转写成功: {result}")
            elif result == "":
                print("⚠ Whisper转写返回空字符串")
            else:
                print("⚠ Whisper转写返回None")
                
            # 检查whisper_model是否已加载
            if hasattr(engine, 'whisper_model') and engine.whisper_model:
                print(f"✓ Whisper模型已加载: {type(engine.whisper_model)}")
            else:
                print("✗ Whisper模型未加载")
                
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