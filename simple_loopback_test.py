#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的WASAPI Loopback测试脚本
用于快速诊断loopback设备问题
"""

import pyaudiowpatch as pyaudio
import time
import threading

def test_single_loopback_device():
    """测试单个loopback设备"""
    p = pyaudio.PyAudio()
    
    try:
        # 获取默认WASAPI loopback设备
        try:
            default_loopback_info = p.get_default_wasapi_loopback()
            print(f"找到默认WASAPI Loopback设备: {default_loopback_info['name']}")
            device_index = default_loopback_info['index']
            device_info = default_loopback_info
            print(f"设备索引: {device_index}")
            print(f"设备名称: {device_info['name']}")
            print(f"默认采样率: {device_info['defaultSampleRate']}")
            print(f"输入通道: {device_info['maxInputChannels']}")
        except Exception as e:
            print(f"获取默认WASAPI Loopback设备失败: {e}")
            return False
        
        # 尝试打开设备
        print("\n尝试打开loopback设备...")
        
        # 尝试不同的通道配置
        channel_configs = [2, 1]  # 先尝试立体声，再尝试单声道
        max_channels = device_info['maxInputChannels']
        
        if max_channels > 2:
            channel_configs.insert(0, max_channels)  # 如果支持更多通道，先尝试最大通道数
        
        stream = None
        for channels in channel_configs:
            if channels > max_channels:
                continue
                
            try:
                print(f"  尝试 {channels} 通道配置...")
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=channels,
                    rate=int(device_info['defaultSampleRate']),
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=1024
                )
                print(f"✅ 设备打开成功! (使用 {channels} 通道)")
                break
            except Exception as e:
                print(f"  ❌ {channels} 通道失败: {e}")
                continue
        
        if stream is None:
            print("❌ 所有通道配置都失败")
            return False
            
        # 尝试读取数据（设置超时）
        print("尝试读取音频数据...")
        
        def read_with_timeout():
            try:
                data = stream.read(1024, exception_on_overflow=False)
                return len(data)
            except Exception as e:
                return f"读取错误: {e}"
        
        # 使用线程和超时机制
        result = [None]
        def read_thread():
            result[0] = read_with_timeout()
        
        thread = threading.Thread(target=read_thread)
        thread.daemon = True
        thread.start()
        thread.join(timeout=3)  # 3秒超时
        
        if thread.is_alive():
            print("⚠️  读取操作超时（可能是因为没有音频输出）")
        elif result[0] is not None:
            if isinstance(result[0], int):
                print(f"✅ 成功读取 {result[0]} 字节数据")
            else:
                print(f"❌ {result[0]}")
        
        stream.close()
        return True
            
    finally:
        p.terminate()

def main():
    print("=" * 60)
    print(" 简单WASAPI Loopback测试")
    print("=" * 60)
    
    print("\n🔍 测试WASAPI Loopback功能...")
    success = test_single_loopback_device()
    
    print("\n" + "=" * 60)
    print(" 测试结果")
    print("=" * 60)
    
    if success:
        print("✅ WASAPI Loopback设备可以正常打开")
        print("\n💡 如果读取超时，请确保:")
        print("   1. 系统正在播放音频")
        print("   2. 音量不为0")
        print("   3. 播放设备与loopback设备匹配")
    else:
        print("❌ WASAPI Loopback设备无法使用")
        print("\n🔧 可能的解决方案:")
        print("   1. 以管理员身份运行")
        print("   2. 更新音频驱动程序")
        print("   3. 检查Windows音频服务")
        print("   4. 重启计算机")

if __name__ == "__main__":
    main()