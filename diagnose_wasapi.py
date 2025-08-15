#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WASAPI Loopback 设备诊断工具
用于诊断和解决 WASAPI loopback 设备不可用的问题
"""

import pyaudiowpatch as pyaudio
import time
import threading
import sys

def print_separator(title):
    """打印分隔线"""
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

def test_basic_audio_system():
    """测试基本音频系统"""
    print_separator("基本音频系统测试")
    
    try:
        p = pyaudio.PyAudio()
        print("✅ PyAudioWPatch 初始化成功")
        
        # 获取设备数量
        device_count = p.get_device_count()
        print(f"📊 检测到 {device_count} 个音频设备")
        
        # 获取主机API信息
        host_api_count = p.get_host_api_count()
        print(f"🖥️  检测到 {host_api_count} 个主机API")
        
        for i in range(host_api_count):
            host_api_info = p.get_host_api_info_by_index(i)
            print(f"   API {i}: {host_api_info['name']} (设备数: {host_api_info['deviceCount']})")
        
        p.terminate()
        return True
        
    except Exception as e:
        print(f"❌ PyAudioWPatch 初始化失败: {e}")
        return False

def list_all_devices():
    """列出所有音频设备"""
    print_separator("所有音频设备列表")
    
    try:
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        
        input_devices = []
        output_devices = []
        
        for i in range(device_count):
            try:
                device_info = p.get_device_info_by_index(i)
                
                device_type = []
                if device_info['maxInputChannels'] > 0:
                    device_type.append("输入")
                    input_devices.append(device_info)
                if device_info['maxOutputChannels'] > 0:
                    device_type.append("输出")
                    output_devices.append(device_info)
                
                type_str = "/".join(device_type) if device_type else "未知"
                
                print(f"设备 {i}: {device_info['name']}")
                print(f"   类型: {type_str}")
                print(f"   输入通道: {device_info['maxInputChannels']}")
                print(f"   输出通道: {device_info['maxOutputChannels']}")
                print(f"   默认采样率: {int(device_info['defaultSampleRate'])}Hz")
                print(f"   主机API: {device_info['hostApi']}")
                print()
                
            except Exception as e:
                print(f"❌ 无法获取设备 {i} 信息: {e}")
        
        print(f"📊 总结: {len(input_devices)} 个输入设备, {len(output_devices)} 个输出设备")
        
        p.terminate()
        return input_devices, output_devices
        
    except Exception as e:
        print(f"❌ 列出设备失败: {e}")
        return [], []

def test_wasapi_loopback_devices():
    """测试WASAPI loopback设备"""
    print_separator("WASAPI Loopback 设备测试")
    
    try:
        p = pyaudio.PyAudio()
        
        # 使用PyAudioWPatch的loopback设备生成器
        loopback_devices = []
        
        print("🔍 扫描 WASAPI Loopback 设备...")
        
        try:
            for loopback_info in p.get_loopback_device_info_generator():
                loopback_devices.append(loopback_info)
                
                print(f"\n📱 Loopback 设备 {loopback_info['index']}: {loopback_info['name']}")
                print(f"   输入通道: {loopback_info['maxInputChannels']}")
                print(f"   输出通道: {loopback_info['maxOutputChannels']}")
                print(f"   默认采样率: {int(loopback_info['defaultSampleRate'])}Hz")
                print(f"   主机API: {loopback_info['hostApi']}")
                
                # 测试设备可用性
                print("   测试设备可用性...")
                available = test_loopback_device(p, loopback_info['index'], loopback_info['name'])
                status = "✅ 可用" if available else "❌ 不可用"
                print(f"   状态: {status}")
                
        except Exception as e:
            print(f"❌ 扫描 loopback 设备失败: {e}")
        
        # 尝试获取默认WASAPI loopback设备
        print("\n🎯 测试默认 WASAPI Loopback 设备...")
        try:
            default_loopback = p.get_default_wasapi_loopback()
            if default_loopback:
                print(f"✅ 找到默认 WASAPI Loopback: {default_loopback['name']}")
                available = test_loopback_device(p, default_loopback['index'], default_loopback['name'])
                status = "✅ 可用" if available else "❌ 不可用"
                print(f"   状态: {status}")
            else:
                print("❌ 未找到默认 WASAPI Loopback 设备")
        except Exception as e:
            print(f"❌ 获取默认 WASAPI Loopback 失败: {e}")
        
        p.terminate()
        return loopback_devices
        
    except Exception as e:
        print(f"❌ WASAPI Loopback 测试失败: {e}")
        return []

def test_loopback_device(p, device_index, device_name):
    """测试单个loopback设备"""
    # 获取设备信息以确定支持的采样率
    try:
        device_info = p.get_device_info_by_index(device_index)
        default_rate = int(device_info['defaultSampleRate'])
        print(f"     设备默认采样率: {default_rate}Hz")
    except:
        default_rate = 48000  # 大多数现代设备的默认采样率
    
    # 尝试多个常见采样率
    sample_rates = [default_rate, 48000, 44100, 96000, 192000]
    
    for rate in sample_rates:
        try:
            print(f"     尝试采样率: {rate}Hz")
            # 尝试打开loopback设备 - PyAudioWPatch直接通过设备索引访问loopback设备
            stream = p.open(
                format=pyaudio.paInt16,
                channels=2,  # 立体声
                rate=rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024
            )
            
            # 尝试读取一些数据
            try:
                data = stream.read(1024, exception_on_overflow=False)
                print(f"     ✅ 成功读取 {len(data)} 字节数据 (采样率: {rate}Hz)")
            except Exception as read_error:
                print(f"     ⚠️  读取数据时出错: {read_error}")
            
            stream.close()
            return True
            
        except Exception as e:
            print(f"     ❌ 采样率 {rate}Hz 失败: {e}")
            continue
    
    print(f"     ❌ 所有采样率都失败")
    return False

def check_audio_output_activity():
    """检查音频输出活动"""
    print_separator("音频输出活动检查")
    
    print("🔊 WASAPI Loopback 需要有音频输出才能工作")
    print("\n请确保:")
    print("1. 系统正在播放音频（音乐、视频等）")
    print("2. 系统音量不是静音状态")
    print("3. 音频设备正常工作")
    
    print("\n💡 测试建议:")
    print("1. 打开一个音乐播放器或视频")
    print("2. 确保能听到声音")
    print("3. 重新运行此诊断工具")

def provide_solutions():
    """提供解决方案"""
    print_separator("常见问题解决方案")
    
    print("🔧 如果 WASAPI Loopback 设备不可用，请尝试以下解决方案:")
    print()
    
    print("1️⃣ 确保音频输出活动:")
    print("   • 播放一些音频（音乐、视频等）")
    print("   • 确保系统音量不是静音")
    print("   • 检查音频设备是否正常工作")
    print()
    
    print("2️⃣ 检查音频驱动程序:")
    print("   • 更新音频驱动程序到最新版本")
    print("   • 确保驱动程序支持 WASAPI")
    print("   • 重启计算机后重试")
    print()
    
    print("3️⃣ Windows 音频设置:")
    print("   • 右键点击任务栏音量图标")
    print("   • 选择'声音设置'或'播放设备'")
    print("   • 确保默认播放设备已启用")
    print("   • 检查设备属性中的高级设置")
    print()
    
    print("4️⃣ 权限问题:")
    print("   • 以管理员身份运行程序")
    print("   • 检查防病毒软件是否阻止音频访问")
    print("   • 确保程序有麦克风访问权限")
    print()
    
    print("5️⃣ 替代方案:")
    print("   • 启用'立体声混音'设备（如果可用）")
    print("   • 使用虚拟音频线缆软件")
    print("   • 考虑使用专业音频接口")
    print()
    
    print("6️⃣ 调试步骤:")
    print("   • 运行 'python -m pyaudiowpatch' 查看详细设备信息")
    print("   • 检查 Windows 事件查看器中的音频相关错误")
    print("   • 尝试其他音频应用程序（如 Audacity）测试 WASAPI")

def main():
    """主函数"""
    print("🎵 WASAPI Loopback 设备诊断工具")
    print("此工具将帮助诊断和解决 WASAPI loopback 设备不可用的问题")
    
    # 1. 测试基本音频系统
    if not test_basic_audio_system():
        print("\n❌ 基本音频系统测试失败，请检查 PyAudioWPatch 安装")
        return
    
    # 2. 列出所有设备
    input_devices, output_devices = list_all_devices()
    
    # 3. 测试WASAPI loopback设备
    loopback_devices = test_wasapi_loopback_devices()
    
    # 4. 检查音频输出活动
    check_audio_output_activity()
    
    # 5. 提供解决方案
    provide_solutions()
    
    # 总结
    print_separator("诊断总结")
    
    available_loopback = sum(1 for device in loopback_devices if test_loopback_device(
        pyaudio.PyAudio(), device['index'], device['name']
    ))
    
    if available_loopback > 0:
        print(f"✅ 找到 {available_loopback} 个可用的 WASAPI Loopback 设备")
        print("🎉 您的系统支持 WASAPI Loopback 录音！")
    else:
        print("❌ 未找到可用的 WASAPI Loopback 设备")
        print("🔧 请参考上述解决方案进行故障排除")
    
    print("\n📝 如果问题仍然存在，请:")
    print("1. 确保正在播放音频")
    print("2. 以管理员身份运行程序")
    print("3. 更新音频驱动程序")
    print("4. 重启计算机")

if __name__ == "__main__":
    main()