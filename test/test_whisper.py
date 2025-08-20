#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Whisper模型加载
"""

import sys
import os

try:
    import whisper
    print("✓ Whisper模块导入成功")
    
    # 测试模型加载
    print("正在加载Whisper base模型...")
    model = whisper.load_model("base")
    print("✓ Whisper base模型加载成功")
    
    # 检查模型属性
    print(f"模型设备: {next(model.parameters()).device}")
    print(f"模型类型: {type(model)}")
    
    print("\n=== Whisper测试完成 ===")
    
except ImportError as e:
    print(f"✗ Whisper模块导入失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Whisper模型加载失败: {e}")
    sys.exit(1)