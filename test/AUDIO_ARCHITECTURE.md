# 音频录制架构重构说明

## 概述

本次重构实现了麦克风和系统音频的独立录制、转写和合并功能，提供了更灵活和精确的音频处理能力。

## 新架构特性

### 1. 独立音频流处理

- **麦克风音频流**: 独立录制麦克风输入
- **系统音频流**: 独立录制系统音频输出
- **混合音频流**: 保持原有的混合录制功能（兼容性）

### 2. 独立数据存储

```python
# 独立的音频数据存储
self.microphone_frames = []      # 麦克风音频帧
self.system_audio_frames = []    # 系统音频帧
self.frames = []                 # 混合音频帧（兼容性）

# 独立的音频缓冲区
self.microphone_buffer = []      # 麦克风实时转写缓冲区
self.system_audio_buffer = []    # 系统音频实时转写缓冲区
self.audio_buffer = []           # 混合音频缓冲区（兼容性）
```

### 3. 独立转写队列

```python
# 独立的转写队列
self.microphone_transcription_queue = queue.Queue()    # 麦克风转写队列
self.system_audio_transcription_queue = queue.Queue()  # 系统音频转写队列
self.transcription_queue = queue.Queue()               # 混合音频转写队列（兼容性）
```

### 4. 独立转写线程

- **microphone_transcribe()**: 专门处理麦克风音频转写
- **system_audio_transcribe()**: 专门处理系统音频转写
- **real_time_transcribe()**: 处理混合音频转写（兼容性）

## 核心功能实现

### 1. 音频录制流程

```python
def record_audio(self):
    while self.recording:
        # 读取麦克风数据
        if self.microphone_enabled:
            mic_data = self.microphone_stream.read(self.chunk)
            mic_array = np.frombuffer(mic_data, dtype=np.int16)
            # 应用增益并存储
            self.microphone_frames.append(mic_array.tobytes())
        
        # 读取系统音频数据
        if self.system_audio_enabled:
            sys_data = self.system_audio_stream.read(self.chunk)
            sys_array = np.frombuffer(sys_data, dtype=np.int16)
            # 应用增益并存储
            self.system_audio_frames.append(sys_array.tobytes())
        
        # 混合音频处理
        mixed_data = mic_array + sys_array
        self.frames.append(mixed_data.tobytes())
```

### 2. 独立转写处理

每个音频源都有独立的转写处理逻辑：

- 独立的音频缓冲区管理
- 独立的转写队列
- 独立的临时文件命名（`_mic.wav`, `_sys.wav`）
- 独立的转写结果标识（`[麦克风]`, `[系统音频]`）

### 3. 文件保存策略

```python
def stop_recording(self):
    # 保存独立的音频文件
    saved_files = []
    
    # 1. 保存麦克风音频
    if self.microphone_frames:
        save_audio_file(mic_audio_file, self.microphone_frames)
        saved_files.append(("麦克风", mic_filename, mic_audio_file))
    
    # 2. 保存系统音频
    if self.system_audio_frames:
        save_audio_file(sys_audio_file, self.system_audio_frames)
        saved_files.append(("系统音频", sys_filename, sys_audio_file))
    
    # 3. 合并音频文件
    if len(saved_files) >= 2:
        merge_audio_files(mic_audio_file, sys_audio_file, merged_file)
        saved_files.append(("合并音频", filename, merged_file))
```

### 4. 音频合并功能

```python
def merge_audio_files(self, mic_file, sys_file, output_file, timestamp):
    # 使用pydub加载音频文件
    mic_audio = AudioSegment.from_wav(mic_file)
    sys_audio = AudioSegment.from_wav(sys_file)
    
    # 确保长度一致
    min_length = min(len(mic_audio), len(sys_audio))
    mic_audio = mic_audio[:min_length]
    sys_audio = sys_audio[:min_length]
    
    # 叠加合并
    merged_audio = mic_audio.overlay(sys_audio)
    merged_audio.export(output_file, format="wav")
```

## 文件输出结构

录制完成后，会在 `audio/` 目录下生成以下文件：

```
audio/
├── microphone_20231201_143022.wav    # 麦克风音频
├── system_audio_20231201_143022.wav   # 系统音频
└── recording_20231201_143022.wav      # 合并音频
```

## 转写结果标识

实时转写结果会带有明确的来源标识：

```
[14:30:25][麦克风] 用户说话内容
[14:30:26][系统音频] 系统播放内容
[14:30:27][混合] 混合音频内容（兼容性）
```

## 内存优化

### 1. 独立缓冲区管理

- 每个音频源都有独立的缓冲区大小限制
- 独立的队列大小控制（最多5个待处理项目）
- 独立的内存清理机制

### 2. 同步数据处理

- 确保麦克风和系统音频数据帧数同步
- 在音频流异常时添加静音数据保持同步
- 统一的时间戳管理

### 3. 资源清理

```python
# 清理所有音频数据
self.microphone_frames.clear()
self.system_audio_frames.clear()
self.frames.clear()

# 清理所有缓冲区
self.microphone_buffer.clear()
self.system_audio_buffer.clear()
self.audio_buffer.clear()

# 清理所有转写队列
clear_queue(self.microphone_transcription_queue)
clear_queue(self.system_audio_transcription_queue)
clear_queue(self.transcription_queue)
```

## 兼容性保证

新架构完全保持向后兼容：

1. **原有功能**: 所有原有的录制和转写功能继续工作
2. **混合音频**: 保留原有的混合音频录制逻辑
3. **UI界面**: 用户界面无需修改
4. **配置选项**: 所有现有配置选项继续有效

## 错误处理

### 1. 音频流异常

- 单个音频流异常不影响其他流
- 异常时自动添加静音数据保持同步
- 详细的错误日志记录

### 2. 转写异常

- 独立的转写线程异常处理
- 转写失败不影响录制过程
- 临时文件自动清理

### 3. 合并异常

- 合并失败时自动使用麦克风文件作为备选
- 详细的错误信息记录
- 不影响独立文件的保存

## 性能优化

1. **并行处理**: 麦克风和系统音频并行录制和转写
2. **内存控制**: 独立的缓冲区大小限制
3. **队列管理**: 防止转写队列堆积
4. **资源释放**: 及时清理临时文件和内存数据

## 使用建议

1. **独立转写**: 启用麦克风和系统音频独立转写，获得更精确的来源识别
2. **文件管理**: 根据需要选择使用独立文件或合并文件
3. **内存监控**: 关注内存使用情况，特别是长时间录制时
4. **错误处理**: 注意查看日志中的错误信息，及时处理异常情况

## 技术细节

- **音频格式**: WAV, 16-bit, 单声道
- **采样率**: 可配置（默认44100Hz）
- **缓冲区**: 可配置大小，默认3秒
- **队列限制**: 最多5个待处理项目
- **合并算法**: 音频叠加（overlay）

这个新架构为音频录制和转写提供了更强大、更灵活的功能，同时保持了良好的性能和稳定性。