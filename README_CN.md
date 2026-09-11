<div align="center">

# SeetaPsych Hertz

<img src="website/public/media/tinyhr-logo.png" width="150" alt="SeetaPsych Hertz 标志">

### 看见脉搏，始于视频

基于普通人脸视频和轻量级 rPPG 模型的非接触式心率估计工具。

[English](README.md) | [简体中文](README_CN.md)

[项目简介](#项目简介) · [安装](#安装) · [演示](#演示) · [数据集](#训练数据) · [性能测试](#模型规模与延迟) · [技术报告](website/public/downloads/tinyhr-technical-report.pdf)

[![TinyHR 演示：人脸视频、预测脉搏波形和心率估计](website/public/media/tinyhr-demo.gif)](website/public/media/demo-full.mp4)

*看 TinyHR 如何把人脸视频变成实时脉搏波形；点击可观看完整演示。*

[![Python](https://img.shields.io/badge/Python-3.10%2B-2563D8?logo=python&logoColor=white)](pyproject.toml)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU%20%7C%20GPU-091D31?logo=onnx&logoColor=white)](seetapsych_hertz/modules/tiny-hr.yml)
[![TinyHR](https://img.shields.io/badge/TinyHR-82%2C177_params-FB5F14)](seetapsych_hertz/modules/tiny-hr.yml)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-75E5C9)](LICENSE)

</div>

## 项目简介

SeetaPsych Hertz 为
[SeetaPsych](https://github.com/seetapsych/seetapsych-lib) 生态提供心率估计模块。
其中的 TinyHR 模型从人脸皮肤的细微颜色变化中恢复 rPPG 波形，再通过滤波与频谱分析
将波形转换为心率。

TinyHR 面向实际部署设计：发布的 ONNX 模型仅含 **82,177 个参数**，文件大小仅
**381 KiB**；训练数据来自 4 个数据集，共覆盖 **1,288 名受试者和 6,307 段视频**。
完成初始视频窗口积累后，默认流式管线每秒输出一次新估计。

## 核心优势

| | 特性 | 实际价值 |
|---|---|---|
| 🌍 | **多来源大规模训练数据** | 4 个 rPPG 数据集覆盖 1,288 名受试者和 6,307 段训练视频，增加人群与拍摄条件的多样性。 |
| 🪶 | **8.2 万参数轻量模型** | ONNX 模型仅 381 KiB，适合资源受限设备与边缘部署场景。 |
| ⚡ | **计算快、滚动响应及时** | Apple M4 CPU 实测 TinyHR 管线约 132 ms，预热后每 1 秒刷新一次心率估计。 |
| 📹 | **非接触式测量** | 使用普通 RGB 摄像头采集人脸视频，估计过程不依赖穿戴式传感器。 |
| 📈 | **先波形、后心率** | 模型先预测可检查的 rPPG 波形，再通过确定性的信号处理得到心率。 |
| 🧩 | **集成 SeetaPsych** | 已提供可用于视频文件和实时视频流的 SeetaPsych 模块配置。 |

## 核心数据

| 训练数据集 | 训练受试者 | 训练视频 | 参数量 | 模型大小 | 测试 MAE |
|---:|---:|---:|---:|---:|---:|
| **4** | **1,288** | **6,307** | **82,177** | **381 KiB** | **3.88 BPM** |

## 演示

演示画面同时显示检测到的人脸、预测的 BVP 波形与心率估计。该视频用于说明处理流程；
模型精度请参见下文实验结果。

## 训练数据

TinyHR 使用 4 个互补的 rPPG 数据集训练。下表完整复现所附技术报告中的数据集表格。

| 数据集 | 受试者 | 视频 | 用途 |
|---|---:|---:|---|
| VIPL-HR V1 | 85 | 1,883 | 训练 |
| VIPL-HR V2 | 500 | 2,498 | 训练 |
| V4V | 103 | 726 | 训练 |
| MCD-rPPG | 600 | 1,200 | 训练；仅使用正脸视频 |
| **合计** | **1,288** | **6,307** | **四来源训练数据** |

独立的 VIPL-HR V1 测试集包含 **22 名受试者和 485 段视频**。技术报告明确说明该测试集
只用于评估，没有参与模型训练。

| 训练配置 | 设置 |
|---|---|
| 批大小 | 4 |
| 初始学习率 | 0.005 |
| 学习率调度器 | OneCycleLR |
| 输入视频片段 | 160 帧 RGB 图像，分辨率 128 × 128 |

来源：[TinyHR 技术报告第 6-7 页](website/public/downloads/tinyhr-technical-report.pdf)。

## 模型规模与延迟

| 测试项 | 结果 | 说明 |
|---|---:|---|
| ONNX 参数量 | **82,177** | 根据发布模型中的初始化张量统计 |
| ONNX 文件大小 | **381 KiB** | SHA-256 与 `tiny-hr.yml` 声明的模型地址一致 |
| TinyHR 处理延迟 | **平均 132 ms** | 包含输入归一化、ONNX 推理、波形归一化和心率信号处理 |
| 中位数 / P95 延迟 | **125 / 164 ms** | ONNX Runtime CPUExecutionProvider 预热后运行 50 次 |
| 首次观测窗口 | **30 FPS 时约 5.3 秒** | 首次估计前需要收集 160 帧 |
| 滚动更新间隔 | **默认 1.0 秒** | 30 FPS 时每 30 帧请求一次新估计 |

测试环境：Apple M4 MacBook Air、10 核 CPU、24 GB 内存、ONNX Runtime 1.30.0、
CPUExecutionProvider，测试日期 2026 年 9 月 11 日。计算延迟不包含摄像头采集和人脸检测，
这两部分会随检测器与部署硬件变化。

实际使用时，需要区分观测时间和计算时间：模型先用一段短视频获得足够的时序生理信息；
窗口准备好后，亚秒级 TinyHR 计算可稳定落在默认的一秒滚动更新周期内。因此，该模型适合
实时健康交互界面、人机交互、情感计算研究以及轻量级远程监测原型。

## TinyHR 工作原理

[![TinyHR 架构和推理流程](website/public/media/tinyhr-flowchart.png)](website/public/downloads/tinyhr-flowchart.pdf)

*点击图片可打开完整架构 PDF。*

| 阶段 | 模块 | 功能 |
|---:|---|---|
| 01 | 帧差融合 Stem | 将相邻帧差异转换为紧凑空间特征，突出细微的时序颜色变化。 |
| 02 | 空间 Patch Embedding | 将每帧特征图从 32 × 32 压缩到 8 × 8，同时保留时间序列。 |
| 03 | 多尺度时间特征块 | 使用多个时间偏移与残差特征融合捕获不同尺度的生理变化。 |
| 04 | 波形预测头 | 为每个输入视频帧生成一个 rPPG 波形采样值。 |
| 05 | 信号处理 | 通过去趋势、带通滤波和 Welch 频谱分析获得 BPM。 |

推理阶段保留 0.75-2.5 Hz 的生理频带，对应 45-150 BPM。滤波后波形的主频按下式转换：

```text
心率（BPM）= 60 × 主频（Hz）
```

训练目标同时约束波形一致性、频域分类和心率分布：

```text
L = 0.2 L_time + L_CE + L_KL
```

## 实验结果

| 数据集 | 测试集 | 实验方案 | MAE |
|---|---|---|---:|
| VIPL-HR V1 | 22 名受试者 · 485 段视频 | 与训练集完全隔离 | **3.88 BPM** |

该结果来自所附技术报告，是指定测试集上的作者报告结果，并非对每段视频误差的保证值。

## 安装

本项目已包含在 SeetaPsych 默认配置中，可通过以下命令下载模块：

```bash
seetapsych-manager download
```

完整框架使用方法请参见
[SeetaPsych](https://github.com/seetapsych/seetapsych-lib)。

## 使用方法

### WebUI

```bash
seetapsych-webui --files seetapsych_hertz/modules/seeta.yml
```

### 代码调用

```python
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.pipeline import Pipeline

factory = Factory()
factory.load_file_modules("seetapsych_hertz/modules/seeta.yml")

pipeline = Pipeline(factory, ...)
pipeline.add_attributes("face/heart_rate")
```

## 模块库

| 模块 | 说明 | 输入方式 |
|---|---|---|
| [AdaChrom](seetapsych_hertz/modules/ada-chrom.yml) | 基于皮肤 ROI 自适应色度分析的无模型 rPPG 方法 | 视频流 · 视频文件 |
| [Seeta](seetapsych_hertz/modules/seeta.yml) | Seeta 心率估计模块 | 视频流 · 视频文件 |
| [TinyHR](seetapsych_hertz/modules/tiny-hr.yml) | 结合 Welch 频谱分析的轻量级波形模型 | 视频流 · 视频文件 |

### TinyHR 参数

| 名称 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `fps` | number | `30` | 用于缓冲和频谱分析的预期视频帧率 |
| `interval` | number | `1` | 连续两次滚动心率估计之间的秒数 |

### AdaChrom 参数

| 名称 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `window_samples` | integer | `300` | 滑动窗口帧数；增大可降低噪声，但会延长响应时间 |
| `roi_regions` | `selection[]` | `["skin_b_adaptive_forehead"]` | 在有效结果融合前分别进行估计的皮肤区域 |

两个模块均提供
[`face/heart_rate`](https://github.com/seetapsych/seetapsych-attributes#faceheart_rate)
属性。

## 项目资源

- [完整演示视频](website/public/media/demo-full.mp4)
- [TinyHR 技术报告](website/public/downloads/tinyhr-technical-report.pdf)
- [模型架构图](website/public/downloads/tinyhr-flowchart.pdf)
- [交互式项目主页源码](website/)
- Hugging Face 模型发布与交互式演示正在规划中。

## 开源许可

本项目使用 [BSD 3-Clause License](LICENSE) 发布。
