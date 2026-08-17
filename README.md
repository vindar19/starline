# Starline — Anime Line-art Coloring Assistant

本项目基于 [mattyamonaca/starline](https://github.com/mattyamonaca/starline) 进行本地化开发，目标是将 Stable Diffusion XL + ControlNet 应用于日本动画制作中的**线稿辅助上色**。

> 当前项目仍处于开发与验证阶段。现阶段重点不是完全自动上色，而是验证 AI 能否理解动画着色流程中的线稿、颜色标注以及区域关系，并逐步建立适合实际动画作画/上色工作的辅助工具。

## 1. 项目背景

我的实际工作流程使用 **RETAS STUDIO PaintMan**。

动画上色用的 `.tga` 线稿文件在 PaintMan 中具有两个相互作用的平面：

- **轮廓平面（Outline Plane）**
  - 保存黑色实线轮廓。
- **颜料平面（Paint Plane）**
  - 保存用于指导上色的颜色标记线。

这些颜色标记不是普通意义上的“画面颜色”，而是动画制作流程中的**上色指示信息**。

目前测试文件 `test.tga` 中使用的规则：

| 颜色 | 含义 |
|---|---|
| 黑色 | 角色/物体轮廓 |
| 红色 | 高光范围 |
| 蓝色 | 阴影范围 |
| 绿色 | 物体细节等辅助信息 |
| 品红色 | 物体细节等辅助信息 |
| 白色 | 透明背景在当前测试输入中的可视化底色 |

一个非常重要的特点是：

**轮廓平面和颜料平面在 PaintMan 中会共同决定最终的填色区域。**

例如：

- 红色辅助线与黑色轮廓共同围成的区域 → 填充高光色。
- 蓝色辅助线与黑色轮廓共同围成的区域 → 填充阴影色。
- 填充后，辅助颜色线会被填充颜色“吃掉”，最终成为该区域的实际颜色。

因此，本项目最终需要解决的并不是简单的“给黑白线稿上色”，而是：

> 理解动画线稿中的结构、颜色标注及其与黑色轮廓之间的区域关系，并生成符合动画制作逻辑的颜色结果。

## 2. 当前目标

当前开发阶段首先进行**单张线稿一致性测试**。

测试目的：

1. 输入实际动画工作流程中的 `.tga` 线稿。
2. 提取黑色轮廓作为 ControlNet 的主要结构条件。
3. 保留/分析红、蓝、绿、品红等颜色标记。
4. 使用提示词控制目标画面风格和上色意图。
5. 观察 Stable Diffusion XL + ControlNet 是否能够：
   - 保持原线稿结构；
   - 正确理解角色和物体关系；
   - 根据提示词完成合理的动画风格上色；
   - 尽可能遵守红色高光、蓝色阴影等区域提示；
   - 减少对原始轮廓的破坏。

当前阶段主要用于**验证模型与线稿的一致性**，后续才会进一步考虑自动化的颜色区域控制、PaintMan 工作流兼容以及批量处理。

## 3. 当前模型组件

目前已经成功验证以下 ControlNet 模型可以正常加载：

### ControlNet Lineart XL

本地目录：

```text
controlnet/lineart/
```

该模型用于提供线稿/结构条件。

### ControlNet Line2Line XL

模型：

```text
mattyamonaca/controlnet_line2line_xl
```

该模型已经成功下载并通过 `diffusers.ControlNetModel.from_pretrained()` 加载测试。

测试结果：

```text
LINE2LINE LOAD OK
<class 'diffusers.models.controlnet.ControlNetModel'>
dtype: torch.float16
```

### SDXL Base / Animagine XL

项目当前围绕 SDXL 系列模型进行测试，并使用 Animagine XL 3.1 等模型作为动画风格生成基础。

实际模型文件较大，因此**不会提交到 GitHub 仓库**。

## 4. 测试输入

当前测试文件：

```text
test/
└── test.tga
```

`test.tga` 的基本信息：

```text
format: TGA
mode: RGBA
size: (1674, 1182)
bands: ('R', 'G', 'B', 'A')
```

Alpha 通道当前为完全不透明：

```text
alpha min: 255
alpha max: 255
alpha mean: 255.0
alpha=0: 0
alpha=255: 1978668
```

需要注意：

**虽然像素级 Alpha 通道为 255，但在 PaintMan 中打开该 TGA 时，实际工作背景表现为透明背景。**

RGB 像素统计结果：

```text
(255, 255, 255) 1959965
(0, 0, 0)          10933
(255, 0, 0)         2859
(0, 0, 255)         2612
(0, 255, 0)         1581
(255, 0, 255)        718
```

因此当前测试文件明确包含：

- 黑色轮廓；
- 红色高光标记；
- 蓝色阴影标记；
- 绿色细节标记；
- 品红色细节标记；
- 白色背景区域。

## 5. 当前预处理流程

已经准备：

```text
prepare_test.py
```

运行：

```bash
python prepare_test.py
```

会分析 TGA，并生成：

```text
test_input/
├── black_line.png
├── color_guide.png
└── debug_overlay.png
```

其中：

### `black_line.png`

提取黑色轮廓，用于后续 ControlNet 结构控制。

### `color_guide.png`

提取红、蓝、绿、品红等颜色标记，用于保存动画着色流程中的颜色提示信息。

### `debug_overlay.png`

用于检查黑色轮廓与颜色标记的位置关系是否正确。

## 6. 当前测试脚本

已经准备：

```text
test_prompt.py
```

当前阶段通过提示词进行测试，主要目的为：

> 在不修改原始动画线稿结构的前提下，让模型根据线稿和提示词生成一张动画风格的彩色结果，以检查模型对线稿结构的一致性。

生成结果放在：

```text
output/
```

由于生成结果、缓存和模型文件体积较大，以下内容不应该提交到 Git：

```text
output/
*.safetensors
```

## 7. GUI

项目已经可以运行本地 Gradio GUI：

```bash
python app.py
```

当前已经确认 GUI 可以正常启动：

```text
Running on local URL: http://127.0.0.1:7860
```

目前 GUI 主要作为模型测试和后续交互式开发基础。

## 8. Windows / Hugging Face 缓存

由于 SDXL 和 ControlNet 模型体积很大，开发过程中曾导致系统盘空间明显下降。

目前已经将 Hugging Face 缓存迁移到：

```text
D:\huggingface_cache
```

并设置了用户环境变量：

```text
HF_HOME=D:\huggingface_cache
HF_HUB_CACHE=D:\huggingface_cache\hub
```

因此后续模型下载和缓存应优先使用 D 盘。

**不要把 Hugging Face 模型缓存提交到 GitHub。**

## 9. Git / GitHub

当前个人 Fork：

```text
https://github.com/vindar19/starline
```

本地 Git `origin` 已经修改为个人仓库：

```text
https://github.com/vindar19/starline.git
```

原始上游项目：

```text
https://github.com/mattyamonaca/starline
```

当前开发是在 Fork 的基础上继续修改。

## 10. 当前 Git 中准备提交的核心内容

目前新增/修改的主要开发文件包括：

```text
prepare_test.py
test_prompt.py
download_hf_range.py

test/
test_input/

controlnet/lineart/
├── config.json
├── README.md
└── .gitattributes
```

其中模型权重、生成结果和临时缓存不会提交。

当前 `.gitignore` 已包含：

```gitignore
output/
*.safetensors
```

并且模型权重相关文件目前被 Git 正确识别为 ignored。

## 11. 当前开发阶段

目前项目进度可以概括为：

```text
原始 Starline
      │
      ▼
Fork 到个人 GitHub
      │
      ▼
Windows 本地开发环境
      │
      ▼
SDXL + ControlNet 模型准备完成
      │
      ├── Lineart XL      ✓
      └── Line2Line XL    ✓
      │
      ▼
实际动画 TGA 测试
      │
      ├── 黑色轮廓提取       ✓
      ├── 颜色标记分析       ✓
      ├── 测试输入生成       ✓
      └── 提示词上色测试     ✓
      │
      ▼
Gradio GUI
      │
      └── 本地启动正常       ✓
      │
      ▼
下一阶段：线稿一致性评估
      │
      ▼
颜色标记 / 高光 / 阴影区域控制
      │
      ▼
PaintMan 工作流兼容
      │
      ▼
批量动画上色辅助
```

## 12. 后续开发重点

后续不会直接把项目变成普通的 AI 图像生成器，而是围绕**动画制作中的实际着色流程**逐步改造。

优先级：

1. **验证线稿一致性**
   - 黑色轮廓是否能够稳定保持；
   - AI 是否会改变角色结构；
   - 是否出现断线、漏线、错误补线。

2. **让颜色标记真正参与生成**
   - 红色 → 高光区域；
   - 蓝色 → 阴影区域；
   - 绿色/品红 → 细节或特定物体提示。

3. **研究轮廓平面 + 颜料平面的联合控制**
   - 不是简单将颜色线直接当成 RGB 图像输入；
   - 而是理解“黑色轮廓 + 颜色标记 = 一个需要填充的区域”。

4. **提高动画风格稳定性**
   - 保持角色设计；
   - 保持物体结构；
   - 保持原画线稿；
   - 减少 AI 自由发挥。

5. **最终目标**
   - 输入 PaintMan 工作流程中的 TGA；
   - 自动分析轮廓与颜色指示；
   - AI 生成辅助上色结果；
   - 尽可能输出可继续进入 PaintMan 修改的结果；
   - 支持动画生产中的批量处理。

## 13. 开发原则

这个项目的核心不是追求单张图片“看起来漂亮”，而是追求：

**结构准确 > 着色逻辑准确 > 风格一致 > 视觉效果**

对于动画生产环境而言，AI 输出必须服从原始线稿和上色指示，而不是为了生成漂亮图片随意修改原画结构。

---

## Upstream

Original project:

https://github.com/mattyamonaca/starline

Personal fork:

https://github.com/vindar19/starline
