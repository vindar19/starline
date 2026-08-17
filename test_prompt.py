# -*- coding: utf-8 -*-

"""
Starline - 单张动画线稿着色测试

目的：
1. 测试不同 Line2Line ControlNet 权重对线稿一致性的影响
2. 测试赛璐璐平涂风格
3. 保留原始黑色轮廓
4. 固定 seed，保证不同参数之间可以公平比较

输入：
    test_input/black_line.png
    test_input/color_guide.png

输出：
    output/test_prompt/
        ├── l2l_020_raw.png
        ├── l2l_020_final.png
        ├── l2l_040_raw.png
        ├── l2l_040_final.png
        ├── l2l_060_raw.png
        ├── l2l_060_final.png
        ├── l2l_080_raw.png
        ├── l2l_080_final.png
        ├── l2l_100_raw.png
        ├── l2l_100_final.png
        └── comparison.png
"""

import os
import gc
import math
import numpy as np
import torch

from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    UniPCMultistepScheduler,
)


# ============================================================
# 1. 路径
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BLACK_LINE_PATH = os.path.join(
    BASE_DIR,
    "test_input",
    "black_line.png"
)

COLOR_GUIDE_PATH = os.path.join(
    BASE_DIR,
    "test_input",
    "color_guide.png"
)

LINEART_CONTROLNET_PATH = os.path.join(
    BASE_DIR,
    "controlnet",
    "lineart"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output",
    "test_prompt"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. 模型
# ============================================================

BASE_MODEL = "cagliostrolab/animagine-xl-3.1"

LINE2LINE_MODEL = "mattyamonaca/controlnet_line2line_xl"

VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"


# ============================================================
# 3. 测试参数
# ============================================================

# 固定随机种子
SEED = 12345

# SDXL 推理步数
NUM_INFERENCE_STEPS = 30

# CFG
GUIDANCE_SCALE = 5.5

# 测试的 Line2Line 权重
LINE2LINE_SCALES = [
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
]

# Lineart ControlNet 权重
LINEART_SCALE = 1.0

# 输出尺寸
#
# 你的原始 TGA：
# 1674 x 1182
#
# SDXL 更适合接近 1024 的尺寸。
#
# 这里保持原始宽高比例。
TARGET_LONG_SIDE = 1024


# ============================================================
# 4. Prompt
# ============================================================

PROMPT = """
anime character,
Japanese animation cel shading,
traditional Japanese anime coloring,
flat anime coloring,
clean flat colors,
solid color regions,
hard cel-shaded shadows,
hard-edged highlights,
limited color palette,
clean anime paint,
simple clean lighting,
2D animation production artwork,
clean color separation,
preserve the original drawing,
preserve the original character design,
preserve the original lineart,
preserve the original proportions
"""

NEGATIVE_PROMPT = """
photorealistic,
realistic,
3d,
3d render,
CG,
painterly,
oil painting,
watercolor,
soft painting,
brush strokes,
soft shading,
gradient shading,
airbrush,
volumetric lighting,
cinematic lighting,
dramatic lighting,
complex lighting,
glossy rendering,
highly detailed texture,
realistic skin texture,
noise,
grain,
blur,
sketch,
rough drawing,
redrawn lineart,
changed lineart,
modified lineart,
extra lines,
missing lines,
new details,
extra fingers,
extra limbs,
deformed anatomy
"""


# ============================================================
# 5. 工具函数
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_rgb(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"文件不存在：\n{path}"
        )

    img = Image.open(path)

    print(
        f"load: {os.path.basename(path)} | "
        f"format={img.format} | "
        f"mode={img.mode} | "
        f"size={img.size}"
    )

    return img.convert("RGB")


def resize_keep_ratio(img, long_side):
    """
    保持原始宽高比，把长边缩放到指定尺寸。
    """

    w, h = img.size

    scale = long_side / max(w, h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # SDXL latent 通常要求尺寸可以被 8 整除
    new_w = max(8, (new_w // 8) * 8)
    new_h = max(8, (new_h // 8) * 8)

    return img.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )


# ============================================================
# 6. 原始黑线保护
# ============================================================

def composite_original_black_line(
    generated,
    black_line,
    threshold=245,
):
    """
    将原始黑色线稿覆盖回 AI 结果。

    black_line.png：
        黑色 = 轮廓
        白色 = 空白

    threshold：
        小于该值认为属于黑线。

    注意：
        这里不是让 AI 再生成黑线，
        而是直接使用原始线稿。
    """

    generated = generated.convert("RGB")
    black_line = black_line.convert("RGB")

    if generated.size != black_line.size:
        black_line = black_line.resize(
            generated.size,
            Image.Resampling.NEAREST
        )

    gen = np.array(generated).astype(np.uint8)
    line = np.array(black_line).astype(np.uint8)

    # 黑线判断
    line_max = line.max(axis=2)
    line_min = line.min(axis=2)

    black_mask = (
        (line_max < threshold) &
        (line_min < threshold)
    )

    # 原始黑线直接覆盖
    result = gen.copy()

    result[black_mask] = np.array(
        [0, 0, 0],
        dtype=np.uint8
    )

    return Image.fromarray(result)


# ============================================================
# 7. 更严格的黑线保护
# ============================================================

def create_black_line_mask(
    black_line,
    threshold=245,
):
    """
    创建黑线 mask。
    """

    img = np.array(
        black_line.convert("RGB")
    )

    gray = (
        0.299 * img[:, :, 0]
        + 0.587 * img[:, :, 1]
        + 0.114 * img[:, :, 2]
    )

    mask = gray < threshold

    return mask


# ============================================================
# 8. 保存参数信息
# ============================================================

def save_run_info():
    path = os.path.join(
        OUTPUT_DIR,
        "run_info.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("STARLINE TEST\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"BASE_MODEL = {BASE_MODEL}\n")
        f.write(
            f"LINEART_CONTROLNET = "
            f"{LINEART_CONTROLNET_PATH}\n"
        )
        f.write(
            f"LINE2LINE_CONTROLNET = "
            f"{LINE2LINE_MODEL}\n"
        )

        f.write(f"SEED = {SEED}\n")
        f.write(
            f"STEPS = "
            f"{NUM_INFERENCE_STEPS}\n"
        )
        f.write(
            f"CFG = "
            f"{GUIDANCE_SCALE}\n"
        )

        f.write(
            f"LINEART_SCALE = "
            f"{LINEART_SCALE}\n"
        )

        f.write(
            f"LINE2LINE_SCALES = "
            f"{LINE2LINE_SCALES}\n"
        )

        f.write(
            f"TARGET_LONG_SIDE = "
            f"{TARGET_LONG_SIDE}\n"
        )

        f.write("\nPROMPT\n")
        f.write("-" * 60 + "\n")
        f.write(PROMPT.strip())
        f.write("\n\nNEGATIVE PROMPT\n")
        f.write("-" * 60 + "\n")
        f.write(NEGATIVE_PROMPT.strip())


# ============================================================
# 9. 生成比较图
# ============================================================

def make_comparison(results):
    """
    results:
        [
            {
                "scale": 0.2,
                "raw": PIL,
                "final": PIL
            },
            ...
        ]
    """

    if not results:
        return

    # 每组两个图：
    # RAW / FINAL
    #
    # 横向排列。
    thumb_w = 512

    rows = []

    for item in results:

        raw = item["raw"]
        final = item["final"]

        ratio = raw.height / raw.width

        thumb_h = int(
            round(thumb_w * ratio)
        )

        raw = raw.resize(
            (thumb_w, thumb_h),
            Image.Resampling.LANCZOS
        )

        final = final.resize(
            (thumb_w, thumb_h),
            Image.Resampling.LANCZOS
        )

        row = Image.new(
            "RGB",
            (
                thumb_w * 2,
                thumb_h + 50
            ),
            "white"
        )

        row.paste(raw, (0, 50))
        row.paste(final, (thumb_w, 50))

        # 不依赖字体，直接用小标题。
        try:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(row)

            draw.text(
                (10, 15),
                f"L2L={item['scale']:.1f} RAW",
                fill="black"
            )

            draw.text(
                (thumb_w + 10, 15),
                f"L2L={item['scale']:.1f} FINAL",
                fill="black"
            )

        except Exception:
            pass

        rows.append(row)

    total_h = sum(
        row.height
        for row in rows
    )

    comparison = Image.new(
        "RGB",
        (
            thumb_w * 2,
            total_h
        ),
        "white"
    )

    y = 0

    for row in rows:
        comparison.paste(
            row,
            (0, y)
        )
        y += row.height

    output_path = os.path.join(
        OUTPUT_DIR,
        "comparison.png"
    )

    comparison.save(
        output_path,
        quality=95
    )

    print()
    print(
        f"[OK] comparison: "
        f"{output_path}"
    )


# ============================================================
# 10. 主程序
# ============================================================

def main():

    print_section(
        "STARLINE CEL-SHADING LINE CONSISTENCY TEST"
    )

    print(
        f"BASE_DIR:\n{BASE_DIR}"
    )

    print(
        f"\nOUTPUT_DIR:\n{OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # 检查 GPU
    # --------------------------------------------------------

    print_section("DEVICE")

    if torch.cuda.is_available():

        device = "cuda"

        print(
            "CUDA: available"
        )

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA capability:",
            torch.cuda.get_device_capability(0)
        )

    else:

        device = "cpu"

        print(
            "WARNING: CUDA unavailable"
        )

        print(
            "This test will be extremely slow."
        )

    # --------------------------------------------------------
    # 加载输入
    # --------------------------------------------------------

    print_section("LOAD INPUT")

    black_line_original = load_rgb(
        BLACK_LINE_PATH
    )

    color_guide_original = load_rgb(
        COLOR_GUIDE_PATH
    )

    print(
        "\nOriginal size:",
        black_line_original.size
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    print_section("RESIZE")

    black_line = resize_keep_ratio(
        black_line_original,
        TARGET_LONG_SIDE
    )

    color_guide = resize_keep_ratio(
        color_guide_original,
        TARGET_LONG_SIDE
    )

    print(
        "ControlNet input:",
        black_line.size
    )

    # 确保两张图尺寸完全一致
    if color_guide.size != black_line.size:

        color_guide = color_guide.resize(
            black_line.size,
            Image.Resampling.NEAREST
        )

    # --------------------------------------------------------
    # 保存实际输入
    # --------------------------------------------------------

    black_line.save(
        os.path.join(
            OUTPUT_DIR,
            "input_black_line.png"
        )
    )

    color_guide.save(
        os.path.join(
            OUTPUT_DIR,
            "input_color_guide.png"
        )
    )

    # --------------------------------------------------------
    # 参数记录
    # --------------------------------------------------------

    save_run_info()

    # --------------------------------------------------------
    # 加载 ControlNet
    # --------------------------------------------------------

    print_section(
        "LOAD CONTROLNETS"
    )

    dtype = torch.float16

    print(
        "\nLoading Lineart ControlNet..."
    )

    lineart_controlnet = (
        ControlNetModel.from_pretrained(
            LINEART_CONTROLNET_PATH,
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=True,
        )
    )

    print(
        "[OK] Lineart ControlNet"
    )

    print(
        "\nLoading Line2Line ControlNet..."
    )

    line2line_controlnet = (
        ControlNetModel.from_pretrained(
            LINE2LINE_MODEL,
            torch_dtype=dtype,
        )
    )

    print(
        "[OK] Line2Line ControlNet"
    )

    # --------------------------------------------------------
    # VAE
    # --------------------------------------------------------

    print_section("LOAD VAE")

    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL,
        torch_dtype=dtype,
    )

    print(
        "[OK] VAE"
    )

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    print_section(
        "LOAD SDXL PIPELINE"
    )

    controlnets = [
        lineart_controlnet,
        line2line_controlnet,
    ]

    pipe = (
        StableDiffusionXLControlNetPipeline
        .from_pretrained(
            BASE_MODEL,
            controlnet=controlnets,
            vae=vae,
            torch_dtype=dtype,
        )
    )

    # UniPC scheduler
    pipe.scheduler = (
        UniPCMultistepScheduler.from_config(
            pipe.scheduler.config
        )
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if device == "cuda":

        pipe = pipe.to(
            "cuda"
        )

        # 关闭不必要的安全检查，
        # 对本地测试速度有帮助。
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    else:

        pipe = pipe.to(
            "cpu"
        )

    # --------------------------------------------------------
    # xformers
    # --------------------------------------------------------

    try:

        pipe.enable_xformers_memory_efficient_attention()

        print(
            "[OK] xFormers enabled"
        )

    except Exception as e:

        print(
            "[INFO] xFormers unavailable:",
            str(e)
        )

    # --------------------------------------------------------
    # Generator
    # --------------------------------------------------------

    if device == "cuda":

        generator = torch.Generator(
            device="cuda"
        ).manual_seed(SEED)

    else:

        generator = torch.Generator(
            device="cpu"
        ).manual_seed(SEED)

    # --------------------------------------------------------
    # 生成
    # --------------------------------------------------------

    print_section(
        "START TEST"
    )

    print(
        "Prompt:"
    )

    print(
        PROMPT.strip()
    )

    print(
        "\nNegative prompt:"
    )

    print(
        NEGATIVE_PROMPT.strip()
    )

    print(
        "\nSeed:",
        SEED
    )

    print(
        "Steps:",
        NUM_INFERENCE_STEPS
    )

    print(
        "CFG:",
        GUIDANCE_SCALE
    )

    print(
        "Lineart:",
        LINEART_SCALE
    )

    results = []

    for index, l2l_scale in enumerate(
        LINE2LINE_SCALES,
        start=1
    ):

        print()
        print(
            "-" * 70
        )

        print(
            f"TEST {index}/{len(LINE2LINE_SCALES)}"
        )

        print(
            f"Lineart ControlNet : "
            f"{LINEART_SCALE}"
        )

        print(
            f"Line2Line ControlNet : "
            f"{l2l_scale}"
        )

        print(
            "-" * 70
        )

        # ----------------------------------------------------
        # 每次重新创建 generator
        #
        # 保证每组参数使用完全相同的随机噪声
        # ----------------------------------------------------

        if device == "cuda":

            generator = torch.Generator(
                device="cuda"
            ).manual_seed(SEED)

        else:

            generator = torch.Generator(
                device="cpu"
            ).manual_seed(SEED)

        # ----------------------------------------------------
        # ControlNet conditioning
        # ----------------------------------------------------

        conditioning_scales = [
            LINEART_SCALE,
            l2l_scale,
        ]

        # ----------------------------------------------------
        # ControlNet 输入
        #
        # 两个 ControlNet 都使用相同的线稿输入。
        #
        # 这里暂时不把 color_guide 直接作为第二张 ControlNet
        # 的 conditioning image。
        #
        # 因为当前阶段我们首先测试：
        # “纯线稿一致性”。
        # ----------------------------------------------------

        control_images = [
            black_line,
            black_line,
        ]

        # ----------------------------------------------------
        # 推理
        # ----------------------------------------------------

        with torch.inference_mode():

            result = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,

                image=control_images,

                controlnet_conditioning_scale=
                    conditioning_scales,

                num_inference_steps=
                    NUM_INFERENCE_STEPS,

                guidance_scale=
                    GUIDANCE_SCALE,

                generator=generator,

                width=black_line.width,
                height=black_line.height,

            ).images[0]

        # ----------------------------------------------------
        # 保存 RAW
        # ----------------------------------------------------

        scale_name = (
            f"{int(round(l2l_scale * 100)):03d}"
        )

        raw_path = os.path.join(
            OUTPUT_DIR,
            f"l2l_{scale_name}_raw.png"
        )

        result.save(
            raw_path
        )

        print(
            "[OK] RAW:",
            raw_path
        )

        # ----------------------------------------------------
        # 原始黑线硬覆盖
        # ----------------------------------------------------

        final = composite_original_black_line(
            result,
            black_line,
            threshold=245
        )

        final_path = os.path.join(
            OUTPUT_DIR,
            f"l2l_{scale_name}_final.png"
        )

        final.save(
            final_path
        )

        print(
            "[OK] FINAL:",
            final_path
        )

        results.append(
            {
                "scale": l2l_scale,
                "raw": result,
                "final": final,
            }
        )

        # ----------------------------------------------------
        # 清理显存
        # ----------------------------------------------------

        if device == "cuda":

            torch.cuda.empty_cache()

        gc.collect()

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print_section(
        "CREATE COMPARISON"
    )

    make_comparison(
        results
    )

    # --------------------------------------------------------
    # 完成
    # --------------------------------------------------------

    print_section(
        "TEST COMPLETE"
    )

    print(
        f"\nResults:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\n重点查看："
    )

    print(
        "comparison.png"
    )

    print(
        "\n比较顺序："
    )

    print(
        "L2L 0.2"
    )

    print(
        "L2L 0.4"
    )

    print(
        "L2L 0.6"
    )

    print(
        "L2L 0.8"
    )

    print(
        "L2L 1.0"
    )

    print(
        "\nRAW = AI 原始结果"
    )

    print(
        "FINAL = 原始黑线重新覆盖后的结果"
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()