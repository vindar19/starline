from PIL import Image
import numpy as np
from pathlib import Path


INPUT = Path("test/test.tga")
OUTPUT_DIR = Path("test_input")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


img = Image.open(INPUT).convert("RGBA")
arr = np.array(img)

rgb = arr[:, :, :3]


# ============================================================
# 1. 黑色轮廓
# ============================================================

black_mask = (
    (rgb[:, :, 0] < 50) &
    (rgb[:, :, 1] < 50) &
    (rgb[:, :, 2] < 50)
)

black_line = np.full_like(rgb, 255)
black_line[black_mask] = 0

Image.fromarray(black_line).save(
    OUTPUT_DIR / "black_line.png"
)


# ============================================================
# 2. 颜色标记图
# ============================================================

# 保留原始颜色，但把白色背景处理成白色
color_guide = rgb.copy()

Image.fromarray(color_guide).save(
    OUTPUT_DIR / "color_guide.png"
)


# ============================================================
# 3. Debug overlay
# ============================================================

overlay = rgb.copy()

# 黑线保持黑色
# 颜色线保持原色
# 白色背景保持白色

Image.fromarray(overlay).save(
    OUTPUT_DIR / "debug_overlay.png"
)


# ============================================================
# 4. 统计
# ============================================================

print("=" * 60)
print("TEST TGA ANALYSIS")
print("=" * 60)

print("size:", img.size)
print("mode:", img.mode)

colors = [
    ((0, 0, 0), "BLACK / OUTLINE"),
    ((255, 0, 0), "RED / HIGHLIGHT"),
    ((0, 0, 255), "BLUE / SHADOW"),
    ((0, 255, 0), "GREEN / DETAIL"),
    ((255, 0, 255), "MAGENTA / DETAIL"),
    ((255, 255, 255), "WHITE / BACKGROUND"),
]

for color, name in colors:
    mask = np.all(rgb == color, axis=2)
    print(
        f"{name:25s}: {int(mask.sum()):8d} pixels"
    )

print()
print("black line output:")
print(OUTPUT_DIR / "black_line.png")

print("color guide output:")
print(OUTPUT_DIR / "color_guide.png")

print("debug overlay:")
print(OUTPUT_DIR / "debug_overlay.png")

print("=" * 60)