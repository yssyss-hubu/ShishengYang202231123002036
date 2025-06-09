import numpy as np
import rasterio
from PIL import Image
import os

def load_band(path):
    with rasterio.open(path) as src:
        return src.read(1)

def scale_to_255(arr, max_val=10000):
    """将原始值从 0–10000 压缩到 0–255"""
    arr_scaled = np.clip((arr / max_val) * 255, 0, 255)
    return arr_scaled.astype(np.uint8)

def save_rgb_image(red_path, green_path, blue_path, output_path, max_val=10000):
    # 加载每个波段
    red = load_band(red_path)
    green = load_band(green_path)
    blue = load_band(blue_path)

    # 缩放到 8 位图像
    red_scaled = scale_to_255(red, max_val)
    green_scaled = scale_to_255(green, max_val)
    blue_scaled = scale_to_255(blue, max_val)

    # 合并为 RGB 图像
    rgb_image = np.dstack((red_scaled, green_scaled, blue_scaled))

    # 保存为 JPG 或 PNG
    img = Image.fromarray(rgb_image, mode='RGB')
    img.save(output_path)
    print(f"保存成功：{output_path}")

if __name__ == "__main__":
    # 设置文件路径（根据你自己的路径调整）
    red_band_path = "B04.tif"   # 红波段（B4）
    green_band_path = "B03.tif" # 绿波段（B3）
    blue_band_path = "B02.tif"  # 蓝波段（B2）
    output_file = "sentinel_rgb_output.jpg"

    # 创建 RGB 图像
    save_rgb_image(red_band_path, green_band_path, blue_band_path, output_file)
