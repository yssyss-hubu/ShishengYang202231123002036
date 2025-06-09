import numpy as np
import rasterio
from PIL import Image
import os


def scale_to_255(arr, max_val=10000):
    """将原始值从 0–10000 压缩到 0–255"""
    arr_scaled = np.clip((arr / max_val) * 255, 0, 255)
    return arr_scaled.astype(np.uint8)


def save_image_from_bands(red, green, blue, output_path, normalize=True, max_val=10000):
    if normalize:
        # 真彩色图像正则化
        rgb = np.dstack((red, green, blue)).astype(float)
        array_min, array_max = rgb.min(), rgb.max()
        rgb_normalized = ((rgb - array_min) / (array_max - array_min)) * 255
        rgb_normalized = rgb_normalized.astype(np.uint8)
    else:
        # 使用线性压缩（固定最大值）
        red = scale_to_255(red, max_val)
        green = scale_to_255(green, max_val)
        blue = scale_to_255(blue, max_val)
        rgb_normalized = np.dstack((red, green, blue))

    img = Image.fromarray(rgb_normalized, mode='RGB')
    img.save(output_path)
    print(f"✅ 图像已保存：{output_path}")


def create_rgb_from_separate_tifs(red_path, green_path, blue_path, output_path, max_val=10000):
    def load_band(path):
        with rasterio.open(path) as src:
            return src.read(1)

    red = load_band(red_path)
    green = load_band(green_path)
    blue = load_band(blue_path)
    save_image_from_bands(red, green, blue, output_path, normalize=False, max_val=max_val)


def create_rgb_from_multiband_tif(tif_file, output_path):
    with rasterio.open(tif_file) as src:
        bands = src.read()
        # 假设波段顺序为 B02, B03, B04, B08, B12
        blue = bands[0]
        green = bands[1]
        red = bands[2]
        save_image_from_bands(red, green, blue, output_path, normalize=True)


# 示例用法
if __name__ == "__main__":
    mode = "separate"  # 可选 "multi" 或 "separate"

    if mode == "multi":
        multiband_file = "E:\desktop\demo1"
        output_file = "rgb_from_multiband.jpg"
        create_rgb_from_multiband_tif(multiband_file, output_file)
    else:
        red_band_path = "B04.tif"
        green_band_path = "B03.tif"
        blue_band_path = "B02.tif"
        output_file = "rgb_from_separate.jpg"
        create_rgb_from_separate_tifs(red_band_path, green_band_path, blue_band_path, output_file)
