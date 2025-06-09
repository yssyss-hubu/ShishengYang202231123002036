import numpy as np
import rasterio
from PIL import Image


def scale_to_255(arr, max_val=10000):
    """将0-10000压缩到0-255"""
    arr_scaled = np.clip((arr / max_val) * 255, 0, 255)
    return arr_scaled.astype(np.uint8)


def create_rgb_from_sentinel_multiband(tif_file, output_path, max_val=10000):
    with rasterio.open(tif_file) as src:
        bands = src.read()  # (5, H, W)

        # 读取并压缩 B04（红）、B03（绿）、B02（蓝）
        red = scale_to_255(bands[2], max_val)  # B04
        green = scale_to_255(bands[1], max_val)  # B03
        blue = scale_to_255(bands[0], max_val)  # B02

        # 合并为 RGB 图像
        rgb = np.dstack((red, green, blue))

        # 保存为图像
        img = Image.fromarray(rgb, mode='RGB')
        img.save(output_path)
        print(f"✅ 已成功保存为 RGB 图像：{output_path}")


# ✅ 示例使用
if __name__ == "__main__":
    input_tif = r"E:\desktop\demo1.tif"  # 改为你的路径
    output_img = "sentinel2_rgb_output.jpg"
    create_rgb_from_sentinel_multiband(input_tif, output_img)
