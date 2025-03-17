import numpy as np
from PIL import Image
import cv2
import os
from pre_image.pre_process import fft_lbp_rgb

def batch_process_images(input_folder, output_folder):
    """Batch process images in the input folder and save LBP images to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # 只处理特定格式的图像
            img_path = os.path.join(input_folder, filename)
            try:
                img = fft_lbp_rgb(img_path)
                
                # 保存 LBP 图像
                img.save(os.path.join(output_folder, filename))
                print(f"Processed and saved: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# 示例用法
in_str = ["./data1024/train/fake", "./data1024/train/real", "./data1024/test2/fake", "./data1024/test2/real"]
out_str = ['./data1024_pre/train/fake', './data1024_pre/train/real', './data1024_pre/test2/fake',
        './data1024_pre/test2/real']
for i in range(len(in_str)):
    input_folder = in_str[i]  # 输入文件夹
    output_folder = out_str[i]  # 输出文件夹
    batch_process_images(input_folder, output_folder)