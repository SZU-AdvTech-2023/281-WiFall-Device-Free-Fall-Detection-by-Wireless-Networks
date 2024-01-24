import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# 定义伽马变换函数
def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    return image.point(table)

# 定义灰度拉伸函数
def grayscale_stretch(image, A, B):
    return image.point(lambda x: (255 / (B - A)) * (x - A) if B != A else 0)

# 定义提高对比度函数
def enhance_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

# 自适应阈值处理
def adaptive_threshold(image, blockSize, C):
    open_cv_image = np.array(image)
    # 确保图像是单通道的灰度图，如果不是，需要先转换
    if len(open_cv_image.shape) == 3:
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    # 应用自适应阈值
    return cv2.adaptiveThreshold(open_cv_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)


# 处理指定文件夹中的所有图像
folders = ['0', '1', '2', '3']
base_path = 'C:/Users\dell\Desktop\shiping\solar-picture-classification-master\pic'  # 假设数据文件夹位于这个路径

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    for img_file in os.listdir(folder_path):
        if img_file.endswith('.png') or img_file.endswith('.jpg'):
            img_path = os.path.join(folder_path, img_file)
            with Image.open(img_path) as img:
                img = img.convert('L')  # 确保图像是灰度图

                # 伽马变换
                gamma_corrected_img = gamma_correction(img, gamma=2.2)  # 假设伽马值为2.2

                # 转换为数组进行灰度拉伸
                gamma_corrected_array = np.array(gamma_corrected_img)
                A = np.min(gamma_corrected_array)
                B = np.max(gamma_corrected_array)
                stretched_img = grayscale_stretch(gamma_corrected_img, A, B)

                # 自适应阈值
                thresholded_img = adaptive_threshold(stretched_img, blockSize=11, C=2)

                # 提高对比度（如果需要）
                enhanced_img = Image.fromarray(thresholded_img)
                enhanced_img = enhance_contrast(enhanced_img, factor = 1.5)  # 根据需要调整

                # 保存图像
                enhanced_img.save(img_path)

print("图像处理完成。")








