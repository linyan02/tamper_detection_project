import torch
import cv2
import numpy as np
from data.preprocess import get_val_transform
from config.params import IMAGE_SIZE, THRESHOLD

transform = get_val_transform(IMAGE_SIZE)


def predict_single_image(model, image, is_segmentation=False):
    # 图像预处理
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image=image_rgb)["image"].unsqueeze(0)

    # 推理
    with torch.no_grad():
        output = model(input_tensor)

    # 后处理
    if is_segmentation:
        # 定位任务：返回掩码
        mask = output.squeeze().numpy()
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        return (mask > THRESHOLD).astype(np.uint8)
    else:
        # 分类任务：返回标签和置信度
        prob = output.item()
        label = "篡改图像" if prob > THRESHOLD else "真实图像"
        return label, prob