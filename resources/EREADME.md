## 原始数据集的百度网盘地址
链接:https://pan.baidu.com/s/1vRBGQkZNmAs_5UP567MWlQ
密码:z5k8

## 数据集的格式如下
```
PSCC-Net_dataset/  # 原始数据集根目录
├── authentic/train2014resize_256/        # 真实未篡改图像（如自然场景图、无编辑的原图）
├── splice/fake       # 拼接篡改图像（从其他图像复制区域粘贴到当前图像）
├── splice/mask       # 拼接篡改图像 -标记
├── copymove/fake     # 复制移动篡改图像（同一图像内复制区域并移动位置）
├── copymove/fake     # 复制移动篡改图像-标记
├── removal/fake      # 移除篡改图像（删除图像中的物体并填充背景）
└── removal/fake      # 移除篡改图像 -标记
```

## 筛选图片的代码
chooseData.py

## 筛选后的代码保存路径
resources/dataset

```
resources/
└── dataset/
    ├── train/                # 训练集（80%数据）
    │   ├── Pristine/         # 训练集-真实图像
    │   └── Tampered/         # 训练集-篡改图像（合并splice/copymove/removal）
    └── val/                  # 验证集（20%数据）
        ├── Pristine/         # 验证集-真实图像
        └── Tampered/         # 验证集-篡改图像（合并splice/copymove/removal）
```