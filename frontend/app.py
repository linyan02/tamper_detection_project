import gradio as gr
import cv2
import numpy as np
from inference.model_loader import load_resnet_model, load_pscc_model
from inference.predictor import predict_single_image

# 加载模型
resnet_model = load_resnet_model()
pscc_model = load_pscc_model()


def detect(image, model_choice):
    if image is None:
        return "请上传图像", "", None

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if model_choice == "ResNet18（分类）":
        label, prob = predict_single_image(resnet_model, image_bgr)
        return label, f"置信度: {prob:.4f}", None
    else:
        label, prob = predict_single_image(resnet_model, image_bgr)
        mask = predict_single_image(pscc_model, image_bgr, is_segmentation=True)
        # 生成热力图
        heatmap = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        return label, f"置信度: {prob:.4f}", overlay


def launch_app(**kwargs):
    with gr.Blocks(title="图像篡改检测系统") as demo:
        gr.Markdown("# 图像篡改检测学习系统")
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="numpy", label="上传图像")
                model_choice = gr.Radio(
                    ["ResNet18（分类）", "PSCC-Net（带定位）"],
                    label="模型选择",
                    value="ResNet18（分类）"
                )
                detect_btn = gr.Button("开始检测")
            with gr.Column(scale=1):
                label_output = gr.Textbox(label="检测结果")
                prob_output = gr.Textbox(label="置信度")
                heatmap_output = gr.Image(label="篡改区域热力图", visible=False)

        detect_btn.click(
            fn=detect,
            inputs=[image_input, model_choice],
            outputs=[label_output, prob_output, heatmap_output]
        )
        model_choice.change(
            fn=lambda x: gr.update(visible=(x == "PSCC-Net（带定位）")),
            inputs=model_choice,
            outputs=heatmap_output
        )

    demo.launch(**kwargs)