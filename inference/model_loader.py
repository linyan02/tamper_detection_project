import torch
from config.paths import MODEL_SAVE_DIR, PRETRAINED_DIR
from models.resnet18 import ResNet18Tamper
from models.pscc_net import PSCCNet

def load_resnet_model(pretrained=False):
    model = ResNet18Tamper(pretrained=pretrained)
    if not pretrained:
        model.load_state_dict(torch.load(MODEL_SAVE_DIR / "resnet18_best.pth", map_location="cpu"))
    model.eval()
    return model

def load_pscc_model(pretrained=False):
    model = PSCCNet()
    if not pretrained:
        model.load_state_dict(torch.load(MODEL_SAVE_DIR / "pscc_net_best.pth", map_location="cpu"))
    model.eval()
    return model