import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from sign_detection.models import TinyConvNet

model = TinyConvNet(3, 32)
model.load_state_dict(torch.load("pretrained/tiny-convnet.pt"))
model.eval()

scripted_model = torch.jit.script(model)
optimized_model = optimize_for_mobile(scripted_model)
optimized_model._save_for_lite_interpreter('pretrained/lite-tiny-convnet.ptl')

print("model successfully exported.")
