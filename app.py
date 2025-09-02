import gradio as gr
import torch
from PIL import Image
import numpy as np
from app.utils import recover_light_sources
from model.model import model
from torchvision import transforms


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
chk=torch.load('model/model_epoch_49.pth',map_location=device)
model.load_state_dict(chk['model_state_dict'])

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),  # -> (C,H,W), dtype=float32, range [0,1]
])

def evaluate(model, image):
    """
    Run the model on the given image tensor and return output as numpy array (H,W,C) in [0,255].
    """
    model.eval()
    with torch.no_grad():
        #image = image.to(device, dtype=torch.float32)
        outputs = model(image)
        outputs = torch.clamp(outputs, 0.0, 1.0)
        outputs_np = outputs.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (outputs_np * 255).astype(np.uint8)  # Convert to uint8 for recovery step

def predict(input_image):
    """
    Predict clean image from flare image, then recover light sources.
    """
    # Resize and prepare input tensor
    input_img = input_image.convert('RGB').resize((512, 512), Image.BILINEAR)

    input_tensor = transform(input_img).unsqueeze(0).to(device, dtype=torch.float32)

    # Get predicted clean image from model
    pred_clean_img = evaluate(model, input_tensor)  # uint8 predicted clean
    # Recover light sources
    final_img = recover_light_sources(network_output=pred_clean_img,original_image=input_img)
    return final_img

demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"),outputs=gr.Image(), examples=["test_imgs/test1.png", "test_imgs/test2.png","test_imgs/test3.png"])
demo.launch()
