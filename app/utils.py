import torch
from PIL import Image
import numpy as np

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('model/model_epoch_49.pth',map_location=device)

def evaluate(model,image):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        #outputs= model(image.unsqueeze(0))
        outputs= model(image)
        return outputs.squeeze(0).squeeze(0).cpu()
    
def predict(input_image):
    #input_image=Image.open(inp_img).convert('RGB')
    input_image=input_image.resize((512,512))
    input_image_torch=torch.tensor(np.array(input_image)).permute(2,0,1).unsqueeze(0).float()/255.0
    mask=evaluate(model,input_image_torch)
    mask=mask.permute(1,2,0).numpy()
    return mask

def calculate_input_illuminance(image):
    """
    Calculate illuminance: I_input = C_r + C_g + C_b
    """
    return np.sum(image, axis=2)

def generate_recovery_weight_matrix(illuminance_matrix, alpha=15):
    """
    Generate recovery weights using power function
    Formula: W_r = ((I_input - min) / (max - min))^α
    """
    I_min = np.min(illuminance_matrix)
    I_max = np.max(illuminance_matrix)
    
    if I_max == I_min:
        normalized = np.zeros_like(illuminance_matrix)
    else:
        normalized = (illuminance_matrix - I_min) / (I_max - I_min)
    
    # Apply power function with α = 15
    W_r = np.power(normalized, alpha)
    return W_r

def recover_light_sources(original_image, network_output, alpha=15):
    """
    Final recovery: I_final = (1 - W_r) ⊙ N(C) + W_r ⊙ C
    """
    # Calculate illuminance and recovery weights
    I_input = calculate_input_illuminance(original_image)
    W_r = generate_recovery_weight_matrix(I_input, alpha)
    
    # Expand to match image dimensions
    W_r_expanded = np.expand_dims(W_r, axis=2)
    W_r_expanded = np.repeat(W_r_expanded, 3, axis=2)
    
    # Convex combination for light source recovery
    I_final = (1 - W_r_expanded) * network_output + W_r_expanded * original_image
    return np.clip(I_final, 0, 255).astype(np.uint8)