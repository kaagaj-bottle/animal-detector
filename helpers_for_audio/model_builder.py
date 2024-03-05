import torch
import torchvision
from torchvision.models import efficientnet_b0

# check the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')



def construct_model(output_shape:int):
    model=efficientnet_b0(weights="DEFAULT").to(device)

    
    model.classifier=torch.nn.Sequential(
        torch.nn.Dropout(p=0.2,inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=output_shape,
                        bias=True).to(device)
    )
    return model




