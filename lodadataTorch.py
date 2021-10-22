import  matplotlib.pyplot as plt
import torch.utils.data
from torchvision import datasets,transforms
from PIL import Image

transform=transforms.Compose([transforms.Resize((28,28)),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])
dataset=datasets.ImageFolder("./data/root",transform=transform)
dl=torch.utils.data.DataLoader(dataset,batch_size=1)
i,l=next(iter(dl))
print(l[0])