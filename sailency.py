import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from PIL import Image

NUM_CLASS = 10

class TinyVGG(nn.Module):
    def __init__(self, filters=10):
        super(TinyVGG, self).__init__()
        self.conv1_1 = nn.Conv2d(3, filters, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(filters * 16 * 16, NUM_CLASS)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.maxpool1(x)

        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.maxpool2(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x
    
# Modify the preprocess function with your training set mean and std
def preprocess(image, size=64, mean=[0.21689771115779877, 0.15157447755336761, 0.14177344739437103], std=[0.2911262810230255, 0.2139122635126114, 0.2060253769159317]):
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(image)

# Modify the deprocess function with your training set mean and std
def deprocess(image, mean=[0.21689771115779877, 0.15157447755336761, 0.14177344739437103], std=[0.2911262810230255, 0.2139122635126114, 0.2060253769159317]):
    inverse_mean = [-m / s for m, s in zip(mean, std)]
    inverse_std = [1 / s for s in std]

    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=inverse_mean, std=inverse_std),
        T.ToPILImage(),
    ])
    return transform(image)

def show_img(PIL_IMG):
    plt.imshow(np.asarray(PIL_IMG))
    

state_dict = torch.load("trained_vgg_best.pth")
model = TinyVGG()
model.load_state_dict(state_dict)
model.eval()

img = Image.open('./images/4/4_1_rotate_2.jpeg') 
X = preprocess(img)
X.requires_grad_()

scores = model(X)
score_max_index = scores.argmax()
score_max = scores[0,score_max_index]

score_max.backward()

saliency, _ = torch.max(X.grad.data.abs(),dim=1)
# code to plot the saliency map as a heatmap
plt.imshow(saliency[0], cmap=plt.cm.hot)
plt.axis('off')
plt.show()