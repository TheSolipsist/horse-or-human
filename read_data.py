from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

train_data = ImageFolder(root='data/train', transform=ToTensor())
test_data = ImageFolder(root='data/validation', transform=ToTensor())

print(train_data.size())