from torch import nn, load, optim
from torchvision.transforms import ToPILImage
from pathlib import Path

data_dir = 'data'
data_filename = 'data.pt'
data = load(Path(__file__).parent.absolute() / data_dir / data_filename)

first_image = data['train'][782][0:-1]
y = data['train'][782][-1]

ToPILImage()(first_image.reshape(3,50,50)).show()
print(y)