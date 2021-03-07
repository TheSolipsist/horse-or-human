from torch import nn, load, optim
from torchvision.transforms import ToPILImage
from pathlib import Path
from read_data import new_size

data_dir = 'data'
data_filename = 'data.pt'
data = load(Path(__file__).parent.absolute() / data_dir / data_filename)

first_image = data['train'][782][0:-1]
y = data['train'][782][-1]

ToPILImage()(first_image.reshape(3, new_size[0], new_size[1])).show()
print(y)