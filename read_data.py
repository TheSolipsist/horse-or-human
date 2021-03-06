import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image

def read_data(data_dir, data_filename):
    labels = ['horses', 'humans']
    label_value = {'horses': 0., 'humans': 1.}
    data_tags = ['train', 'validation']
    data = {data_tag: [] for data_tag in data_tags}
    for data_tag in data_tags:
        for label in labels:
            path_to_directory = Path(__file__).parent.absolute() / data_dir / data_tag / label
            for image_path in path_to_directory.glob('*'):    
                img = Image.open(image_path).convert("RGB")
                img_tensor = transforms.ToTensor()(img).flatten()
                img_tensor = torch.cat((img_tensor, torch.tensor([label_value[label]])))
                data[data_tag].append(img_tensor)
        data[data_tag] = torch.stack(data[data_tag]) # Turn list to tensor
    torch.save(data, Path(__file__).parent.absolute() / data_dir / data_filename)

data_dir = 'data'
data_filename = 'data.pt'
read_data(data_dir, data_filename)