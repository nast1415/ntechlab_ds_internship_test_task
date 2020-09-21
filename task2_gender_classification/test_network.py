import torch
import json
import sys

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional layer with 3 input chanels and 6 filters each of the size 5x5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Convolutional layer with 6 input chanels and 16 filters each of the size 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Pooling layer (max pool) with the size of 2x2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully-connected layer with the number of input neurons: # chanels * height * width
        # Height and width were calculated for the input size of 64x64
        self.fc1 = nn.Linear(16 * 13 * 13, 120) 
        # Fully-connected layer with the number of input neurons 120 and number of output neurons 84
        self.fc2 = nn.Linear(120, 84)
        # Fully-connected layer with the number of input neurons 84 and number of output neurons 84
        self.fc3 = nn.Linear(84, 84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Apply ReLU non-linearity and pooling to the CONV1 layer
        x = self.pool(F.relu(self.conv2(x))) # Apply ReLU non-linearity and pooling to the CONV2 layer
        
        x = x.view(-1, 16 * 13 * 13) # Reshape x to the vector (for FC layer)
        x = F.relu(self.fc1(x)) # Apply ReLU non-linearity to the FC1 layer
        x = F.relu(self.fc2(x)) # Apply ReLU non-linearity to the FC2 layer
        x = self.fc3(x) # Apply FC3 layer
        return x

# Get test directory
test_dir = sys.argv[1]

# Create custom transformation as a composition of three transformations. 
# It will be applied to all images in train and test datasets
img_processing_transform = transforms.Compose(
    [transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

'''
Custom class which extends torchvision.datasets.ImageFolder to get not only images and labels,
but also images' paths
'''
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    # override the original __getitem__ method
    def __getitem__(self, index):
        # standart return of the ImageFolder
        standart_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        
        # the image file path
        img_path = self.imgs[index][0]
        
        # make a new tuple that includes original and the path
        extended_tuple = (standart_tuple + (img_path,))
        return extended_tuple

# Create dataset with all train data using custom ImageFolderWithPaths class
dataset = ImageFolderWithPaths(root=test_dir, transform=img_processing_transform)
# Upload test data
test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, drop_last=False, num_workers=4)

# Define classes names
classes = ('female', 'male')

# Create dictionary for results of classification
prediction_results = {}

# Upload trained model
nn_model_path = './gender_classification_net.pth'
net = Net()
net.load_state_dict(torch.load(nn_model_path))

# Apply model on the test data and save results to the prediction_results dict
with torch.no_grad():
    for data in test_data_loader:
        images, labels, paths = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        for el in range(len(predicted)):
            prediction_results.update({paths[el].split('/')[-1] : classes[predicted[el]]})

# Save results of prediction to the json file
with open('process_results.json', 'w') as outfile:
    json.dump(prediction_results, outfile)

print("Processing results are now stored in 'process_results.json'")