import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

"""
NOTE: This file was built using the code from Udacity's Machine Learning Nanodegree as a base
(Therefore it might have the same structure as other Udacity projects (such as the comment "Load the stored model parameters")
The udacity template files can be found here:
https://github.com/udacity/sagemaker-deployment - referenced in July / August 2020
"""

"""
The reference used for building the image classifier was the Pytorch documentation available here:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

and here:
https://pytorch.org/docs/stable/torchvision/
"""
# define the CNN architecture
class convClassifier(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(convClassifier, self).__init__()
        # Define layers of a CNN
        self.fc1 = nn.Linear(25088,5000)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(5000,133)


    def forward(self, x):
        ## Define forward behavior
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_device(self):
        device = self.fc1.weight.device
        return device
