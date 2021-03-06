import torch.nn as nn
import torch.nn.functional as F

"""
--------------References------------------
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
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3,32,kernel_size=5)
        self.conv2 = nn.Conv2d(32,64,kernel_size=5)
        self.conv3 = nn.Conv2d(64,128,kernel_size=5)
        self.mp = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(73728,1000)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1000,133)

    def forward(self, x):
        ## Define forward behavior
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = F.relu(self.mp(self.conv3(x)))

        x = x.view(in_size,-1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_device(self):
        device = self.conv1.weight.device
        return device
