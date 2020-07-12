import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


# define the CNN architecture
class convClassifier(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(convClassifier, self).__init__()
        # Define layers of a CNN
        # TODO: Specify model architecture
        # self.classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(25088,5000)),
        #                                         ('relu',nn.ReLU()),
        #                                         ('drop',nn.Dropout(p=0.5)),
        #                                         ('fc2',nn.Linear(5000,133)),
        #                                         ('output',nn.LogSoftmax(dim=1))
        #                                         ]))
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
