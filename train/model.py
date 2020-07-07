import torch
import torch.nn as nn
from collections import OrderedDict


# define the CNN architecture
class convClassifier(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(convClassifier, self).__init__()
        ## Define layers of a CNN
        ## TODO: Specify model architecture
        classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(25088,5000)),
                                                ('relu',nn.ReLU()),
                                                ('drop',nn.Dropout(p=0.5)),
                                                ('fc2',nn.Linear(5000,133)),
                                                ('output',nn.LogSoftmax(dim=1))
                                                ]))

    # def forward(self, x):
    #     ## Define forward behavior
    #     in_size = x.size(0)
    #     x = F.relu(self.mp(self.conv1(x)))
    #     x = F.relu(self.mp(self.conv2(x)))
    #     x = x.view(in_size,-1)
    #     x = self.fc(x)
    #     return F.log_softmax(x, dim=1)

    def get_device(self):
        device = self.conv1.weight.device
        return device
