import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class convClassifier(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3,32,kernel_size=5)
        self.conv2 = nn.Conv2d(32,64,kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(179776,len(dog_category_dict))

    def forward(self, x):
        ## Define forward behavior
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size,-1)
        x = self.fc(x)
        return F.log_softmax(x)

#-#-# You so NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()
