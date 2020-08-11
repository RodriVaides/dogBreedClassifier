import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import torch
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models

from model import convClassifier

"""
NOTE: This file was built using the code from Udacity's Machine Learning Nanodegree as a base
(Therefore it might have the same structure as other Udacity projects (such as the comment "Load the stored model parameters")
The udacity template files can be found here:
https://github.com/udacity/sagemaker-deployment - referenced in July / August 2020
"""
def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.vgg16(pretrained=True)

    classifier =  convClassifier()


    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        classifier.load_state_dict(torch.load(f))

    model.classifier = classifier

    model.to(device).eval()

    print("Done loading model.")
    return model

def _get_train_data_loader(batch_size, data_dir):
    print("Get train data loader.")

    train_dir = os.path.join(data_dir,'train')
    valid_dir = os.path.join(data_dir,'valid')

    # Defining transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    #Using datasets.ImageFolder to load the data and apply the transforms to each dataset
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    #Creating dataloaders for each dataset
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    loaders_scratch={
        'train':trainloader,
        'valid':validloader
    }
    return loaders_scratch

    # ------------------

def train(n_epochs, loaders, model, optimizer, criterion, device, model_dir):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    print_every = 100
    steps = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=True)

    for epoch in range(1, n_epochs+1):
        print('device: {}'.format(device))
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        ###################
        # train the model #
        ###################
        model.train()

        for batch_idx, (data, target) in enumerate(loaders['train']):
            try:
                # move to GPU
                data = data.to(device)
                target = target.to(device)
                model.to(device)

                ## find the loss and update the model parameters accordingly
                optimizer.zero_grad()

                output = model.forward(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                ## record the average training loss, using something like
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
                if steps % print_every == 0:
                    print("Epoch: {}/{}... ".format(epoch, n_epochs),
                          "Loss: {:.4f}".format(train_loss/print_every))
            except OSError as err:
                print("there was an error in batch: {}".format(batch_idx))
                print(err)

        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            data = data.to(device)
            target = target.to(device)
            model.to(device)

            ## update the average validation loss
            output_valid = model.forward(data)
            v_loss = criterion(output_valid, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (v_loss.data - valid_loss))

        scheduler.step(train_loss)
        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            print("Validation loss improved in epoch: {}, Previous valid loss: {}, new valid loss:{}".format(epoch, valid_loss_min, valid_loss))
            valid_loss_min = valid_loss
    save_model(model.classifier, model_dir)

def save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)

if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    # torch.manual_seed(args.seed)

    # Load the training data.
    loaders = _get_train_data_loader(args.batch_size, args.data_dir)

    # Build the model.
    model = models.vgg16(pretrained=True)

    for parameter in model.parameters():
        parameter.requires_grad = False

    classifier = convClassifier()

    model.classifier = classifier
    model = model.to(device)

    print("Model loaded")

    # Train the model.
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.05)
    loss_fn = torch.nn.CrossEntropyLoss()

#    train(model, train_loader, args.epochs, optimizer, loss_fn, device)
    train(args.epochs, loaders, model, optimizer, loss_fn, device, args.model_dir)
