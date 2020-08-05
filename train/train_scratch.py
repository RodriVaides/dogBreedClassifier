import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
#import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models

from model_scratch import convClassifier

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    #model_info = {}
    #model_info_path = os.path.join(model_dir, 'model_info.pth')
    #with open(model_info_path, 'rb') as f:
    #    model_info = torch.load(f)

    #print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = convClassifier()


    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

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
    # loss_list=[]

    for epoch in range(1, n_epochs+1):
        # print('device: {}'.format(device))
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        # print(model.get_device())
        # print('device: {}'.format(device))
        for batch_idx, (data, target) in enumerate(loaders['train']):
            try:
                # move to GPU
                data = data.to(device)
                target = target.to(device)
                model.to(device)
                # print("Device used for training: {}".format(model.get_device()))
#                if use_cuda:
#                    data, target = data.cuda(), target.cuda()
                ## find the loss and update the model parameters accordingly
                #data.resize_(data.size()[0], 3*224*224)
                optimizer.zero_grad()

                output = model.forward(data)
                loss = criterion(output, target)
                # loss_list.append(loss)

                loss.backward()
                optimizer.step()

                ## record the average training loss, using something like
                #train_loss += loss.item()
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
                # if steps % print_every == 0:
                #     print("Epoch: {}/{}... ".format(epoch, n_epochs),
                #           "Loss: {:.4f}".format(train_loss/print_every))
            except OSError as err:
                print("there was an error in batch: {}".format(batch_idx))
                print(err)
        ######################
        # validate the model #
        ######################
        # model.eval()
        # for batch_idx, (data, target) in enumerate(loaders['valid']):
        #     # move to GPU
        #     if use_cuda:
        #         data, target = data.cuda(), target.cuda()
        #     ## update the average validation loss
        #

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
        print('Epoch: {} \t Training Loss: {:.6f} \t Validation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        ## TODO: save the model if validation loss has decreased

        if valid_loss < valid_loss_min:
            print("Validation loss improved in epoch: {}, Previous valid loss: {}, new valid loss:{}".format(epoch, valid_loss_min, valid_loss))
            valid_loss_min = valid_loss
        save_model(model, model_dir)
    # return trained model
#    return model

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
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')

    # Model Parameters
    # parser.add_argument('--embedding_dim', type=int, default=32, metavar='N',
    #                     help='size of the word embeddings (default: 32)')
    # parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
    #                     help='size of the hidden dimension (default: 100)')
    # parser.add_argument('--vocab_size', type=int, default=5000, metavar='N',
    #                     help='size of the vocabulary (default: 5000)')

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
    model = convClassifier()

    model = model.to(device)

    print("Model loaded")

    # Train the model.
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    loss_fn = torch.nn.CrossEntropyLoss()

#    train(model, train_loader, args.epochs, optimizer, loss_fn, device)
    train(args.epochs, loaders , model, optimizer, loss_fn, device, args.model_dir)

    # Save the parameters used to construct the model
    # model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    # with open(model_info_path, 'wb') as f:
    #     model_info = {
    #         'embedding_dim': args.embedding_dim,
    #         'hidden_dim': args.hidden_dim,
    #         'vocab_size': args.vocab_size,
    #     }
    #     torch.save(model_info, f)

	# Save the word_dict
    # word_dict_path = os.path.join(args.model_dir, 'word_dict.pkl')
    # with open(word_dict_path, 'wb') as f:
    #     pickle.dump(model.word_dict, f)

	# Save the model parameters
#    model_path = os.path.join(args.model_dir, 'model.pth')
#    with open(model_path, 'wb') as f:
#        torch.save(model.cpu().state_dict(), f)
