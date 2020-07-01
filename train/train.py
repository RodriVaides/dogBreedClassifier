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

from model import convClassifier

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

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
    # valid_dir = os.path.join(data_dir,'valid')
    # test_dir = os.path.join(data_dir,'test')

    # Defining transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # test_validation_transforms = transforms.Compose([transforms.Resize(256),
    #                                       transforms.CenterCrop(224),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize([0.485, 0.456, 0.406],
    #                                                            [0.229, 0.224, 0.225])])

    #Using datasets.ImageFolder to load the data and apply the transforms to each dataset
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    # valid_data = datasets.ImageFolder(valid_dir, transform=test_validation_transforms)
    # test_data = datasets.ImageFolder(test_dir, transform=test_validation_transforms)

    #Creating dataloaders for each dataset
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    # testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    # validloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    # loaders_scratch={
    #     'train':trainloader,
    #     'test':testloader,
    #     'valid':validloader
    # }
    return trainloader

    # ------------------

def train(n_epochs, loaders, model, optimizer, criterion, device):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    print_every = 100
    steps = 0

    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders):
            try:
                # move to GPU
                data = data.to(device)
                target = target.to(device)
#                if use_cuda:
#                    data, target = data.cuda(), target.cuda()
                ## find the loss and update the model parameters accordingly
                #data.resize_(data.size()[0], 3*224*224)
                optimizer.zero_grad()

                output = model.forward(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                ## record the average training loss, using something like
                #train_loss += loss.item()
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
        # model.eval()
        # for batch_idx, (data, target) in enumerate(loaders['valid']):
        #     # move to GPU
        #     if use_cuda:
        #         data, target = data.cuda(), target.cuda()
        #     ## update the average validation loss
        #

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        ## TODO: save the model if validation loss has decreased
    # return trained model
    return model


# def train(model, train_loader, epochs, optimizer, loss_fn, device):
#     """
#     This is the training method that is called by the PyTorch training script. The parameters
#     passed are as follows:
#     model        - The PyTorch model that we wish to train.
#     train_loader - The PyTorch DataLoader that should be used during training.
#     epochs       - The total number of epochs to train for.
#     optimizer    - The optimizer to use during training.
#     loss_fn      - The loss function used for training.
#     device       - Where the model and data should be loaded (gpu or cpu).
#     """
#
#     # TODO: Paste the train() method developed in the notebook here.
#     for epoch in range(1, epochs + 1):
#         print("Epoch {}/{}".format(epoch,epochs))
#         model.train()
#         total_loss = 0
#         for batch in train_loader:
#             batch_X, batch_y = batch
#
#             batch_X = batch_X.to(device)
#             batch_y = batch_y.to(device)
#
#             # TODO: Complete this train method to train the model provided.
#             model.zero_grad()
#             output = model(batch_X)
#             loss = loss_fn(output, batch_y)
#             loss.backward()
#
#             optimizer.step()
#
#             total_loss += loss.data.item()
#         print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
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
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    # Build the model.
    model = convClassifier().to(device)
    # model = convClassifier(args.embedding_dim, args.hidden_dim, args.vocab_size).to(device)

    # with open(os.path.join(args.data_dir, "word_dict.pkl"), "rb") as f:
    #     model.word_dict = pickle.load(f)

    print("Model loaded")
    # print("Model loaded with embedding_dim {}, hidden_dim {}, vocab_size {}.".format(
    #     args.embedding_dim, args.hidden_dim, args.vocab_size
    # ))

    # Train the model.
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

#    train(model, train_loader, args.epochs, optimizer, loss_fn, device)
    train(args.epochs, train_loader, model, optimizer, loss_fn, device)

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
    # model_path = os.path.join(args.model_dir, 'model.pth')
    # with open(model_path, 'wb') as f:
    #     torch.save(model.cpu().state_dict(), f)
