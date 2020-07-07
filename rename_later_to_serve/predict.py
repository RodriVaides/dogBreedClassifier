import argparse
import json
import os
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from model import convClassifier

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    # model_info = {}
    # model_info_path = os.path.join(model_dir, 'model_info.pth')
    # with open(model_info_path, 'rb') as f:
        # model_info = torch.load(f)

    # print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = convClassifier()

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print("Done loading model.")
    return model

def input_fn(input_data, content_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = input_data.to(device)
    return data

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return prediction_output

def predict_fn(input_data, model):
    print('Inferring sentiment of input data.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Process input_data so that it is ready to be sent to our model.
    #       You should produce two variables:
    #         data_X   - A sequence of length 500 which represents the converted review
    #         data_len - The length of the review

    test_words = review_to_words(input_data)
    data_X, data_len = convert_and_pad(model.word_dict, test_words)

        #data_X = None
    #data_len = None

    # Using data_X and data_len we construct an appropriate input tensor. Remember
    # that our model expects input data of the form 'len, review[500]'.
    data_pack = np.hstack((data_len, data_X))
    data_pack = data_pack.reshape(1, -1)

    data = torch.from_numpy(data_pack)
    data = data.to(device)

    # Make sure to put the model into evaluation mode
    model.eval()

    # TODO: Compute the result of applying the model to the input data. The variable `result` should
    #       be a numpy array which contains a single integer which is either 1 or 0

    with torch.no_grad(): output = model.forward(data)
    result = np.int(np.round(output.numpy()))

    return result
