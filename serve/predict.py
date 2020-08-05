import argparse
# import json
import os
# import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
#
# from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models
#
from PIL import Image
#
from model import convClassifier
import requests
from io import BytesIO

# from torchvision import datasets

dog_category_dict = {121: 'Pointer',
 68: 'French_bulldog',
 123: 'Poodle',
 111: 'Nova_scotia_duck_tolling_retriever',
 42: 'Canaan_dog',
 23: 'Bichon_frise',
 35: 'Briard',
 85: 'Irish_setter',
 77: 'Great_dane',
 60: 'English_cocker_spaniel',
 74: 'Glen_of_imaal_terrier',
 88: 'Irish_wolfhound',
 53: 'Collie',
 108: 'Norwegian_elkhound',
 76: 'Gordon_setter',
 34: 'Boykin_spaniel',
 6: 'American_foxhound',
 83: 'Icelandic_sheepdog',
 57: 'Dandie_dinmont_terrier',
 75: 'Golden_retriever',
 124: 'Portuguese_water_dog',
 69: 'German_pinscher',
 7: 'American_staffordshire_terrier',
 122: 'Pomeranian',
 25: 'Black_russian_terrier',
 81: 'Havanese',
 86: 'Irish_terrier',
 1: 'Afghan_hound',
 26: 'Bloodhound',
 132: 'Yorkshire_terrier',
 48: 'Chinese_crested',
 73: 'Giant_schnauzer',
 131: 'Xoloitzcuintli',
 100: 'Maltese',
 18: 'Bedlington_terrier',
 33: 'Boxer',
 2: 'Airedale_terrier',
 66: 'Finnish_spitz',
 118: 'Petit_basset_griffon_vendeen',
 27: 'Bluetick_coonhound',
 64: 'Entlebucher_mountain_dog',
 103: 'Miniature_schnauzer',
 41: 'Cairn_terrier',
 87: 'Irish_water_spaniel',
 117: 'Pembroke_welsh_corgi',
 22: 'Bernese_mountain_dog',
 126: 'Silky_terrier',
 9: 'Anatolian_shepherd_dog',
 0: 'Affenpinscher',
 90: 'Japanese_chin',
 82: 'Ibizan_hound',
 16: 'Bearded_collie',
 36: 'Brittany',
 12: 'Australian_terrier',
 44: 'Cardigan_welsh_corgi',
 70: 'German_shepherd_dog',
 28: 'Border_collie',
 17: 'Beauceron',
 106: 'Norfolk_terrier',
 93: 'Komondor',
 10: 'Australian_cattle_dog',
 98: 'Lhasa_apso',
 92: 'Kerry_blue_terrier',
 39: 'Bulldog',
 125: 'Saint_bernard',
 61: 'English_setter',
 67: 'Flat-coated_retriever',
 80: 'Greyhound',
 40: 'Bullmastiff',
 52: 'Cocker_spaniel',
 110: 'Norwich_terrier',
 31: 'Boston_terrier',
 120: 'Plott',
 58: 'Doberman_pinscher',
 59: 'Dogue_de_bordeaux',
 114: 'Papillon',
 128: 'Tibetan_mastiff',
 45: 'Cavalier_king_charles_spaniel',
 56: 'Dalmatian',
 109: 'Norwegian_lundehund',
 97: 'Leonberger',
 84: 'Irish_red_and_white_setter',
 130: 'Wirehaired_pointing_griffon',
 104: 'Neapolitan_mastiff',
 102: 'Mastiff',
 112: 'Old_english_sheepdog',
 50: 'Chow_chow',
 101: 'Manchester_terrier',
 38: 'Bull_terrier',
 19: 'Belgian_malinois',
 116: 'Pekingese',
 63: 'English_toy_spaniel',
 95: 'Labrador_retriever',
 24: 'Black_and_tan_coonhound',
 78: 'Great_pyrenees',
 71: 'German_shorthaired_pointer',
 8: 'American_water_spaniel',
 3: 'Akita',
 30: 'Borzoi',
 29: 'Border_terrier',
 21: 'Belgian_tervuren',
 13: 'Basenji',
 32: 'Bouvier_des_flandres',
 107: 'Norwegian_buhund',
 129: 'Welsh_springer_spaniel',
 15: 'Beagle',
 127: 'Smooth_fox_terrier',
 105: 'Newfoundland',
 46: 'Chesapeake_bay_retriever',
 14: 'Basset_hound',
 54: 'Curly-coated_retriever',
 4: 'Alaskan_malamute',
 72: 'German_wirehaired_pointer',
 49: 'Chinese_shar-pei',
 96: 'Lakeland_terrier',
 55: 'Dachshund',
 113: 'Otterhound',
 65: 'Field_spaniel',
 47: 'Chihuahua',
 37: 'Brussels_griffon',
 5: 'American_eskimo_dog',
 79: 'Greater_swiss_mountain_dog',
 119: 'Pharaoh_hound',
 62: 'English_springer_spaniel',
 43: 'Cane_corso',
 20: 'Belgian_sheepdog',
 115: 'Parson_russell_terrier',
 11: 'Australian_shepherd',
 91: 'Keeshond',
 99: 'Lowchen',
 51: 'Clumber_spaniel',
 94: 'Kuvasz',
 89: 'Italian_greyhound'}

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

def input_fn(input_url, content_type):


    img_url = input_url.decode("utf-8")
    input_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    # #
    response = requests.get(img_url)
    input_img = Image.open(BytesIO(response.content))
    input_data = input_transforms(input_img).unsqueeze(0)

    # meta data dict used for debugging and analysing the input data
    input_meta_data = {}
    input_meta_data['input_url_type'] = str(type(input_url))
    input_meta_data['input_url'] = str(input_url)
    input_meta_data['input_url_decoded'] = str(img_url)
    input_meta_data['input_url_decoded_type'] = str(type(img_url))
    input_meta_data['transformed_value'] = str(input_data)
    input_meta_data['transformed_value_shape'] = str(input_data.shape)

    print(str(input_meta_data))

    return (img_url,input_data)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return prediction_output


def predict_fn(input_tuple, model):
    print('Inferring sentiment of input data.')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = input_tuple[1].to(device)

    # Make sure to put the model into evaluation mode
    model.eval()

    # TODO: Compute the result of applying the model to the input data. The variable `result` should
    #       be a numpy array which contains a single integer which is either 1 or 0

    with torch.no_grad(): output = model.forward(data)
    result = int(output.argmax())
    # result = np.int(np.round(output.numpy()))
    result_dict = {}
    url_folder_name = str(result+1).zfill(3)+"."+dog_category_dict[result]

    # Creating response dictionary (Json)
    result_dict['predicted_value'] = result
    result_dict['predicted_name'] = dog_category_dict[result]
    result_dict['predicted_result_url'] = "https://my-sage-maker-instance-test-20-03-2020-2.s3.eu-central-1.amazonaws.com/img_inputs/display/{}/file1.jpg".format(url_folder_name)
    result_dict['input_img_url'] = input_tuple[0]

    return str(result_dict)
