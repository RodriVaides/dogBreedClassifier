# Dog Breed Classifier
Dog breed classifier is a project developed as part of Udacity's Machine Learning Engineer Nanodegree, where a machine learning model was built to classify pictures of dogs into their corresponding breeds. The project is built with Pytorch, and it uses Convolutional neural networks to build a model from scratch as well as a pre-trained model to classify images. In this project a static website is also included as an interface for users to classify images.

## Link to github project (if viewing README outside of github)
https://github.com/RodriVaides/dogBreedClassifier

# Repository contents
The main file where most of the content and notes for the project are contained is the Jupyter notebook "dog_app.ipynb".

The "train" and "serve" folders contain the files that were used to train and deploy the mode using Amazon SageMaker.

The folder "haarcascades" includes the xml file required by the OpenCV face detector which is used at the beginning of the Jupyter notebook.

The folder "website files" contains the .html & javascript files that were used to build the static website for the users.

# Dependencies

* boto3==1.14.12
* botocore==1.17.12
* numpy==1.18.5
* opencv-python==4.2.0.34
* Pillow==7.1.2
* python-dotenv==0.13.0
* requests==2.24.0
* s3transfer==0.3.3
* sagemaker==1.66.0
* sagemaker-containers==2.8.6.post2
* scikit-learn==0.23.1
* scipy==1.5.0
* torch==1.5.1
* torchvision==0.6.1
* tqdm==4.46.1

for a full list (pip freeze) of dependencices see [requirements.txt](requirements.txt)

# Instructions to use web app
To use the web application you can use the following link:
http://www.dog-classifier-demo.com.s3-website.eu-central-1.amazonaws.com
which will take you to a website deployed in amazon.

You can also use the file [index.html](website_files/index.html) to view the website.

# Steps for obtaining the dataset

There are 2 datasets which will be used for this project. Both datasets have been provided by Udacity as part of the Machine Learning nanodegree and can be downloaded from the following links:

* <strong>Dog image files:</strong><br>
https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip

* <strong>Human image files:</strong><br>
https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip

# License
For information about the license please view the [LICENSE](LICENSE) file
