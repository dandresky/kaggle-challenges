## Competition Description

This is a challenge to classify whether images contain either a dog or a cat. It uses the Asirra (Animal Species Image Recognition for Restricting Access) dataset which is typically used for CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart) or HIP (Human Interactive Proof) web protection challenges.

This challenge was closed in 2014 so I am not submitting my results. However, I need to do some experimentation on deep learning neural nets for my DSI capstone project and this challenge afforded me a baseline to learn from and compare results.

## Source Code and Data

- [src/cnn_model_v1.py](src/cnn_model_v1.py) is a shallow net with just three convolution and pooling layers trained on 2000 images (1000 cats and 1000 dogs) and obtains a validation accuracy of ~80%. It is based on a Keras blog with minor modifications.    
- [src/cnn_model_v2.py](src/cnn_model_v2.py) is the first of two steps to create a deep learning net capable of training on a small amount of data. Its purpose is to train the fully connected layer on top of a pre-trained VGG16 net. Weight are saved for use in the next script.
- [src/cnn_model_v3.py](src/cnn_model_v3.py) combines FC layer weights obtained in the previous script with a fully trained VGG16 net and fine tunes the last stage of the net.

## Results

- cnn_model_v1 obtains an accuracy of ~80% which was state of the art when the challenge was originally launched.
- cnn_model_v2 obtains an accuracy of ~90%
- cnn_model_v3 otains and accuracy of ?
