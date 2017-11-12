#**Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/center.jpg "Grayscaling"
[image3]: ./examples/left.jpg "Recovery Image"
[image4]: ./examples/right.jpg "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"


---
###Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* clone.py and clone_helper.py for setting up training pipline
* model5n.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
sh python drive.py model5n.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model I used is modeled after Nvidia's self driving paper.

My model consists of a 4 layer convolution neural network with 5x5 and 3x3 kernal sizes  (model.py lines 15-21). 

It also includes three pooling layer to reduce the number of training parameter at lines 12, 17 and 19. 

The fully connected layer consist of 4 layer with 1164 to 10 neuron, same as the one published in the Nvidia paper.

Here is a deital description of the model 
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0
_________________________________________________________________
average_pooling2d_1 (Average (None, 45, 160, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 41, 156, 32)       2432
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 20, 78, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 74, 32)        25632
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 70, 32)        25632
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 6, 35, 32)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 33, 32)         9248
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 2, 31, 32)         9248
_________________________________________________________________
flatten_1 (Flatten)          (None, 1984)              0
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              2310540
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11
=================================================================
Total params: 2,504,803
Trainable params: 2,504,803

####2. Attempts to reduce overfitting in the model

To reduce the overfitting, I stop the training when the error rate start to increase. Which is at 2 epochs.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 34).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to leverage existing model and retrain it for this project.

My first step was to use a convolution neural network model similar to the Lanet I thought this model might be appropriate because of it's simplicity and its relative easy to train. After getting the training pipeline setup, I preceed to swtich to Nivida's model, as it was a proven model for self driving car.  

My biggest challenge was to find a model that fit's on my GTX 1050 graphics card. The Nvidia's model was too big to fit on my graphics card. So I added three pooling layer to reduce the number of parameter. 

The final step was to run the simulator to see how well the car was driving around track one. At first, the vehicle fell off the track before getting on and off the bridge section. However, I was able to resolve this by using the left and right camera image and increasing the correction steer on line 6 of `clone_helper.py`

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 15-27) consisted of a convolution neural network with a series of convolution and pooling layers modeled after the Nvidia's model.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. These images show what a recovery looks like from left and right camera:

![alt text][image3]
![alt text][image4]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by increasing loss and the second epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
