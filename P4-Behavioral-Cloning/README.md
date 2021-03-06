# **Behavioral Cloning** 
### Adarsh Raj
---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_class.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* modelfinal.h5 containing a trained convolution neural network 
* README.md file summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 2x2 filter sizes and depths between 24 and 64 (model_class.py lines 88-92) 

The model includes RELU layers to introduce nonlinearity (code line 88-92), and the data is normalized in the model using a Keras lambda layer (code line 85). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model_class.py lines 97-101). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 75-66). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model_class.py line 105). It was optimised to the mean square error of the steering angle

#### 4. Appropriate training data

Training data provided by Udacity was chosen to keep the vehicle driving on the road. It contains 9 laps of track along with recovery data. Hence i thought this is good.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
The model used by the NVIDIA team for steering control of an autonomous vehicle was used. This model had 5 convolutional layers and proved success in steering control. I augmented the training data. The centre, left and right images was also used. Steering correction was also added. The validation data was split from the trainig data in the ratio 20% to 80%. Finally the simulator was used to see how well the car drove around the track. The vehicle did fall off the track with increased speed.

At the end, the vehicle is able to drive autonomously around the track without leaving the road.

![](Images/NVIDIA.JPG)

#### 2. Final Model Architecture

The final model architecture (model_class.py lines 84-102) consisted of a convolution neural network with the following layers and layer sizes ...

![](Images/finalmodel.JPG)


#### 3. Creation of the Training Set & Training Process

I used the data provided by Udacity to train my model. Here is an example image of center lane driving:

![](Images/center_2016_12_01_13_43_53_287.jpg)


I also used the left and right camera images. The left camera image is as below:

![](Images/left_2016_12_01_13_43_53_287.jpg)


The right camera image is as below:

![](Images/right_2016_12_01_13_43_53_287.jpg)

I augmented the data set to flip images and angles to generalise the model (model_class.py lines 67 -72)


After the collection process, I had 48216 number of data points. I then preprocessed this data by normalising and mean centering the image. I also cropped out all the irrelevant information from the image to reduce computional requirements. The final image was 65x320x3


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by increasing loss after 2. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![](Images/Loss.JPG)



**Note: Please refer to the video in the repo for final results. Please downlaod the data provided by Udacity to run the model_class.py**
