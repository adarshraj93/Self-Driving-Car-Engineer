# Self-Driving-Car-Engineer

This is the Github repository for the Self Driving Car Engineer Nanodegree from Udacity. This currently contains all the projects completed to successfully pass the nanodgree program. This wouldnt have been possible without the :insightful concepts taught by the instructors, inputs and inetraction with mentors, peers & community and inquisitiveness to learn groundbreaking concepts and algorithms  along with my motivation to overcome failure while implementating the code


## Nanodegree Concepts
This nanodegree gives great insights and the practical application of the following ideas and concepts in autonomous systems:
1. Computer Vision
2. Deep Learning & Convolutional Nueral Networks
3. Sensor Fusion
4. Localization & Mapping
5. Path Planning
6. Controls
7. Systems Integration

Each project focusses on each of the above topics and giving knowledge about teh different concepts and alogorithms used in the industry. It is very detailed and interesting.


## Softwares & Tools Used
The projects are majorly done on the following languages:
1. Python
2. C++
3. ROS

Basic Linux commands and working on Jupyter notebooks is useful. Basic knowledge of Make file is also a plus.


## Project Overview
### 1. Computer Vision
In this module two projects were implemented  to identify lanes lanes on difficult roads and to track vehicles

Project 1: [**Find Lane Lines**](https://github.com/adarshraj93/Self-Driving-Car-Engineer/tree/master/P1-Find-Lane-Lines)
A pipeline code was implemeted to identify lane lines, first on an image and then on a video. Python, OpenCV and other tools were used to implement and build on the concepts learnt in class.

Project 2: [**Advanced Lane Lines**](https://github.com/adarshraj93/Self-Driving-Car-Engineer/tree/master/P2-Advanced-Lane-Lines)
Extending on the first project, pipeline code was built to identify lane boundaries on a video from a front facing camera. The output identifies: the position of the lane lines, location of the vehicle wrt the centre of the lane and the radius of curvature of the road.


### 2. Deep Learning
In this module two more projects were carried out. This module focussed on the concepts of machine learning and its practical applications on autonomous vehicle development. The concepts to build and train deep neural networks were taught by NVIDIA experts.

Project 3: [**Traffic Sign Classifier**](https://github.com/adarshraj93/Self-Driving-Car-Engineer/tree/master/P3-Traffic-Sign-Classifier)
A Convolutional Neural Network was built on Tensorflow to identify and classify traffic sign images from MNST dataset. A Lenet-5 model was trained and validated to an accuracy of > 90% on the validation and test dataset

Project 4: [**Behavioral Cloning**](https://github.com/adarshraj93/Self-Driving-Car-Engineer/tree/master/P4-Behavioral-Cloning)
NVIDIA's neural network from End to End Learning for Self-Driving Cars' paper was built and trained using Tensorflow' Keras API to clone human steering behavior to autonomously drive the car on dacity's Self Driving Car Simulator. The dataset used to trained was generated from the same simulator. The images from 3 camera angles (left, centre & right positon on the car) along with the steering angle, throttle, brake and speed during each frame was used.


### 3. Sensor Fusion
Sensor fusion engineers from Mercedes-Benz taught the fundamental mathematical tools to implement a Kalman Filter. This filter is sued to predict and determine with certainity the location of other vehicles on the road.

Project 5: [**Extended Kalman Filter**](https://github.com/adarshraj93/Self-Driving-Car-Engineer/tree/master/P5-Extended-Kalman-Filter)
An Extended Kalman Fiter was implemented using a Constant Velocity (CV) model in C++. Data from RADAR and LIDAR was fused to track a bicycle's (travels around the car) position and velocity.


### 4. Localization
Localization is used to determine the vehicle position in the world. The principles of Markov Localisation was used to program a particle filter to determin the precise location the vehicle using data and a map

Project 6: [**Kidnapped Vehicle Project**](https://github.com/adarshraj93/Self-Driving-Car-Engineer/tree/master/P6-Kidnapped-Vehicle-Project)
A 2-D particle filter was implemented with a map to localize a vehicle in C++. The particle filter's map, obeservation data, control data and initial localization information was used by the filter to localize the vehicle within a desired accuracy with a specified run time of 100 seconds.


### 5. Planning
The Mercedes-Benz team taught a 3 stage planning model. Firstly it involved predicting the behavior of other vehicles on the road followed by the behavior of the interested or controlled car. Lastly a trajectory generation for the path the controlled vehicle needs to take while perfoming a requried maneuver like a lane change.

Project 7: [**Path Planning Project**](https://github.com/adarshraj93/Self-Driving-Car-Engineer/tree/master/P7-Path-Planning-Project)
A path planner was desired for a car driving on a 3 lane highway. The planner was able to drive the car around a crowded highway. The planner involved a model that predicted the behavior of other vehicles on the highway. The planner also contained a model which set the required speed for the controlled car depending on traffic ahead of the car, speed limit and deceleration jerk design constraints. It aslo defined when and which lane to change to overtake slow moving traffic and keeping up to the speed limit of the highway. Finally the planner also generated a trajectory for smooth lane changing. It succesfully drove around the highway of 4.32 miles without any collision


### 6. Controls
Uber ATG walked through building a proportional-integral-derivative (PID) controller to actuate the vehicle. A Proportional–Integral–Derivative (PID) Controller is one of the most common control loop feedback mechanisms. A PID controller continuously calculates an error function (which in our case is the distance from the center of the lane) and applies a correction based on these P, I, and D terms.

Project 8: [**PID Control Project**](https://github.com/adarshraj93/Self-Driving-Car-Engineer/tree/master/P8-PID-Control-Project)
A PID controller was developed to steer a self driving car around the track in the Simulator. The controlled was manaully tuned. An attempt to use the Twiddle (optimization) alogrithm  was done but due to lack of time it was overlooked.


### 7. System Integration
In the final module, ROS was introduced to integrate all the different required core functionalities to drive a car autonomously. This is the final capstone project for this nanodegree program.

Project 9: [**System Integration ROS**](https://github.com/adarshraj93/Self-Driving-Car-Engineer/tree/master/P9-%20System%20Integration%20ROS)
The 4 major subsystem for any autonomous car, SENSOR, PERCEPTION, PLANNING and CONTROL was integrated. ROS nodes were built to implement the core functionality of the autonomous vehicle system, including traffic light detection, throttle, drive-by-wire, vehicle control, and waypoint following. This software system will be deployed on Carla (Udacity’s Self Driving Lincoln MKZ) to autonomously drive it around a test track meeting all the requitements.

Traffic light classification was not a must for passing this project. This was not done due to lack of time and will implemented in the future



## Note
Please note this repository is still under developement. The extra curricular project sections will be imnplemented in the future and added to the repo. Projects will also be updated with newer concepts and ideas. If you have any inputs or feedback that you would like to share with me, please connect and reach out to me on [**LinkedIN**](https://www.linkedin.com/in/adarsh-rajks/) or on [**GitHub**](https://github.com/adarshraj93)
