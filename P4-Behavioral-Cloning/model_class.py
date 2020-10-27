import csv
import cv2
import numpy as np

# Read all the lines from csv file to get the image name. Source code from class
lines = []
with open('C:/Users/Adarsh/Desktop/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
del lines[0] #Delete the first header file

# print(lines[0])
# print(line[0])
# source_path = line[0]
# print(source_path)
# filename = source_path.split('/')[-1]
# print(filename)
# current_path = 'C:/Users/Adarsh/Desktop/data/IMG/' + filename
# print(current_path)
#
# image = cv2.imread(current_path)
# print(image)
# cv2.imshow('img', image)
# cv2.waitKey(0)

# Store all the training images and steering angle measurements. Source code from class
# Modified the code to read centre, left and right images. Also offset the steering values for left and right images
images = []
measurements = []
# for line in lines:
#     source_path = line[0]
#     filename = source_path.split('/')[-1]
#     current_path = 'C:/Users/Adarsh/Desktop/data/IMG/' + filename
#     image = cv2.imread(current_path)
#     images.append(image)
#     measurement = float(line[3])
#     measurements.append(measurement)

for line in lines:
    source_path_centre = line[0]
    source_path_left = line[1]
    source_path_right = line[2]
    filename_centre = source_path_centre.split('/')[-1]
    filename_left = source_path_left.split('/')[-1]
    filename_right = source_path_right.split('/')[-1]
    centre_path = 'C:/Users/Adarsh/Desktop/data/IMG/' + filename_centre
    left_path = 'C:/Users/Adarsh/Desktop/data/IMG/' + filename_left
    right_path = 'C:/Users/Adarsh/Desktop/data/IMG/' + filename_right
    #image = cv2.imread(centre_path)
    image = cv2.cvtColor(cv2.imread(centre_path), cv2.COLOR_BGR2RGB) #since CV2 reads an image in BGR we need to convert it to RGB since in drive.py it is RGB
    images.append(image)
    #image = cv2.imread(left_path)
    image = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2RGB)
    images.append(image)
    #image = cv2.imread(right_path)
    image = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2RGB)
    images.append(image)
    centre_measurement = float(line[3])
    measurements.append(centre_measurement)
    left_measurement = centre_measurement + 0.2
    measurements.append(left_measurement)
    right_measurement = centre_measurement - 0.2
    measurements.append(right_measurement)

# Augment all images. Source code from class
augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

# Define the training data
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# Build the training model. The Nividia model is used for training
#ref: https://arxiv.org/pdf/1604.07316v1.pdf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda

# Preprocess incoming data, centered around zero with small standard deviation
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3))) #Normalising the image and mean centering
model.add(Cropping2D(cropping=((70,25),(0,0))))
# Convolutional layers
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
#Fully-Connected Layers
model.add(Flatten())
#model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

# Compile and train the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train,y_train, validation_split=0.2, shuffle= True, epochs=3, verbose=1)

model.save('modelfinal.h5')

### print the keys contained in the history object
print(history_object.history.keys())

import matplotlib.pyplot as plt
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# Print out the model using Keras
model.summary()