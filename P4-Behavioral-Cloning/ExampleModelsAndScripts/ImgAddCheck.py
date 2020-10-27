import csv
import cv2
import numpy as np

lines = []
with open('C:/Users/Adarsh/Desktop/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
del lines[0]

# print(lines[0])
# print(line[0])
# source_path = line[0]
# source_path_left = line[1]
# source_path_right = line[2]
# print(source_path)
# print(source_path_left)
# print(source_path_right)
# filename = source_path.split('/')[-1]
# print(filename)
# current_path = 'C:/Users/Adarsh/Desktop/data/IMG/' + filename
# print(current_path)

images = []
measurements = []
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
    image = cv2.imread(centre_path)
    images.append(image)
    image = cv2.imread(left_path)
    images.append(image)
    image = cv2.imread(right_path)
    images.append(image)
    centre_measurement = float(line[3])
    measurements.append(centre_measurement)
    left_measurement = centre_measurement + 0.2
    measurements.append(left_measurement)
    right_measurement = centre_measurement - 0.2
    measurements.append(right_measurement)