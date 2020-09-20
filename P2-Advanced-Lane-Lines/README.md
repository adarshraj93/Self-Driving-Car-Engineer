## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

 
## Introduction
In this project, the primary goal of the project is to develop a pipeline to identify the lane boundaries in a video. The video stream should be able to detect the lane boundaries along with radius of curvature of the lane lines and vehicle psotion with respect to the centre of the lane.

The final image needs to look like this:
![Lanes Image](./examples/example_output.jpg)

## Goals of the Project
The goals / steps of this project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Reflection
### 1. Camera Calibration
The OpenCv functions cv2.findChessboardCorners(), cv2.drawChessboardCorners() and cv2.calibrateCamera() are used to calibrate the camera. The 20 images located in ./camera_cal are used as input for camera calibration.

cv2.findChessboardCorners() determines the input image as a view of the chessboard pattern and locates the internal chessboard corners. cv2.drawChessboardCorners() draws draws the individual chessboard corners.

The object points, which is the location of the chessboard corners in real world space is appended with positive corner detection with every image. The image points which is the pixel location is also appended. These image and object points are inputed in the function cal_undistort() to calibrate the camera using cv2.calibrateCamera(). The object and image points are appened in the function "camera_calibrate()"" in the code

cv2.calibrateCamera() returns the camera matrix (mtx) and distortion coefficients that is used for distortion correction of images in the function cv2.undistort(). These are defined in the function "cal_undistort()"
``` python
def cal_undistort(img, objpoints, imgpoints):
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    
    # Undistort the Image
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return mtx, dist, dst
```

Example of Undistorted Camera Calibration Image:
![](output_images/Image1.PNG)


### 2. Distortion Correction on Test Images
The above function "cal_undistort()" is used on the test images for distortion correction

Example of distortion correction on Raw Image:
![](output_images/Image2.PNG)


### 3. Apply Color and Gradient Thresholds
The sobel operater in the X direction, the magnitude of the gradient and direction of the gardient is used to filter out pixels that aren't of interest. The RGB image is converted to HLS color space to apply color thresholding. The S channel is used to identify the lane lines. The final image is a binary image with all the combined color and gradient threshold.

Example of color and gradient threshold:
![](output_images/Image3.PNG)


### 4. Perspective Transform
The perspective transform is used to obtain a bird's eye view of the camera view. cv2.getPerspectiveTransform() is used to do this is task. The input is the thresholded binary image

Example of prespective transform:
![](output_images/Image4.PNG)
