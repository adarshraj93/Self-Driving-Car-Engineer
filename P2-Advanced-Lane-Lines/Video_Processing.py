import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle
from moviepy.editor import VideoFileClip

mtx = None
dist = None
M = None
Minv = None
line_prev = None
num_full_search = 0

class Line():
    def __init__(self):
        self.fullsearch = False
        self.left_lane_inds = None
        self.right_lane_inds = None
        self.left_fit = None
        self.right_fit = None
        self.left_fit_cr = None
        self.right_fit_cr = None
        self.yvals = None
        self.left_fitx = None
        self.right_fitx = None
        self.y_bottom = None
        self.y_top = None
        self.left_x_bottom = None
        self.left_x_top = None
        self.right_x_bottom = None
        self.right_x_top = None
        self.left_curverads = None
        self.right_curverads = None
        self.mean_left_curverad = None
        self.mean_right_curverad = None

def calibrate_camera(path):
    ## Prepare object points
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    ## Local arrays to store object & image points from all the images
    objpoints = []
    imgpoints = []


    images = glob.glob(path)
    errcount = 0
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            image_dims = (img.shape[0], img.shape[1])
        else:
            errcount +=1

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_dims, None, None)
    return mtx, dist

def cal_undistort(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def gradient_color_thresh(img):
    # Apply Sobel and and magnitude gradient
    ksize = 5
    grad_binary = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(30,150))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thres=(50,100))
    dir_binary = dir_thresh(img, sobel_kernel=ksize, dir_thres=(0.7,1.3))
    # Apply Color threshold (along the s-channel in HLS color space)
    s_binary = hls_select(img,s_thresh=(175,250))

    # Combine both gradient & color threshold
    combined_binary = np.zeros_like(grad_binary)
    combined_binary[(((grad_binary == 1) & (mag_binary == 1) & (dir_binary == 1))| (s_binary == 1))] = 1
    #combined_binary[((grad_binary == 1) | (s_binary == 1))] = 1
    return combined_binary

def abs_sobel_thresh(img, orient='x', sobel_kernel=9, thresh=(0,255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary

def mag_thresh(img, sobel_kernel=9, mag_thres=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Calculate the magnitude
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobel_mag/np.max(sobel_mag))
    # 5) Create a binary mask where mag thresholds are met
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= mag_thres[0]) & (scaled_sobel <= mag_thres[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary

def dir_thresh(img, sobel_kernel=9, dir_thres=(0, np.pi/2)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary =  np.zeros_like(absgraddir)
    binary[(absgraddir >= dir_thres[0]) & (absgraddir <= dir_thres[1])] = 1
    # Return the binary image
    return binary

def hls_select(img, s_thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    # 2) Apply a threshold to the S channel
    binary = np.zeros_like(s)
    binary[(s > s_thresh[0]) & (s <= s_thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary

def perspective_transform(img):
    # define 4 source points for perspective transformation
    src = np.float32([[220,719],[1220,719],[750,480],[550,480]])
    # define 4 destination points for perspective transformation
    dst = np.float32([[240,719],[1040,719],[1040,300],[240,300]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    # Return the resulting image
    return warped, M

def Initial_find_line_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 20
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    return left_lane_inds, right_lane_inds, out_img

def process_fit(binary_warped, left_lane_inds, right_lane_inds):
    left_fit, right_fit = curve_fit(binary_warped, left_lane_inds, right_lane_inds)
    left_fit_cr, right_fit_cr = convert_pixel_to_meters(binary_warped, left_lane_inds, right_lane_inds)
    yvals, left_fitx, right_fitx = fit_lines(binary_warped, left_fit, right_fit)

    line = Line()
    line.left_lane_inds = left_lane_inds
    line.right_lane_inds = right_lane_inds
    line.left_fit = left_fit
    line.right_fit = right_fit
    line.left_fit_cr = left_fit_cr
    line.right_fit_cr = right_fit_cr
    line.yvals = yvals
    line.left_fitx = left_fitx
    line.right_fitx = right_fitx
    line.y_bottom = np.min(yvals)
    line.y_top = np.max(yvals)
    line.left_x_bottom = left_fit[0]*line.y_bottom**2 + left_fit[1]*line.y_bottom + left_fit[2]
    line.left_x_top = left_fit[0]*line.y_top**2 + left_fit[1]*line.y_top + left_fit[2]
    line.right_x_bottom = right_fit[0]*line.y_bottom**2 + right_fit[1]*line.y_bottom + right_fit[2]
    line.right_x_top = right_fit[0]*line.y_top**2 + right_fit[1]*line.y_top + right_fit[2]
    left_curverads, right_curverads = radius_of_curvatures(line.yvals, left_fit_cr, right_fit_cr)
    line.left_curverads = left_curverads
    line.right_curverads = right_curverads
    line.mean_left_curverad = np.mean(left_curverads)
    line.mean_right_curverad = np.mean(right_curverads)

    return line

def curve_fit(binary_warped, left_lane_inds, right_lane_inds):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

def convert_pixel_to_meters(binary_warped, left_lane_inds, right_lane_inds):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    return left_fit_cr, right_fit_cr

def fit_lines(binary_warped, left_fit, right_fit):
    yvals = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
    right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]
    return yvals, left_fitx, right_fitx

def radius_of_curvatures(yvals, left_fit, right_fit):
    left_curverads = ((1 + (2*left_fit[0]*yvals + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverads = ((1 + (2*right_fit[0]*yvals + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    return left_curverads, right_curverads

def find_lines_prior(binary_warped, left_fit, right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) +
        left_fit[1]*nonzeroy + left_fit[2] - margin)) &
    (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) +
        right_fit[1]*nonzeroy + right_fit[2] - margin)) &
    (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit_new = np.polyfit(lefty, leftx, 2)
    right_fit_new = np.polyfit(righty, rightx, 2)

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    return left_lane_inds, right_lane_inds, out_img

def is_good_fit(prev, curr):
    # check if left_x_bottom and right_x_bottom are within 15 pixels
    if abs(prev.left_x_bottom - curr.left_x_bottom) <= 15:
        if abs(prev.right_x_bottom - curr.right_x_bottom) <= 15:
                if abs(curr.mean_left_curverad) < (abs(prev.mean_left_curverad*100)):
                    if abs(curr.mean_right_curverad) < (abs(prev.mean_right_curverad*100)):
                        return True
    return False

def draw_lines(undist, warped, yvals, left_fitx, right_fitx, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (mtxinv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def annotate_result(result, line):
    lx = line.left_x_top
    rx = line.right_x_top
    xcenter = np.int(result.shape[1]/2)
    offset = (rx - xcenter) - (xcenter - lx)
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    vehicle_offset =  offset * xm_per_pix

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Radius of curvature_Left  = %.2f m' % (line.mean_left_curverad),
        (5, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'Radius of curvature_Right = %.2f m' % (line.mean_left_curverad),
        (5, 60), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'Vehicle is %.2f m from lane center' % (vehicle_offset),
               (5, 90), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return result

def process_image(img):
    global mtx, dist, line_prev, num_full_search
    if mtx is None or dist is None:
         mtx, dist = calibrate_camera("./camera_cal/calibration*.jpg")
    undist_img = cal_undistort(img, mtx, dist)
    thresh_binary = gradient_color_thresh(undist_img)
    binary_warped, M = perspective_transform(thresh_binary)

    left_lane_inds = None
    right_lane_inds = None
    out_img = None
    plotSearchArea = True
    line = None
    if line_prev is None:
        left_lane_inds, right_lane_inds, out_img = Initial_find_line_pixels(binary_warped)
        plotSearchArea = False
        line = process_fit(binary_warped, left_lane_inds, right_lane_inds)
        num_full_search = num_full_search + 1
    else:
        left_lane_inds, right_lane_inds, out_img = find_lines_prior(binary_warped,
            line_prev.left_fit, line_prev.right_fit)
        line = process_fit(binary_warped, left_lane_inds, right_lane_inds)
        # check for a good fit
        if is_good_fit(line_prev, line) is False:
            left_lane_inds, right_lane_inds, out_img = Initial_find_line_pixels(binary_warped)
            plotSearchArea = False
            line = process_fit(binary_warped, left_lane_inds, right_lane_inds)
            num_full_search = num_full_search + 1

    result = draw_lines(undist_img, binary_warped, line.yvals, line.left_fitx, line.right_fitx, np.linalg.inv(M))

    Final_result = annotate_result(result, line)

    line_prev = line
    return Final_result

def process_video():
    global num_full_search
    output1 = 'project_video_output.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    output_clip1 = clip1.fl_image(process_image)
    output_clip1.write_videofile(output1, audio=False)

    output2 = 'challenge_video_output.mp4'
    clip2 = VideoFileClip("challenge_video.mp4")
    output_clip2 = clip2.fl_image(process_image)
    output_clip2.write_videofile(output2, audio=False)

    output3 = 'harder_challenge_video_output.mp4'
    clip3 = VideoFileClip("harder_challenge_video.mp4")
    output_clip3 = clip3.fl_image(process_image)
    output_clip3.write_videofile(output3, audio=False)

    print("Num full searches", num_full_search)
    return output_clip1, output2, output3
    #return output2

#output = process_video()
