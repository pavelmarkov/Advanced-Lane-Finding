import numpy as np
import cv2
import glob # for loading multiple images at once
import globals
import matplotlib.pyplot as plt

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
  # Calculate directional gradient
  # 1) Convert to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  # 2) Take the derivative in x or y given orient = 'x' or 'y'
  if(orient=='x'):
      sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  else:
      sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  # 3) Take the absolute value of the derivative or gradient
  abs_sobel = np.absolute(sobel)
  # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
  scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
  # 5) Create a mask of 1's where the scaled gradient magnitude 
          # is > thresh_min and < thresh_max
  grad_binary = np.zeros_like(scaled_sobel)
  grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1 # Apply threshold
  # 6) Return this mask as your binary_output image
  return grad_binary
def s_chanel(img, s_thresh=(210, 255)):
  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
  l_channel = hls[:,:,1]
  s_channel = hls[:,:,2]
  s_binary = np.zeros_like(s_channel)
  s_chanel_filter = (s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])
  s_binary[s_chanel_filter] = 1
  return s_binary
def pipeline(img):
  img = np.copy(img)
  x = abs_sobel_thresh(img, orient='x', sobel_kernel=9, thresh=(30, 180))
  s = s_chanel(img, s_thresh=(110, 255))
  color = np.dstack(( np.zeros_like(x), x, s)) * 255
  grey = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
  (thresh, bw)  = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY)
  return bw