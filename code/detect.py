import numpy as np
import cv2
import matplotlib.pyplot as plt

class Detector:
  # HYPERPARAMETERS
  # Choose the number of sliding windows
  nwindows = 9
  # Set the width of the windows +/- margin
  margin = 100
  # Set minimum number of pixels found to recenter window
  minpix = 50
  left_fit = []
  right_fit = []
  #difference in fit coefficients between last and new fits
  diffs =[0] 
  last_n_fits = []
  frame_num = 0
  # Define conversions in x and y from pixels space to meters
  ym_per_pix = 30/720 # meters per pixel in y dimension
  xm_per_pix = 3.7/700 # meters per pixel in x dimension
  left_curverad = 0
  right_curverad = 0 
  curverad_str = ''
  offset_str = ''
  def find_lane_pixels(self, binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//self.nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    if(len(self.left_fit) == 0 or (int(self.frame_num) % 10) == 0):
      left_lane_inds, right_lane_inds, out_img = self.findGoodInds(binary_warped, draw=False)
    else:
      left_lane_inds, right_lane_inds, _ = self.search_around_poly(binary_warped)
    #print(len(left_lane_inds))
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # out_img[lefty, leftx] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]

    left_fitx, right_fitx, ploty = self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    # measure curvature
    left_fit_cr = np.polyfit(ploty*self.ym_per_pix, left_fitx*self.xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*self.ym_per_pix, right_fitx*self.xm_per_pix, 2)
    self.left_curverad, self.right_curverad = self.measure_curvature_pixels(left_fit_cr, right_fit_cr, ploty)
    self.curverad_str = "Radius of curvature: {:.0f}m".format((self.left_curverad + self.right_curverad) / 2)

    # offset
    offset = self.car_offset(left_fitx, right_fitx)
    self.offset_str = "Vehicle offset from the center: {:.2f}m".format(offset)
    # print(self.curverad_str)
    # out_img[ploty.astype(int), left_fitx.astype(int)] = [0, 0, 255]
    # out_img[ploty.astype(int), right_fitx.astype(int)] = [255, 0, 0]

    # Draw green area
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    out_img = np.zeros_like(out_img)
    cv2.fillPoly(out_img, np.int_([pts]), (0,255, 0))

    return leftx, lefty, rightx, righty, out_img

  def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    self.left_fit = left_fit
    self.right_fit = right_fit
    #print('num of pts: {}'.format(len(leftx)))
    if(len(self.last_n_fits) > 10):
      self.last_n_fits.pop(0)
    if(len(self.last_n_fits) > 0):
      means = np.mean(self.last_n_fits, axis=0)
      self.diffs = np.absolute(left_fit[0:2] - means[0:2])
      #print(self.diffs)
    if(self.diffs[0] > 0.00020000 and len(leftx) < 27000):
      #print('BAD-------------->')
      self.left_fit = means
    else:
      self.last_n_fits.append(self.left_fit)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
    right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
    
    return left_fitx, right_fitx, ploty

  def findGoodInds(self, binary_warped, draw=True):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//self.nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(self.nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - self.margin
        win_xleft_high = leftx_current + self.margin
        win_xright_low = rightx_current - self.margin
        win_xright_high = rightx_current + self.margin
        
        # Draw the windows on the visualization image
        if(draw==True):
          cv2.rectangle(out_img,(win_xleft_low,win_y_low),
          (win_xleft_high,win_y_high),(0,255,0), 2) 
          cv2.rectangle(out_img,(win_xright_low,win_y_low),
          (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > self.minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > self.minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    return left_lane_inds, right_lane_inds, out_img

  def search_around_poly(self, binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
                    self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + 
                    self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
                    self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + 
                    self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))
    return left_lane_inds, right_lane_inds, binary_warped

  def measure_curvature_pixels(self, left_fit, right_fit, ploty):
    '''
    Calculates the curvature of polynomial functions.
    '''
    y_eval = np.max(ploty)
    #print(y_eval)

    # Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1 + (2*left_fit[0]*y_eval*self.ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*self.ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    return left_curverad, right_curverad

  def car_offset(self, leftx, rightx):
    ## Image mid horizontal position 
    mid_imgx = 640    
    ## Car position with respect to the lane
    car_pos = (leftx + rightx)/2   
    ## Horizontal car offset 
    offsetx = np.mean((mid_imgx - car_pos) * self.xm_per_pix)

    return offsetx