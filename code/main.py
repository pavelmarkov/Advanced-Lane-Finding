import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#--------------------------#
import globals
import calib
import thresh
import trans
import detect
import utils
#-----------Camera Calibration---------------#
cc = calib.calibrateCam()
cc.computePoints()
mtx, dist, img, undist_img = cc.computeCoef()
cc.saveFig(img, undist_img)
#-----------Distortion-corrected image---------------#
test_img1 = cv2.imread(globals.TEST_IMGS+'test1.jpg')
test1_undist = cc.undistort(test_img1)
cc.saveFig(test_img1, test1_undist, img_name='test1_undist.png')
#-----------Binary image---------------#
test_pipline_image = mpimg.imread(globals.TEST_IMGS+'test_transform_curved.png')
binary_img = thresh.pipeline(test_pipline_image)
cv2.imwrite(globals.OUTPUT_IMGS+'binary.png', binary_img)
#-----------Perspective transform---------------#
tr = trans.PerspectTransform()
tr.computeMatrices(binary_img)
binary_warped = tr.transform(binary_img)
transform_straight = cv2.imread(globals.TEST_IMGS+'test_transform_straight.png')
transform_curved = cv2.imread(globals.TEST_IMGS+'test_transform_curved.png')
tr.verify(transform_straight, 'transform_straight.png')
tr.verify(transform_curved, 'transform_curved.png')
#-----------Identified lane-line pixels---------------#
d = detect.Detector()
leftx, lefty, rightx, righty, rect_img = d.find_lane_pixels(binary_warped)
#utils.showimg('rect_img', rect_img)
#--------------------------#
def putText(img):
  cv2.putText(img, d.curverad_str, 
            (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (255, 255, 255), 2, cv2.LINE_AA)
  cv2.putText(img, d.offset_str, 
            (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (255, 255, 255), 2, cv2.LINE_AA)

def process_frame(frame):
  undist_frame = cc.undistort(frame)
  binary_img = thresh.pipeline(undist_frame)
  binary_warped = tr.transform(binary_img)
  leftx, lefty, rightx, righty, rect_img = d.find_lane_pixels(binary_warped)
  #----------To form image for report (can be deleted)-----------------#
  if(d.frame_num == 30):
    poly_fit_img = np.dstack((binary_warped, binary_warped, binary_warped))
    poly_fit_img[lefty, leftx] = [255, 0, 0]
    poly_fit_img[righty, rightx] = [255, 0, 0]
    left_fitx, right_fitx, ploty = d.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    for i in range(-2, 3):
      poly_fit_img[ploty.astype(int), left_fitx.astype(int)+i] = [0, 255, 0]
      poly_fit_img[ploty.astype(int), right_fitx.astype(int)+i] = [0, 255, 0]
    cv2.imwrite(globals.OUTPUT_IMGS+'poly_fit.png', poly_fit_img)
    print(30)
  #---------------------------#
  warp_back_img = tr.transformInv(rect_img)
  combined_img = cv2.addWeighted(frame,1,warp_back_img,0.5,0)
  putText(combined_img)
  if(d.frame_num == 30):
    cv2.imwrite(globals.OUTPUT_IMGS+'output.png', combined_img)
  return combined_img

def process_video(video_name='../project_video.mp4'):
  cap = cv2.VideoCapture(video_name)
  fourcc = cv2.VideoWriter_fourcc(*'MP4V')
  out = cv2.VideoWriter('../output_video.mp4', fourcc, 20.0, (1280,720))
  while(cap.isOpened()):
    ret, frame = cap.read()
    if(ret==False):
      break
    d.frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
    processed_frame = process_frame(frame)
    cv2.imshow('processed_frame', processed_frame)
    out.write(processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      #cv2.imwrite(globals.TEST_IMGS+'test__.png', frame)
      break
  cap.release()
  out.release()
  cv2.destroyAllWindows()

#video_name='../challenge_video.mp4'
process_video()




