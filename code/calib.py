import numpy as np
import cv2
import glob # for loading multiple images at once
import globals
import matplotlib.pyplot as plt

class calibrateCam:
	# Make a list of calibration images
	images_for_cal_path = globals.CALIB_IMGS
	img_for_test_path = globals.CALIB_TEST_IMG
	output_img_dir = globals.OUTPUT_IMGS
	# prepare object points
	nx = 9  # the number of inside corners in x
	ny = 6  # the number of inside corners in y
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((ny*nx, 3), np.float32)
	objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
	# Arrays to store object points and image points from all the images.
	objpoints = []  # 3d points in real world space
	imgpoints = []  # 2d points in image plane.
	# camera calibration matrix and distortion coefficients
	mtx = []
	dist = []
	def computePoints(self):
		# Step through the list and search for chessboard corners
		images_for_cal = glob.glob(self.images_for_cal_path)
		for fname in images_for_cal:
			img = cv2.imread(fname)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			# Find the chessboard corners
			ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
			# If found, add object points, image points
			if ret == True:
				self.objpoints.append(self.objp)
				self.imgpoints.append(corners)
	def computeCoef(self):
		# Draw and display the corners
		img = cv2.imread(self.img_for_test_path)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, 
																												gray.shape[::-1], None, None)
		undist_img = cv2.undistort(img, mtx, dist, None, mtx)
		self.mtx = mtx
		self.dist = dist
		return mtx, dist, img, undist_img
	def undistort(self, img):
		return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
	def saveFig(self, img, undist_img, img_name='cal_and_undist.png'):
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
		f.tight_layout()
		ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		ax1.set_title('Original Image', fontsize=50)
		ax2.imshow(cv2.cvtColor(undist_img, cv2.COLOR_BGR2RGB))
		ax2.set_title('Undistorted Image', fontsize=50)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		plt.savefig(self.output_img_dir+img_name)
	
# cc = calibrateCam()
# cc.computePoints()
# mtx, dist, undist_img = cc.computeCoef()
# cv2.imshow(winname='undist_img', mat=undist_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()