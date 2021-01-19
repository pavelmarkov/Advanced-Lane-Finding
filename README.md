## Advanced Lane Finding

### Distortion Correction, Camera Calibration, Perspective Transform, Gradients and Color Spaces, Finding the Lines, Measuring Curvature.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"
[image1]: ./output_images/cal_and_undist.png "Undistorted"
[image2]: ./output_images/test1_undist.png "Road Transformed"
[image3]: ./output_images/binary.png "Binary Example"
[image4]: ./output_images/transform_curved.png "Warp Example"
[image5]: ./output_images/poly_fit.png "Fit Visual"
[image6]: ./output_images/output.png "Output"
[video1]: ./output_video.mp4 "Video"

---

### Camera Calibration

#### 1. Camera matrix and distortion coefficients. An example of a distortion corrected calibration image.

The code for this step is contained in lines 7 through 56 of the file called `calib.py` (code/calib.py), and in lines 12 through 20 of the file called `main.py` (code/main.py)

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. An example of a distortion-corrected image.

![alt text][image2]

#### 2. Color transforms, gradients or other methods to create a thresholded binary image. An example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 34 through 41 in `thresh.py`). Here's an example of my output for this step.

![alt text][image3]

#### 3. Perspective transform and an example of a transformed image.

The code for my perspective transform includes a class called `PerspectTransform`, which appears in lines 5 through 72 in the file `trans.py` (code/trans.py). The `computeMatrices()` method takes as inputs an binary image (`binary_img`), and then, using source (`src`) and destination (`dst`) points, computes perspective transform (`M`) and inverse perspective transform (`Minv`) matrices. I chose to hardcode the source and destination points in the following manner:

```python
    h, w = binary_img.shape
    self.src = np.float32(
        [[w//2-40, h//2+90],
        [w//4-20, h],#-30
        [w-w//4+20, h],#+150
        [w//2+40, h//2+90]])
    self.dst = np.float32(
        [[(w / 4)-50, 0],
        [(w / 4)-50, h],
        [(w * 3 / 4)-50, h],
        [(w * 3 / 4)-50, 0]])
```

This resulted in the following source and destination points:

|  Source  | Destination |
| :------: | :---------: |
| 600, 450 |   270, 0    |
| 300, 720 |  270, 720   |
| 980, 720 |  910, 720   |
| 680, 450 |   910, 0    |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart.

![alt text][image4]

#### 4. Identify lane-line pixels and fit their positions with a polynomial.

Class `Detector` for detectiong lane-lines on image and fitting polynomial defined in the file called `detect.py` (code/detect.py). Method `find_lane_pixels` takes binary image as input and returns color image with drawn green area between two polynomials (lane-lines).

![alt text][image5]

#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 201 through 222 in my code in `detect.py`. Function `measure_curvature_pixels` measures line curvature and function `car_offset` gives car position with respect to image center.

#### 6. Result image plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 77 through 84 in my code in `detect.py` in the function `find_lane_pixels()`. Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Link to final video output.

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Consideration of problems/issues faced in implementation of this project. What could be improved about algorithm/pipeline, and what hypothetical cases would cause the pipeline to fail.

During the project, there was a problem with detecting yellow lines on the lighter region of the road. Tuning parameters in the s-channel threshold solved the issue.

In order to "smooth" the green region, marked as "road between lane-lines" and make it more stable through all the video, I use mean values of the last n fitted polynomes' coefficients.

The pipeline to identify the lane-line might fail with different lighting conditions or in presence of other line-like objects near the lane-lines.
Also, it's critical for lines to have "not too extreme" curvature and to be visible for a solution to work.

The approach I took can be improved by adjusting threshold parameters of "sobel" and "s-channel" depending on how bright/dark a given frame is.
