import numpy as np
import cv2
import matplotlib.pyplot as plt
import globals
class PerspectTransform:
  w = 0
  h = 0
  src = []
  dst = []
  M = []
  Minv = []
  output_img_dir = globals.OUTPUT_IMGS
  def computeMatrices(self, binary_img):
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
    self.w = w
    self.h = h
    self.M = cv2.getPerspectiveTransform(self.src, self.dst)
    self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)
  def transform(self, binary_img):
    binary_warped = cv2.warpPerspective(binary_img, self.M, dsize=(self.w, self.h), flags=cv2.INTER_LINEAR)
    return binary_warped
  def transformInv(self, color_img):
    h, w, d = color_img.shape
    warp_back_img = cv2.warpPerspective(color_img, self.Minv, dsize=(w,h), flags=cv2.INTER_LINEAR)
    return warp_back_img
  def verify(self, img, name='test_transfrom.png'):
    print(
      '''
        | Source        | Destination   | 
        |:-------------:|:-------------:| 
        | {:.0f}, {:.0f}      | {:.0f}, {:.0f}        | 
        | {:.0f}, {:.0f}      | {:.0f}, {:.0f}      |
        | {:.0f}, {:.0f}      | {:.0f}, {:.0f}      |
        | {:.0f}, {:.0f}      | {:.0f}, {:.0f}        |      
      '''.format(self.src[0][0], self.src[0][1],
                self.dst[0][0], self.dst[0][1],
                self.src[1][0], self.src[1][1],
                self.dst[1][0], self.dst[1][1],
                self.src[2][0], self.src[2][1],
                self.dst[2][0], self.dst[2][1],
                self.src[3][0], self.src[3][1],
                self.dst[3][0], self.dst[3][1],)
    )
    transformed_img = self.transform(img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    pts = self.src.astype(int)
    isClosed = True
    color = (255, 0, 0) 
    thickness = 3
    img = cv2.polylines(img, [pts],  
                      isClosed, color, thickness) 
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=30)
    pts = self.dst.astype(int)
    transformed_img = cv2.polylines(transformed_img, [pts],  
                      isClosed, color, thickness) 
    ax2.imshow(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
    ax2.set_title('Bird-eye view', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(self.output_img_dir+name)
    return transformed_img  