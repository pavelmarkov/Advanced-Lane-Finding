import cv2
def showimg(name, img):
  cv2.imshow(winname=name, mat=img)
  cv2.waitKey(0)
  cv2.destroyWindow(name)