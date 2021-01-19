#...
cc.saveFig(img, undist_img)
# cv2.imshow(winname='undist_img', mat=undist_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#.

#...
binary_img = thresh.pipeline(test_pipline_image)
# cv2.imshow('binary', binary_img)
# cv2.imwrite(globals.OUTPUT_IMGS+'binary.png', binary_img)
#.