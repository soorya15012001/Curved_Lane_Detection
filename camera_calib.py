########## For caliberating a camera take 20 images from different angles of a chess board

import cv2
import glob
import numpy as np

objpoints = []
imgpoints = []

images = glob.glob("camera_cal/*")

objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)


for image in images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if r == True:
        imgpoints.append(corners)
        objpoints.append(objp)
shape = (img.shape[0], img.shape[1])
ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

# we just need mtx and dist for finding the final output
def undistort(distorted_img):
    return cv2.undistort(distorted_img, mtx, dist, None, mtx)

test = cv2.imread("camera_cal/calibration1.jpg")
output = undistort(test)

cv2.imshow("ip", test)
cv2.imshow("test", output)
cv2.waitKey(0)

