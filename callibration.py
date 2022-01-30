import numpy as np
import cv2 as cv,cv2
import glob
import matplotlib.pyplot as plt
import open3d
################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9,4)
frameSize = (1980,1080)

left = r"C:\Main Folder\Unity\stereo\Assets\screenshots\screen_1960x1080_2021-12-21_15-49-24.png"
right =r"C:\Main Folder\Unity\stereo\Assets\screenshots\screen_1960x1080_2021-12-21_15-49-27.png"

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.


imagesLeft = sorted(glob.glob(r"C:\Main Folder\Unity\stereo\Assets\screenshots/new_left/*.png"))
imagesRight = sorted(glob.glob(r"C:\Main Folder\Unity\stereo\Assets\screenshots/new_right/*.png"))

for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if retL and retR == True:

        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        #cv.imshow('img left', imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        #cv.imshow('img right', imgR)
        #cv.waitKey(2000)


cv.destroyAllWindows()


############## CALIBRATION #######################################################

retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape

#stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1],cv.CV_16SC2)

newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))
retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

#stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1],cv.CV_16SC2)

########## Stereo Vision Calibration #############################################

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same

criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
temp = newCameraMatrixR
# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, (widthR, heightR), criteria_stereo, flags)
print(trans)
print(rot)
print(newCameraMatrixL)
print(newCameraMatrixR)
print(distL)
print(distR)
print(retStereo)
x, y = cv.initUndistortRectifyMap(temp, distR, None, newCameraMatrixR, grayL.shape[::-1],cv.CV_16SC2)
dst = cv.remap(cv2.imread(right), x, y, cv.INTER_LINEAR)
plt.imshow(dst)
plt.show()

########## Stereo Rectification #################################################
rectifyScale= 1
#rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], np.identity(3),trans)
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, (widthR, heightR), rot,trans)

x, y = cv.initUndistortRectifyMap(newCameraMatrixR, distR, None, projMatrixR,  (widthR, heightR),cv.CV_16SC2)
dst = cv.remap(grayR, x, y, cv.INTER_LINEAR)
plt.imshow(dst)
plt.show()

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, (widthR, heightR),cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, (widthR, heightR), cv.CV_16SC2)


dst = cv.remap(cv2.imread(left), stereoMapL[0], stereoMapL[1], cv.INTER_LINEAR)
plt.imshow(dst)
plt.show()
dst = cv.remap(cv2.imread(right), stereoMapR[0], stereoMapR[1], cv.INTER_LINEAR)
plt.imshow(dst)
plt.show()
print("Saving parameters!")
cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()


cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()




stereo = cv2.StereoSGBM_create(numDisparities=40, blockSize=25)
left = r"C:\Main Folder\Unity\stereo\Assets\screenshots\screen_1960x1080_2021-12-21_15-49-24.png"
right =r"C:\Main Folder\Unity\stereo\Assets\screenshots\screen_1960x1080_2021-12-21_15-49-27.png"
"""
img_left = cv2.remap(cv2.imread(left), stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
img_right = cv2.remap(cv2.imread(right), stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,0)
"""
img_left = cv2.imread(left)
img_right = cv2.imread(right)
disparity = stereo.compute(cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY))
disparity_SGBM = cv2.normalize(disparity, disparity, alpha=255,beta=0, norm_type=cv2.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)
disparity_SGBM = disparity
plt.imshow(disparity_SGBM, cmap='plasma')
plt.show()
# Convert disparity map to float32 and divide by 16 as show in the documentation

#disparity_map = np.float32(np.divide(disparity_SGBM, 16.0))
disparity_map = disparity_SGBM
Q = np.float32([[1,0,0,0],
    [0,-1,0,0],
    [0,0,250*0.05,0],
    [0,0,0,1]])

# Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(disparity_map, Q, handleMissingValues=False)
# Get color of the reprojected points
colors = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
# Get rid of points with value 0 (no depth)
mask_map = disparity_map > disparity_map.min()

# Mask colors and points.
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

pcl = open3d.PointCloud()
pcl.points = open3d.Vector3dVector(output_points)
#pcl.colors = open3d.Vector3dVector(output_colors)
open3d.draw_geometries([pcl])

# Function to create point cloud file
def create_point_cloud_file(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header =''' ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
        '''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')


output_file = 'pointCloud.ply'

# Generate point cloud file
create_point_cloud_file(output_points, output_colors, output_file)

