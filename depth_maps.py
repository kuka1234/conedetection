import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d
from open3d.open3d.geometry import create_rgbd_image_from_color_and_depth

from mpl_toolkits import mplot3d

def nothing(x):
    pass

"""
cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', 600, 600)
cv2.createTrackbar('numDisparities', 'disp', 14, 17, nothing)
cv2.createTrackbar('blockSize', 'disp', 17, 50, nothing)
cv2.createTrackbar('preFilterType', 'disp', 0, 1, nothing)
cv2.createTrackbar('preFilterSize', 'disp', 2, 25, nothing)
cv2.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
cv2.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, nothing)
cv2.createTrackbar('speckleRange', 'disp',  10, 100, nothing)
cv2.createTrackbar('speckleWindowSize', 'disp', 3, 25, nothing)
cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 25, nothing)
cv2.createTrackbar('minDisparity', 'disp', 5, 25, nothing)
"""
#numDisparities = 12*16
#numDisparities=numDisparities, blockSize=17*2 + 5
stereo = cv2.StereoSGBM_create(numDisparities=40, blockSize=25)



# Setting the updated parameters before computing disparity map


left = r"C:\Main Folder\Unity\stereo\Assets\screenshots\screen_1960x1080_2021-12-21_15-49-24.png"
right =r"C:\Main Folder\Unity\stereo\Assets\screenshots\screen_1960x1080_2021-12-21_15-49-27.png"





while True:
    # Updating the parameters based on the trackbar positions
    """
    numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
    blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
    preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
    preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
    preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
    textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
    minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')


    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    #stereo.setPreFilterType(preFilterType)
    #stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    #stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)
    """

    disparity = stereo.compute(cv2.cvtColor(cv2.imread(left), cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(cv2.imread(right), cv2.COLOR_BGR2GRAY))
    #disparity = (disparity / 16.0 - minDisparity) / numDisparities

    disparity_SGBM = cv2.normalize(disparity, disparity, alpha=255,beta=0, norm_type=cv2.NORM_MINMAX)
    disparity_SGBM = np.uint8(disparity_SGBM)
    disparity_SGBM = disparity
    plt.imshow(disparity_SGBM, cmap='plasma')
    plt.show()
    """
    point3d = cv2.reprojectImageTo3D(disparity_SGBM,  np.identity(3), np.array([0., 0., 0.]), cam1_matrix, np.array([0., 0., 0., 0.]), handleMissingValues=False)
    mask_map = disparity_SGBM > disparity_SGBM.min()

    # Mask colors and points.
    output_points = point3d[mask_map]
    output_colors = disparity_SGBM[mask_map]
    """


    matrix = []
    for i in range(len(disparity_SGBM)):
        for j in range(len(disparity_SGBM[i])):
            matrix.append([i,j,disparity_SGBM[i][j]])

    pcl = open3d.PointCloud()
    pcl.points = open3d.Vector3dVector(matrix)
    open3d.draw_geometries([pcl])

    # flip the orientation, so it looks upright, not upside-down
    #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    #open3d.draw_geometries([pcd])  # visualize the point cloud


    #plt.show(points)


    #plt.pause(0.001)
    break


