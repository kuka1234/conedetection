import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import open3d

# 1. Detect keypoints and their descriptors
# Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

img2 = cv.imread(r"C:\Main Folder\Unity\stereo\Assets\screenshots\screen_1960x1080_2021-12-21_15-49-24.png", cv.COLOR_RGB2GRAY)
img1 = cv.imread(r"C:\Main Folder\Unity\stereo\Assets\screenshots\screen_1960x1080_2021-12-21_15-49-27.png",cv.COLOR_RGB2GRAY)
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
"""
imgSift = cv.drawKeypoints(img1, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow("SIFT Keypoints", imgSift)
cv.waitKey()
"""
# Match keypoints in both images
# Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Keep good matches: calculate distinctive image features
# Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
matchesMask = [[0, 0] for i in range(len(matches))]
good = []
pts1 = []
pts2 = []

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        # Keep this keypoint pair
        matchesMask[i] = [1, 0]
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

# Draw the keypoint matches between both pictures
# Still based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask[300:500],
                   flags=cv.DrawMatchesFlags_DEFAULT)
"""
keypoint_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches[0:500], None, **draw_params)
cv.imshow("Keypoint matches", keypoint_matches)
cv.waitKey()
"""
# ------------------------------------------------------------
# STEREO RECTIFICATION

# Calculate the fundamental matrix for the cameras
# https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

# We select only inlier points
pts1 = pts1[inliers.ravel() == 1]
pts2 = pts2[inliers.ravel() == 1]
"""
# Visualize epilines
# Adapted from: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, _ = img1src.shape
    print(r,c,_)
    #img1color = cv.cvtColor(img1src, cv.COLOR_GRAY2BGR)
    #img2color = cv.cvtColor(img2src, cv.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    i = 0
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        if i % 1 == 0:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
            img1color = cv.line(img1src, (x0, y0), (x1, y1), color, 1)
            img1color = cv.circle(img1src, tuple(pt1), 5, color, -1)
            img2color = cv.circle(img2src, tuple(pt2), 5, color, -1)
        i += 1
    return img1color, img2color


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.suptitle("Epilines in both images")
plt.show()
"""
# Stereo rectification (uncalibrated variant)
# Adapted from: https://stackoverflow.com/a/62607343
h1, w1,_ = img1.shape
h2, w2,_ = img2.shape
_, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1))
# Undistort (rectify) the images and save them
# Adapted from: https://stackoverflow.com/a/62607343
img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))


fig, axes = plt.subplots(1, 2, figsize=(15, 10))
axes[0].imshow(img1_rectified, cmap="gray")
axes[1].imshow(img2_rectified, cmap="gray")
axes[0].axhline(250)
axes[1].axhline(250)
axes[0].axhline(450)
axes[1].axhline(450)
plt.suptitle("Rectified images")
plt.show()

stereo = cv.StereoSGBM_create(numDisparities=40, blockSize=25)
disparity = stereo.compute(cv.cvtColor(img1_rectified, cv.COLOR_BGR2GRAY),
                               cv.cvtColor(img2_rectified, cv.COLOR_BGR2GRAY)).astype(np.float32) /16.0
    #disparity = (disparity / 16.0 - minDisparity) / numDisparities

disparity_SGBM = cv.normalize(disparity, disparity, alpha=255,beta=0, norm_type=cv.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)
disp = disparity
plt.imshow(disparity_SGBM, cmap='plasma')
plt.show()



"""
disp = stereo.compute(img1, img2).astype(np.float32) / 16.0
cv.imshow("",disp)
cv.waitKey()
"""
ply_header = '''ply
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
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    #verts = np.hstack([verts])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

print('generating 3d point cloud...',)
h, w = img1_rectified.shape[:2]
f = 0.8*w                          # guess for focal length
Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
Q = np.float32([[1,0,0,0],
    [0,-1,0,0],
    [0,0,250*0.05,0],
    [0,0,0,1]])
points = cv.reprojectImageTo3D(disp, Q)
#colors = cv.cvtColor(img1_rectified, cv.COLOR_BGR2RGB)
colors = img1_rectified
mask = disp > disp.min()
out_points = points[mask]
out_colors = colors[mask]
print(out_colors)
pc = open3d.PointCloud()
pc.points = open3d.Vector3dVector(out_points)
#pc.points = open3d.create_rgbd_image_from_color_and_depth(out_colors,out_points, convert_rgb_to_intensity=True)
open3d.draw_geometries([pc])
out_fn = 'out.ply'
write_ply(out_fn, out_points, out_colors)
print('%s saved' % out_fn)

cv.imshow('disparity', (disp-5)/40)
cv.waitKey()

print('Done')