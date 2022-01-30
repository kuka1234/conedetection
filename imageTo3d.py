import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import open3d
from datetime import datetime
img1 = cv.imread(r"C:\Main Folder\Unity\stereo\Assets\screenshots\screen_1960x1080_2021-12-29_23-29-16.png", cv.COLOR_RGB2GRAY)
img2 = cv.imread(r"C:\Main Folder\Unity\stereo\Assets\screenshots\screen_1960x1080_2021-12-29_23-29-18.png",cv.COLOR_RGB2GRAY)

mono_img = cv.imread(r"C:\Main Folder\Unity\stereo\Assets\screenshots\screen_1960x1080_2021-12-29_23-29-13.png", cv.COLOR_RGB2GRAY)


stereo = cv.StereoSGBM_create(numDisparities=40, blockSize=25)
disparity = stereo.compute(cv.cvtColor(img1, cv.COLOR_BGR2GRAY),
                               cv.cvtColor(img2, cv.COLOR_BGR2GRAY)).astype(np.float32) /16.0
#disparity = (disparity / 16.0 - minDisparity) / numDisparities

disparity_SGBM = cv.normalize(disparity, disparity, alpha=255,beta=0, norm_type=cv.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)
disp = disparity_SGBM
plt.imshow(disparity_SGBM, cmap='plasma')
plt.show()

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
h, w = mono_img.shape[:2]
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
colors = cv.cvtColor(mono_img, cv.COLOR_BGR2RGB)
#colors = img1_rectified
mask = disp > disp.min()
out_points = points[mask]
out_colors = colors[mask]

pc = open3d.PointCloud()
pc.points = open3d.Vector3dVector(out_points)
open3d.draw_geometries([pc])

"""
#out_fn = "pointCloud:" + str(datetime.now().time()) + ".ply"
out_fn = 'c.ply'
write_ply(out_fn, out_points, out_colors)
print('%s saved' % out_fn)
"""

print('Done')
