import open3d as o3d
from open3d.open3d.geometry import radius_outlier_removal, statistical_outlier_removal, select_down_sample
import numpy as np
import copy
import cone_detection_v2 as cd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from csv import writer


def display_inlier_outlier(cloud, ind): # displays difference in point clouds
    inlier_cloud = select_down_sample(cloud, ind)
    outlier_cloud = select_down_sample(cloud,ind, invert=True)
    #print("Showing outliers (red) and inliers (gray): ")
    #outlier_cloud.paint_uniform_color([1, 0, 0])
    #inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    #o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    inlier_cloud.colors = colors
    pcds.append(inlier_cloud)

def draw_registration_result(source, target, transformation): # paints different parts of point clouds
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def construct_img(cords, colors):  # creates colored image out of points in point cloud
    constructed_img = np.zeros([1080, 1920, 3], dtype=np.uint8)
    for ind, cord in enumerate(cords):
        try:
            (constructed_img[int(cord[1])])[int(cord[0])] = colors[ind] * 255
        except:
            pass
    print("Image constructed.")
    return constructed_img

def get_cone_centres(constructed_img): #returns centre and cords of cones in 2d image
    obj = cd.cone_detect
    cones, centres = obj.get_pixels(obj, constructed_img)
    print("Cone coordinates found.")
    return cones, centres

def get_depth_cone(points, cones):
    plotting = [[], [], []]
    for ind, cone in enumerate(cones):
        average_depth = []
        for i, pixel in enumerate(cone[0]):
            if i% 20 == 0:
                index = np.where((points[:, 0] == pixel[1]) & (-points[:, 1] == pixel[0]))
                try:
                    average_depth.append(points[index][0][2])
                except:
                    pass

        plotting[0].append(centres[ind][0])
        plotting[1].append(centres[ind][1])
        plotting[2].append(np.average(average_depth))
    print("Depth found")
    return plotting

def post_process_cloud(file):
    pcds = [] #stores point clouds
    pcd = o3d.io.read_point_cloud(file)
    pcds.append(pcd)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    mask = points[:,2] > 200  # removes background
    #mask = points[:,0] <650
    pcd.points = o3d.utility.Vector3dVector(points[mask])  # creates new points
    pcd.colors = o3d.utility.Vector3dVector(colors[mask])  # updates new colors
    #o3d.visualization.draw_geometries(pcds)

    #pcd.paint_uniform_color([0,0,0])
    cl, ind = statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio = 0.020)  # removes noise
    #display_inlier_outlier(pcd, ind)
    #points = np.asarray(pcds[0].points).copy()
    points = np.asarray(cl.points)
    cords = np.column_stack([points[:,0],-points[:,1]])  # stores cords of all points
    colors =  np.asarray(cl.colors)
    print("Finished post-processing")
    return points, cords, colors


points, cords, colors = post_process_cloud(r"C:\Main Folder\conedetection\c.ply")

constructed_img = construct_img(cords, colors)
cones, centres = get_cone_centres(constructed_img)

plotting = get_depth_cone(points, cones)

save_list = np.column_stack([plotting[0],plotting[2]])
"""
with open('GFG.csv', 'a') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(save_list)
    f_object.close()
"""

print(plotting)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(plotting[0], plotting[1], plotting[2])
ax.axes.set_xlim3d(left=0, right=1920)
ax.axes.set_ylim3d(bottom=0, top=1080)
ax.axes.set_zlim3d(bottom=0, top=2000)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("depth")
plt.show()




"""
average_depth = []
for pixel in (pixels[2])[0]:
    #print(pixel)
    temp_p = points[:, 0] == pixel[0]
    temp_p_2 = points[:, 1] == -pixel[1]
    #combined = np.column_stack((points[:,0],points[:,1]))
    #asdf = np.isin(combined, pixel)
    #print(points[np.logical_and(temp_p,temp_p_2)])
    #average_depth.append(points[np.logical_and(temp_p,temp_p_2)][0][2])
    #print(points[np.logical_and(temp_p,temp_p_2)])

print(average_depth)
"""
"""
pcd = o3d.io.read_point_cloud(r"C:\Main Folder\conedetection\c.ply")
points = np.asarray(pcd.points)
mask = points[:,2] > 200
pcd.points = o3d.utility.Vector3dVector(points[mask])
pcd.paint_uniform_color([0.5,1,1])
cl, ind = statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio = 0.020)
display_inlier_outlier(pcd, ind)

#o3d.visualization.draw_geometries(pcds)
trans_init = np.asarray([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

draw_registration_result(pcds[0], pcds[1], trans_init)

print("Apply point-to-point ICP")
reg_p2p = o3d.registration.registration_icp(
    pcds[0], pcds[1], 0.02, trans_init,
    o3d.registration.TransformationEstimationPointToPoint(),o3d.registration.ICPConvergenceCriteria(max_iteration=200))
print(reg_p2p)
draw_registration_result(pcds[0], pcds[1], reg_p2p.transformation)
"""