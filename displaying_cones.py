import csv
import numpy as np
import matplotlib.pyplot as plt


def read_cords():
    coordinates = []
    with open("C:\Main Folder\conedetection\GFG.csv", 'r') as locations:
        csvreader = csv.reader(locations)
        for ind, row in enumerate(csvreader):
            coordinates.append([])
            for ind2, cord in enumerate(row):
                if len(cord[1:-1].split("         ")) == 1:
                    coordinates[ind].append(cord[1:-1].split("         "))
                else:
                    coordinates[ind].append(cord[1:-1].split("        "))
    print(coordinates[0])
    x, y = map(list,zip(*coordinates[0]))
    plt.scatter(x,y)
    plt.xlim(-500,500)
    plt.ylim(-500,500)
    plt.show()


plt.figure("first")
cords1 = [[0,0], [1,1], [2,0], [0,2]]
x, y = map(list,zip(*cords1))
plt.scatter(x,y)


plt.figure("second")
cords2 = [[2,2], [1,1], [2,0], [0,0]]
x, y = map(list,zip(*cords2))
plt.scatter(x,y)

def get_eigen_vectors(point1, point2): # returns vector between 2 points and 90 degrees. 1 -> 2
    vector = point2 - point1
    vector2 = np.matmul([[0,1], [-1,0]], vector)
    return vector, vector2

def get_change_of_basis(eigen1, eigen2):
    return np.linalg.inv([[eigen1[0], eigen2[0]],[eigen1[1], eigen2[1]]])

def vectors_from_point(point, coords):
    new_coords = []
    for coord in coords:
        new_coords.append(coord- point)

    return new_coords

def get_pos_of_points(matrix, coords):
    vectors = []
    for coord in coords:
        vectors.append(np.matmul(matrix, np.asarray(coord)))

    return vectors

def different_basis(b1, b2, coords):  # b1 -> b2
    new_coords = []

    for coord in coords:
        print(np.matmul(b1,coord))
        new_coord = np.matmul(b2,np.matmul(b1,coord))
        new_coords.append(new_coord)

    return new_coords

def full_process(point1, point2, coords, print_info):
    if print_info == True:
        print(point1, "to", point2)
    point1 = np.transpose((np.asarray(point1)))
    point2= np.transpose((np.asarray(point2)))
    eigen1, eigen2 = get_eigen_vectors(point1, point2)
    matrix =  get_change_of_basis(eigen1,eigen2)
    new_coords = vectors_from_point(point1, coords)
    transformed = get_pos_of_points(matrix, new_coords)

    return transformed, matrix

print(cords1)
t1, m1 = full_process(cords1[0], cords1[1], cords1, True)
print(t1)
print(m1)

print(cords2)
t2, m2 = full_process(cords2[2], cords2[1], cords2, True)
print(t2)
print(m2)

print("----------")
print(m1, m2)
print(different_basis(m1, m2, cords1))

plt.show()