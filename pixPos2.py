import cv2
import yaml
import os
import numpy as np
import math


curPath = os.path.dirname(os.path.realpath(__file__))
yamlPath = os.path.join(curPath, "config2.yaml")
f = open(yamlPath, 'r', encoding='utf-8')
cfg = f.read()
d = yaml.load(cfg,Loader=yaml.FullLoader)

Rrow1 = d['Rrow1']
Rrow2 = d['Rrow2']
Rrow3 = d['Rrow3']

Rrow1 = np.array(Rrow1)
Rrow2 = np.array(Rrow2)
Rrow3 = np.array(Rrow3)
camera_positon = d['camera_positon']

camera_positon =np.array([[camera_positon[0]],[camera_positon[1]],[camera_positon[2]]])


R = np.array([Rrow1,Rrow2,Rrow3])
t = - np.matrix(R) * np.matrix(camera_positon)

#t = d['t']
tvec = np.array([[t[0]],[t[1]],[t[2]]])


mtx = np.array([[d['fx'],0.0,d['ux']],[0.0,d['fy'],d['uy']],[0.0,0.0,1.0]])
dist = np.array([[d['k1']],[d['k2']],[d['p1']],[d['p2']],[d['k3']]])

def uv2camera (u,v,camera_matrix,distortion_cofficients,depth):

    fx_ = camera_matrix[0, 0]
    fy_ = camera_matrix[1, 1]
    cx_ = camera_matrix[0, 2]
    cy_ = camera_matrix[1, 2]

    k1 = distortion_cofficients[0, 0]
    k2 = distortion_cofficients[1, 0]
    p1 = distortion_cofficients[2, 0]
    p2 = distortion_cofficients[3, 0]
    k3 = distortion_cofficients[4, 0]

    k1 = 0
    k2 = 0
    p1 = 0
    p2 = 0
    k3 = 0

    x = (u - cx_) * 1.0 / fx_
    y = (v - cy_) * 1.0 / fy_
    z = 1.0
    r_2 = x * x + y * y
    x_distorted = x * (1 + k1 * r_2 + k2 * r_2 * r_2 + k3 * r_2 * r_2 * r_2) + 2 * p1 * x * y + p2 * (r_2 + 2 * x * x)
    y_distorted = y * (1 + k1 * r_2 + k2 * r_2 * r_2 + k3 * r_2 * r_2 * r_2) + p1 * (r_2 + 2 * y * y) + 2 * p2 * x * y

    p_c = np.array([[x_distorted*depth],[y_distorted*depth],[z*depth]])
    #print(" p_c", p_c)

    return p_c


def camera2world (p_c,R,tvec):

    T = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    T[0, 0] = R[0, 0]
    T[0, 1] = R[0, 1]
    T[0, 2] = R[0, 2]
    T[1, 0] = R[1, 0]
    T[1, 1] = R[1, 1]
    T[1, 2] = R[1, 2]
    T[2, 0] = R[2, 0]
    T[2, 1] = R[2, 1]
    T[2, 2] = R[2, 2]
    T[0, 3] = tvec[0, 0]
    T[1, 3] = tvec[1, 0]
    T[2, 3] = tvec[2, 0]
    T[3, 0] = 0.0
    T[3, 1] = 0.0
    T[3, 2] = 0.0
    T[3, 3] = 1.0
    tmp = np.array([[0.0], [0.0], [0.0], [1.0]])
    tmp[0, 0] = p_c[0, 0]
    tmp[1, 0] = p_c[1, 0]
    tmp[2, 0] = p_c[2, 0]
    T_inv = np.linalg.inv(T)
    p_w = np.dot(T_inv,tmp)
    p_w_3D = np.array([[0.0], [0.0], [0.0]])
    p_w_3D[0, 0] = p_w[0, 0]
    p_w_3D[1, 0] = p_w[1, 0]
    p_w_3D[2, 0] = p_w[2, 0]

    return p_w_3D

def uv2world(u, v, camera_matrix, distortion_cofficients, depth ,R,tvec):

    p_c = uv2camera(u, v, camera_matrix, distortion_cofficients, depth)
    p_w_3D = camera2world(p_c, R, tvec)

    return p_w_3D

def Cal_n3_2(u,v,z):

    point_test = uv2world(u,v, mtx, dist, 1, R, tvec)
    point_test_2 = uv2world(u,v, mtx, dist, 2, R, tvec)

    tmp_x = point_test_2[0, 0] - point_test[0, 0]
    tmp_y = point_test_2[1, 0] - point_test[1, 0]
    tmp_z = point_test_2[2, 0] - point_test[2, 0]
    n3 = (z - (point_test[2, 0])) / tmp_z
    p_w_3D = np.array([[0.0], [0.0], [0.0]])
    p_w_3D[0, 0] = point_test[0, 0] + n3 * tmp_x
    p_w_3D[1, 0] = point_test[1, 0] + n3 * tmp_y
    p_w_3D[2, 0] = z

    return p_w_3D

def getPos(U, V):
    Z = 0.0; #根据坐标系提前确定高程值 
    point_world_D = Cal_n3_2(U,V,Z)
    return point_world_D

if __name__ == "__main__":
    point_world_D = getPos(1039.0, 925.0)
    print(point_world_D)


 
    





