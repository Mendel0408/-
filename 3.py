# connect to gis DB and read features 连接到GIS数据库并读取特征

# Now change the directory更改工作目录
# os.chdir('C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\')

# 设置工作空间
arcpy.env.workspace = r"C:\\Users\\liat\\Documents\\buildings\\Default.gdb"
# 定义坐标参考系
wgs84 = pyproj.CRS("EPSG:4326")
itm = pyproj.CRS("EPSG:2039")

import arcpy

# 设置输入文件路径和字段名称
input = r"C:\Users\liat\Documents\buildings\points_random100Kpoints.shp"
# input = r"C:\Users\liat\Documents\buildings\motti_12_6_22\points_random200K7777.shp"

fieldx = "SHAPE@"
fieldy = "HI_PNT_Y"
fieldz = "HI_PNT_Z"
# field_height = "HEIGHT"

# 用ArcPy将输入要素类转换为NumPy数组
import arcpy

array = arcpy.da.FeatureClassToNumPyArray(input, ("SHAPE@XYZ"), skip_nulls=True)
# print(array["SHAPE@XYZ"])


import csv


# **********
# read data from the features file 读取特征点文件中的数据
# **********
def read_points_data(filename, pixel_x, pixel_y, scale):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        recs = []
        arrs = []
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}') 读取列名
                line_count += 1
                names = row
                indx = names.index(pixel_x)
                indy = names.index(pixel_y)
            else:
                line_count += 1
                symbol = row[6]
                pixel = np.array([int(row[indx]), int(row[indy])]) / scale
                # print('pixel',pixel[0],pixel[1]) 排除无效数据
                if pixel[0] > 0 and pixel[1] > 0:
                    height = float(row[5]) + float(row[2])
                    pos3d = np.array([float(row[3]), float(row[4]), height])
                    name = row[1]

                    arr = [pos3d[0], pos3d[1], float(row[5]), symbol, name]
                    rec = {'symbol': symbol,
                           'pixel': pixel,
                           'pos3d': pos3d,
                           'height': height,
                           'name': name}
                    recs.append(rec)
                    arrs.append(arr)
        print(f'Processed {line_count} lines.')
        return recs, arrs


features, arr_features_0539 = read_points_data('features.csv', 'Pixel_x_' + pixel_add_name, 'Pixel_y_' + pixel_add_name,
                                               1.0)
'''for feature in features:
    if feature['pixel'][0] >0 and  feature['pixel'][1] >0:
        print(feature['pixel'])
        print(feature['pos3d'][1])'''

# 读取图像并进行预处理
import PIL
import PIL.Image
import cv2

# check data is valid 检查数据
fc = 'C:\\Users\\liat\\Documents\\buildings\\Default.gdb\\old_city'
fields = ['POINT_X', 'POINT_Y', 'POINT_Z']
gis2d_list = []
res_list = []

calc_3d_to_2d_with_H = []

gis2d_np_part = array["SHAPE@XYZ"]
print('\ncamera_location: ', camera_location)

image = PIL.Image.open(img_name)
img = np.array(image)

# gis2d = gis2d_np_part - camera_location # remove camera locations
# print(gis2d)
for i in range(gis2d_np_part.shape[0]):
    gis2d = gis2d_np_part[i] - camera_location
    gis2d_H = np.matmul(H, gis2d)
    gis2d_H = gis2d_H / gis2d_H[2]
    calc_3d_to_2d_with_H.append(gis2d_H[0:2])

X = []
y = []
for seq, target in calc_3d_to_2d_with_H:
    X.append(seq)
    y.append(target)

features_pnt_list_3d = []
symbol_noted = []
features_pnt_list = np.array(arr_features_0539)
# print(features_pnt_list[0][0])
for i in range(features_pnt_list.shape[0]):
    pnt_xyz = [float(features_pnt_list[i][0]), float(features_pnt_list[i][1]), float(features_pnt_list[i][2])]
    # print(pnt_xyz)
    gis2d = np.array(pnt_xyz) - np.array(camera_location)
    gis2d_H = np.matmul(H, gis2d)
    gis2d_H = gis2d_H / gis2d_H[2]
    features_pnt_list_3d.append(gis2d_H[0:2])
    symbol_noted.append(features_pnt_list[i][3])

X_noted = []
y_noted = []
for seq, target in features_pnt_list_3d:
    X_noted.append(seq)
    y_noted.append(target)
m = np.array(symbol_noted)

# make segments greed points 检测图像中的角点
img = cv2.imread(pred_img_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray, 11115, 0.01, 0.5)
# corners = cv2.goodFeaturesToTrack(gray, 999995, 0.01, 0.0000000000001)
X_segments = []
y_segments = []

for corner in corners:
    x_segment, y_segment = corner[0]
    x_segment = int(x_segment)
    y_segment = int(y_segment) + y_segment_move  # add variable
    X_segments.append(x_segment)
    y_segments.append(y_segment)
    # print('x: ',x_segment,'y: ', y_segment )

X_min_segment = min(X_segments)
X_max_segment = max(X_segments)
y_min_segment = min(y_segments)
y_max_segment = max(y_segments)

'''from matplotlib import image
from matplotlib import pyplot as plt

# to read the image stored in the working directory
data = image.imread(img_name)

# to draw a point on co-ordinate (200,300)

fig, ax = plt.subplots()
fig.set_figwidth(40)
fig.set_figheight(10)
ax.scatter(X, y, s=20, marker='o', c='red')
ax.scatter(X_noted, y_noted,s=10)

for i, txt in enumerate(X_noted):
    ax.annotate(m[i], (X_noted[i], y_noted[i]), c='white')
plt.scatter(X_noted, y_noted, s=100, marker='o', c='blue')
plt.scatter(X_segments, y_segments, s=10, marker='o', c='white')
plt.imshow(data)
plt.show()'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import glob
import math


# 定义从2D图像坐标转换到3D空间坐标的函数
def fromRealImage2DTocalc3D(H_inv, camera_location, calc2Dpixel, real3D):
    # print('77H_inv',H_inv)
    # H_inv = H_inv * -1
    real3D = [real3D[0], real3D[1], real3D[2]]
    H_inv_mult_pixel = np.dot(H_inv, calc2Dpixel)
    v_norm = math.sqrt(pow(H_inv_mult_pixel[0], 2) + pow(H_inv_mult_pixel[1], 2) + pow(H_inv_mult_pixel[2], 2))

    v_3d_direction = H_inv_mult_pixel / v_norm

    # in 0518 alpha_minus = False
    alpha_minus = False
    if alpha_minus == False:
        v_3d_direction = -1 * v_3d_direction
    # print('v_3d_direction',v_3d_direction)
    # function to check that norm == 1
    v_3d_direction_norm = math.sqrt(pow(v_3d_direction[0], 2) + pow(v_3d_direction[1], 2) + pow(v_3d_direction[2], 2))
    # print('v_3d_direction_norm',v_3d_direction_norm)
    alpha = np.dot(v_3d_direction, np.subtract(real3D, camera_location))
    calc3D = np.add(np.dot(alpha, v_3d_direction), camera_location)
    d = np.subtract(calc3D, real3D)

    d_norm = math.sqrt(pow(d[0], 2) + pow(d[1], 2) + pow(d[2], 2))
    # print('d:',d_norm)
    # print('\n','\n', "real3D:",real3D,'\n',"calc3D:",calc3D,'\nalpha',alpha)
    distanceReal3DCalc3D = np.linalg.norm(real3D - calc3D)
    # print('distance real3D - calc3D :',distanceReal3DCalc3D)
    # print(7777,(d_norm),'\n','\n')
    return d_norm, alpha, calc3D


# 定义从3D坐标转换到2D图像坐标的函数
def gis3d_to_2d(pixels, pos3ds, manual_points_in_src, camera_location):
    gis2d_list = np.zeros((pixels.shape[0], 2))
    for i in range(pixels.shape[0]):  # pixels len
        # if i==1:
        # print('pos3ds[i,:]:',pos3ds[i,:],'camera_location:', camera_location )
        manual_points_in_src[i] = pixels[i, 0] != 0 or pixels[i, 1] != 0
        gis2d = pos3ds[i, :] - camera_location  # remove camera locations
        gis2d = np.array([gis2d[0], gis2d[1], gis2d[2]])
        if gis2d[2] != 0:
            gis2d = gis2d / gis2d[2]  # normal
        gis2d_list[i, :] = gis2d[0:2]
    return gis2d_list, manual_points_in_src


# good values when know 3d feature
from scipy import spatial
from scipy.spatial import distance

# 计算特征点与最近的3D点之间的距离，并进行3D重建
calc_pts = []
dists = []
ds = []
dss = []
d_norms = []
alphas = []
for i in range(len(features)):
    if features[i]['pixel'][0] > 0 and features[i]['pixel'][1] > 0:
        distances, indexes = spatial.KDTree(calc_3d_to_2d_with_H).query([X_noted[i], y_noted[i]], k=3)

        ds = []

        datas = []
        alphas = []

        d_norms = []
        pts = []
        for index in indexes:
            data = {}

            p1Real_order = np.array([calc_3d_to_2d_with_H[index][0], calc_3d_to_2d_with_H[index][1], 1])
            p1Real_order_norm = p1Real_order / p1Real_order[2]
            # print('---------------------------------------------------------------')
            d_norm, alpha, calc3D = fromRealImage2DTocalc3D(M, camera_location, p1Real_order_norm, \
                                                            arr_features_0539[i])
            # print('alpha:',alpha)
            alphas.append(alpha);
            # distance 3d point in index - real feature 3d point
            distance3dPoints = np.linalg.norm(calc3D -
                                              [arr_features_0539[i][0], arr_features_0539[i][1],
                                               arr_features_0539[i][2]])
            # print('distance3dPoints:',distance3dPoints)
            d_norms.append(d_norm)
            pts.append(calc3D)
            # print(d_norm,alpha,calc3D)
            # print('---------------------------------------------------------------')
        alphasSort = sorted(alphas)
        diff = 1
        alphasTrasholdOne = []
        for j in range(len(alphas)):
            if not alphasTrasholdOne or abs(alphasSort[j] - alphasTrasholdOne[-1]) <= diff:
                alphasTrasholdOne.append(alphasSort[j])
            else:
                newJ = alphas.index(alphasSort[j])
                if (len(d_norms) == 3):
                    d_norms.pop(newJ)
                    pts.pop(newJ)
                elif (len(d_norms) == 2):
                    d_norms.pop(newJ - 1)
                    pts.pop(newJ - 1)

        min_ds = np.min(d_norms)
        calc_pt = [arr_features_0539[i][3], pts[np.argmin(d_norms)].tolist()]
        dss.append(min_ds)
        calc_pts.append(calc_pt)
print('calc_pts : ', calc_pts)
print('ds real 3d and calc 3d : ', dss)
print(len(calc_pts), len(calc_pts))
print('ds median : ', np.median(dss))
print('ds mean : ', np.mean(dss))

# good values when know 3d feature
from scipy import spatial

# 计算图像段的3D坐标，并进行3D重建
calc_pts_segments = []
calc_3dpoly_segments = []
dists = []
ds = []
dss = []
d_norms = []
alphas = []
segment_3d = []
segment_2d = []
for i in range(len(X_segments)):
    distances, indexes = spatial.KDTree(calc_3d_to_2d_with_H).query([X_segments[i], y_segments[i]], k=3)

    # print('distances: ',distances,calc_3d_to_2d_with_H[indexes[0]],[X_segments[i], y_segments[i]])

    ds = []

    datas = []
    alphas = []
    d_norms = []
    d_all_indexes = []
    pts = []
    # for index in range(len(array["SHAPE@XYZ"])):
    for index in indexes:
        p1Real_order = np.array([X_segments[i], y_segments[i], 1])
        p1Real_order_norm = p1Real_order / p1Real_order[2]
        # print('---------------------------------------------------------------')
        d_norm, alpha, calc3D = fromRealImage2DTocalc3D(M, camera_location, p1Real_order_norm, \
                                                        array["SHAPE@XYZ"][index])
        # distance 3d point in index - real feature 3d point
        distance3dPoints = np.linalg.norm(calc3D -
                                          [array["SHAPE@XYZ"][index][0], array["SHAPE@XYZ"][index][1],
                                           array["SHAPE@XYZ"][index][2]])
        d_norms.append(d_norm)
        d_all_indexes.append(index)

        alphasSort = sorted(alphas)
        diff = 1
        alphasTrasholdOne = []
        for j in range(len(alphas)):
            if not alphasTrasholdOne or abs(alphasSort[j] - alphasTrasholdOne[-1]) <= diff:
                alphasTrasholdOne.append(alphasSort[j])
            else:
                newJ = alphas.index(alphasSort[j])
                if (len(d_norms) == 3):
                    d_norms.pop(newJ)
                    d_all_indexes.pop(newJ)
                elif (len(d_norms) == 2):
                    d_norms.pop(newJ - 1)
                    d_all_indexes.pop(newJ - 1)
        min_ds = np.min(d_norms)
        min_d_index = d_all_indexes[np.argmin(d_norms)]
        if (calc_3d_to_2d_with_H[min_d_index][0] >= X_min_segment and \
                calc_3d_to_2d_with_H[min_d_index][0] <= X_max_segment and \
                calc_3d_to_2d_with_H[min_d_index][1] >= y_min_segment and \
                calc_3d_to_2d_with_H[min_d_index][1] <= y_max_segment):
            segment_3d.append(array["SHAPE@XYZ"][min_d_index])
            segment_2d.append(calc_3d_to_2d_with_H[min_d_index])
            calc_pt = ['', array["SHAPE@XYZ"][min_d_index].tolist()]
            calc_pts_segments.append(calc_pt)
            calc_3dpoly_segments.append(array["SHAPE@XYZ"][min_d_index].tolist())
            dss.append(min_ds)
        # print('calc_pts_segments : ',calc_pts_segments)
print('len calc_pts_segments : ', len(calc_pts_segments))
# print('ds real 3d and calc 3d : ',dss)
print('ds median : ', format(np.median(dss), 'f'))
print('ds mean : ', format(np.mean(dss), 'f'))

# 绘制图像和重建结果
border_2d_points_X = []
border_2d_points_y = []
for seg in segment_2d:
    # print(index)
    # print(red_points_2d[index])
    border_2d_points_X.append(seg[0])
    border_2d_points_y.append(seg[1])

from matplotlib import image
from matplotlib import pyplot as plt

# to read the image stored in the working directory 读取并显示图像
data = image.imread(img_name)

# to draw a point on co-ordinate (200,300)

fig, ax = plt.subplots()
fig.set_figwidth(40)
fig.set_figheight(10)
ax.scatter(X, y, s=20, marker='o', c='red')
ax.scatter(border_2d_points_X, border_2d_points_y, s=200, marker='o', c='green')

plt.scatter(X_segments, y_segments, s=10, marker='o', c='white')

plt.imshow(data)
plt.show()

import alphashape
import matplotlib.pyplot as plt
from descartes import PolygonPatch

# 使用Alpha形状算法计算2D和3D的边界
points = np.array(segment_2d)

alpha = 0.5 * alphashape.optimizealpha(points)
hull = alphashape.alphashape(points, alpha)
hull_pts = hull.exterior.coords.xy

fig, ax = plt.subplots()
fig.set_figwidth(40)
fig.set_figheight(5)  # 10
plt.scatter(hull_pts[0], hull_pts[1], color='green')
a77 = PolygonPatch(hull, fill=False, color='green')
ax.add_patch(a77)
plt.gca().invert_yaxis()
plt.show()

import alphashape
import matplotlib.pyplot as plt
from descartes import PolygonPatch

points = np.array(segment_3d)[:, [0, 1]]

alpha = 0.5 * alphashape.optimizealpha(points)
hull = alphashape.alphashape(points, alpha)
hull_pts = hull.exterior.coords.xy

fig, ax = plt.subplots()
ax.scatter(hull_pts[0], hull_pts[1], color='green')
a77 = PolygonPatch(hull, fill=False, color='green')
ax.add_patch(a77)

# 构建3D多边形的顶点
hash_segment_3d = {}
for s in segment_3d:
    # hash_segment_3d[s]
    # print(round(s[0],4),round(s[1],4))
    hash_segment_3d[round(s[0], 4)] = {}
    hash_segment_3d[round(s[0], 4)][round(s[1], 4)] = round(s[2], 4)

hull_exterior_coords = list(zip(*hull.exterior.coords.xy))

res77 = []
for coord in hull_exterior_coords:
    # print(coord)
    # print(hash_segment_3d[round(coord[0],4)][round(coord[1],4)])
    x = round(coord[0], 4)
    y = round(coord[1], 4)
    z = hash_segment_3d[round(coord[0], 4)][round(coord[1], 4)]
    # print([x,y,z])
    res77.append([x, y, z])