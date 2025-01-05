# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import glob
import math


# **********
# Calculate true and pixel distances between features
# **********
def correlate_features(features, depth_val):
    result = ['id', 'sym_s', 'x_s', 'y_s', 'pixel_x_s', 'pixel_y_s', 'calc_pixel_x_s', 'calc_pixel_y_s',
              'sym_t', 'x_t', 'y_t', 'pixel_x_t', 'pixel_y_t', 'calc_pixel_x_t', 'calc_pixel_y_t',
              'dis_m_x', 'dis_m_y', 'dis_m', 'dis_pix_x', 'dis_pix_y', 'dis_pix', 'dis_c_pix_x', 'dis_c_pix_y',
              'dis_c_pix', 'bear_pix', 'dis_depth_pix', 'bear_c_pix', 'dis_depth_c_pix']

    results = []
    results.append(result)
    count = 1
    i = 0
    j = 0
    features.remove(features[0])  # remove the headers
    features.sort()  # sort alphabethically
    for f1 in features:
        i = j
        while i < len(features):
            if f1[1] != features[i][1]:
                dis_m_x = int(features[i][3]) - int(f1[3])
                dis_m_y = int(features[i][4]) - int(f1[4])
                dis_m = math.sqrt(math.pow(dis_m_x, 2) + math.pow(dis_m_y, 2))

                if f1[5] != 0 and features[i][5] != 0:
                    dis_pix_x = int(features[i][5]) - int(f1[5])
                    dis_pix_y = int(features[i][6]) - int(f1[6])
                else:
                    dis_pix_x = 0
                    dis_pix_y = 0
                dis_pix = math.sqrt(math.pow(dis_pix_x, 2) + math.pow(dis_pix_y, 2))

                if features[i][7] != 0 and f1[7] != 0:
                    dis_c_pix_x = int(features[i][7]) - int(f1[7])
                    dis_c_pix_y = int(features[i][8]) - int(f1[8])
                else:
                    dis_c_pix_x = 0
                    dis_c_pix_y = 0
                dis_c_pix = math.sqrt(math.pow(dis_c_pix_x, 2) + math.pow(dis_c_pix_y, 2))

                bear_pix = calc_bearing(f1[5], f1[6], features[i][5], features[i][6])
                if bear_pix != 0 and bear_pix <= 180:
                    dis_depth_pix = (abs(bear_pix - 90) / 90 + depth_val) * dis_pix
                elif bear_pix != 0 and bear_pix > 180:
                    dis_depth_pix = (abs(bear_pix - 270) / 90 + depth_val) * dis_pix
                else:
                    dis_depth_pix = 0

                bear_c_pix = calc_bearing(f1[7], f1[8], features[i][7], features[i][8])
                if bear_c_pix != 0 and bear_c_pix <= 180:
                    dis_depth_c_pix = (abs(bear_c_pix - 90) / 90 + depth_val) * dis_c_pix
                elif bear_c_pix != 0 and bear_c_pix > 180:
                    dis_depth_c_pix = (abs(bear_c_pix - 270) / 90 + depth_val) * dis_c_pix
                else:
                    dis_depth_c_pix = 0

                result = [str(count), f1[1], f1[3], f1[4], f1[5], f1[6], f1[7], f1[8], features[i][1], features[i][3],
                          features[i][4], features[i][5], features[i][6], features[i][7], features[i][8],
                          dis_m_x, dis_m_y, dis_m, dis_pix_x, dis_pix_y, dis_pix, dis_c_pix_x, dis_c_pix_y, dis_c_pix,
                          bear_pix, dis_depth_pix, bear_c_pix, dis_depth_c_pix]

                results.append(result)
                count += 1
            i += 1
        j += 1
    return results


# **********
# Calculation of the bearing from point 1 to point 2
# **********
def calc_bearing(x1, y1, x2, y2):
    if x1 == 0 or x2 == 0 or y1 == 0 or y2 == 0:
        degrees_final = 0
    else:
        deltaX = x2 - x1
        deltaY = y2 - y1

        degrees_temp = math.atan2(deltaX, deltaY) / math.pi * 180

        if degrees_temp < 0:
            degrees_final = 360 + degrees_temp
        else:
            degrees_final = degrees_temp

        if degrees_final < 180:
            degrees_final = 180 - degrees_final
        else:
            degrees_final = 360 + 180 - degrees_final

    return degrees_final


# **********
# Camera calibration process
# **********
def calibrate_camera(size):
    CHECKERBOARD = (6, 9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, size, 0.001)  # was 30

    objpoints = []  # Creating vector to store vectors of 3D points for each checkerboard image
    imgpoints = []  # Creating vector to store vectors of 2D points for each checkerboard image

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    images = glob.glob(
        '.\camera_calibration\images\*.jpg')  # TODO: change the path according to the path in your environmrnt
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            print(fname)

        cv2.waitKey(0)

    cv2.destroyAllWindows()
    h, w = img.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs


# **********
# Find homographies function
# **********
def find_homographies(recs, camera_locations, im, show, ransacbound, outputfile):
    # print(recs)
    pixels = []
    pos3ds = []
    symbols = []
    for r in recs:
        pixels.append(r['pixel'])
        pos3ds.append(r['pos3d'])
        symbols.append(r['symbol'])
    pixels = np.array(pixels)
    pos3ds = np.array(pos3ds)
    symbols = np.array(symbols)
    loc3ds = []
    grids = []
    for cl in camera_locations:
        grids.append(cl['grid_code'])
        loc3ds.append(cl['pos3d'])
    grids = np.array(grids)
    loc3ds = np.array(loc3ds)
    num_matches = np.zeros((loc3ds.shape[0], 2))
    scores = []
    for i in range(0, grids.shape[0], 1):  # 50
        if grids[i] >= grid_code_min:
            if show:
                print(i, grids[i], loc3ds[i])
            num_matches[i, 0], num_matches[i, 1] = find_homography(recs, pixels, pos3ds, symbols, loc3ds[i], im, show,
                                                                   ransacbound, outputfile)
        else:
            num_matches[i, :] = 0
        score = [i + 1, num_matches[i, 0], num_matches[i, 1], grids[i], loc3ds[i][0], loc3ds[i][1], loc3ds[i][2]]
        scores.append(score)

    if show is False:
        outputCsv = output.replace(".png", "_location.csv")
        csvFile = open(outputCsv, 'w', newline='', encoding='utf-8')
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(['location_id', 'min_score', 'max_score', 'grid_code', 'Z', 'X', 'Y'])
        for s in scores:
            csvWriter.writerow(s)

    return num_matches


def from2d_to_3d(point_2d, H):
    point_3d = np.matmul(np.linalg.inv(H), point_2d)
    point_3d = point_3d / point_3d[2]
    return point_3d


def fromRealImage2DTocalc3D(H_inv, camera_location, calc2Dpixel, real3D):
    H_inv_mult_pixel = np.dot(H_inv, calc2Dpixel)
    v_norm = math.sqrt(pow(H_inv_mult_pixel[0], 2) + pow(H_inv_mult_pixel[1], 2) + pow(H_inv_mult_pixel[2], 2))

    v_3d_direction = H_inv_mult_pixel / v_norm
    v_3d_direction = -1 * v_3d_direction
    # function to check that norm == 1
    v_3d_direction_norm = math.sqrt(pow(v_3d_direction[0], 2) + pow(v_3d_direction[1], 2) + pow(v_3d_direction[2], 2))
    print('v_3d_direction_norm', v_3d_direction_norm)
    alpha = np.dot(v_3d_direction, np.subtract(real3D, camera_location))
    calc3D = np.add(np.dot(alpha, v_3d_direction), camera_location)
    d = np.subtract(calc3D, real3D)

    d_norm = math.sqrt(pow(d[0], 2) + pow(d[1], 2) + pow(d[2], 2))
    print('d_norm:', d_norm)
    print('\n', '\n', "real3D:", real3D, '\n', "calc3D:", calc3D, '\nalpha', alpha)
    # print(7777,(d_norm),'\n','\n')
    return d_norm


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


# **********
# Find homography function
# **********
def find_homography(recs, pixels, pos3ds, symbols, camera_location, im, show, ransacbound, outputfile):
    # manual_points_in_src = np.zeros(pixels.shape[0])#manual_points_in_src = real point or not - 1/0
    # print('recs77: ',recs)
    pos2 = np.zeros((pixels.shape[0], 2))
    good = np.zeros(pixels.shape[0])
    gis2d_list, manual_points_in_src = gis3d_to_2d(pixels, pos3ds, good, camera_location)
    for i in range(pixels.shape[0]):
        good[i] = pixels[i, 0] != 0 or pixels[i, 1] != 0
        p = pos3ds[i, :] - camera_location
        p = np.array([p[0], p[1], p[2]])
        if p[2] != 0:
            p = p / p[2]
        pos2[i, :] = p[0:2]
    # print(pixels)

    if show:
        # M, mask = cv2.findHomography(pos2[good==1],pixels[good==1],cv2.LMEDS)
        M, mask = cv2.findHomography(pos2[good == 1], pixels[good == 1], cv2.RANSAC, 1000)
        # M, mask = cv2.findHomography(pos2[good==1],pixels[good==1],cv2.RANSAC,120)
    else:
        M, mask = cv2.findHomography(pos2[good == 1], pixels[good == 1], cv2.RANSAC, 120)
    M = np.linalg.inv(M)
    H_inv = np.linalg.inv(M)
    greens_features = []
    if show:
        print('Mmask77', np.sum(mask))
        print('M77', M)
    if show:
        plt.figure(figsize=(40, 20))
        plt.imshow(im)
        for rec in recs:
            symbol = rec['symbol']
            pixel = rec['pixel']
            if pixel[0] != 0 or pixel[1] != 0 and symbol == 'A2':
                plt.text(pixel[0], pixel[1], symbol, color='yellow', fontsize=20, weight='bold')
            else:
                print('black: ', rec['name'])

                # plt.text(pixel[0],pixel[1],symbol, style='italic',fontsize=30, weight ='bold', bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))
    err1 = 0
    err2 = 0
    feature = ['id', 'symbol', 'name', 'x', 'y', 'pixel_x', 'pixel_y', 'calc_pixel_x', 'calc_pixel_y']
    features = []
    features.append(feature)
    for i in range(pos2[good == 1].shape[0]):
        pp = np.array([pos2[good == 1][i, 0], pos2[good == 1][i, 1], 1.0])
        p1Real = pixels[good == 1][i, :]
        real3d = pos3ds[good == 1][i, :]

        p2Real = np.array([gis2d_list[good == 1][i, 0], gis2d_list[good == 1][i, 1], 1.0])
        p1Calc = from2d_to_3d(p2Real, H_inv)

        if show:
            p2Real_order = np.array([pp[0], pp[1], pp[2]])
            p2Real_order_norm = p2Real_order / p2Real_order[2]
            # p1Calc_order = from2d_to_3d(p2Real_order_norm,H_inv)PP2

            p1Real_order = np.array([p1Real[0], p1Real[1], 1])
            p1Real_order_norm = p1Real_order / p1Real_order[2]
            p1Calc_order = from2d_to_3d(p2Real_order_norm, M)
            fromRealImage2DTocalc3D(M, camera_location, p1Real_order_norm, real3d);
            print('H', M)
            print('H inv', H_inv)

            print("p1Calc:", p1Calc, "p1Calc_order:", p1Calc_order, '\n', "p1Real:" \
                  , p1Real, '\n', \
                  "real3d:", real3d, '\n', "camera_location:", camera_location, '\n')

        p1 = pixels[good == 1][i, :]
        pp = np.array([pos2[good == 1][i, 0], pos2[good == 1][i, 1], 1.0])
        pp2 = np.matmul(np.linalg.inv(M), pp)
        pp2 = pp2 / pp2[2]
        P1 = np.array([p1[0], p1[1], 1.0])
        PP2 = np.matmul(M, P1)
        PP2 = PP2 / PP2[2]
        P2 = pos2[good == 1][i, :]
        if show and good[i]:
            print(i)
            print(mask[i] == 1, p1, pp2[0:2], np.linalg.norm(p1 - pp2[0:2]))
            print(mask[i] == 1, P2, PP2[0:2], np.linalg.norm(P2 - PP2[0:2]))
        if mask[i] == 1:
            err1 += np.linalg.norm(p1 - pp2[0:2])
            err2 += np.linalg.norm(P2 - PP2[0:2])
        if show:
            distance2dPoints = []
            distance2dPoint = np.linalg.norm(p1Real - [pp2[0], pp2[1]])
            print('distance between real 2d and calc 2d:', distance2dPoint)
            distance2dPoints.append(distance2dPoint)
            color = 'green' if mask[i] == 1 else 'red'
            plt.plot([p1[0], pp2[0]], [p1[1], pp2[1]], color=color, linewidth=3)
            plt.plot(p1[0], p1[1], marker='X', color=color, markersize=5)
            plt.plot(pp2[0], pp2[1], marker='o', color=color, markersize=5)
            if color == 'green':
                distance2dPoints.append(distance2dPoint)
            sym = ''
            name = ''
            for r in recs:
                px = r['pixel'].tolist()
                if px[0] == p1[0] and px[1] == p1[1]:
                    sym = r['symbol']
                    name = r['name']
                    x = r['pos3d'][0]
                    y = r['pos3d'][1]
                    break
            feature = [i, sym, name, x, y, p1[0], p1[1], pp2[0], pp2[1]]
            if (color == 'green'):
                print('green: ', feature)
            else:
                print('red: ', feature)
                greens_features.append(feature)
            features.append(feature)

    i = -1
    for r in recs:  # Extracting features that were not noted on the image (pixel_x and pixel_y are 0)
        i += 1
        p1 = pixels[i, :]
        if p1[0] == 0 and p1[1] == 0:
            pp = np.array([pos2[i, 0], pos2[i, 1], 1.0])
            pp2 = np.matmul(np.linalg.inv(M), pp)
            pp2 = pp2 / pp2[2]
            if show:
                if r['symbol'] == 'A2':
                    plt.text(pp2[0], pp2[1], r['symbol'], color='black', fontsize=30, style='italic',
                             weight='bold')
                    plt.plot(pp2[0], pp2[1], marker='s', markersize=10, color='black')
                x = r['pos3d'][0]
                y = r['pos3d'][1]
                feature = [i, recs[i]['symbol'], recs[i]['name'], x, y, 0, 0, pp2[0], pp2[1]]
                features.append(feature)
    if show:
        outputCsv = output.replace(".png", "_accuracies.csv")
        csvFile = open(outputCsv, 'w', newline='', encoding='utf-8')
        csvWriter = csv.writer(csvFile)
        for f in features:
            csvWriter.writerow(f)

        # send features to the function that correlates between the feature themsrlves
        results = correlate_features(features, 1)
        # get the results and write to a nother CSV file
        outputCsv = output.replace(".png", "_correlations.csv")
        csvFile = open(outputCsv, 'w', newline='', encoding='utf-8')
        csvWriter = csv.writer(csvFile)
        for r in results:
            csvWriter.writerow(r)

        print('Output file: ', outputfile)
        plt.savefig(outputfile, dpi=300)
        plt.show()

    err2 += np.sum(1 - mask) * ransacbound
    if show:
        print('err', err1, err1 / np.sum(mask), err2, err2 / np.sum(mask))
        print('distance2dPoints mean:', np.average(distance2dPoints))
    return err1, err2


# **********
# read data from the features file
# **********
def read_points_data(filename, pixel_x, pixel_y, scale):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        recs = []
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
                names = row
                indx = names.index(pixel_x)
                indy = names.index(pixel_y)
            else:
                line_count += 1

                symbol = row[6]
                pixel = np.array([int(row[indx]), int(row[indy])]) / scale
                height = float(row[5]) + float(row[2])
                pos3d = np.array([float(row[3]), float(row[4]), height])
                name = row[1]
                print(393, pixel)
                rec = {'symbol': symbol,
                       'pixel': pixel,
                       'pos3d': pos3d,
                       'name': name}
                recs.append(rec)
        print(f'Processed {line_count} lines.')
        return recs


# **********
# read data from the potential camera locations file
# **********
def read_camera_locations():
    with open(camera_locations) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        recs = []
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
                names = row
            else:
                line_count += 1
                grid_code = int(row[2])
                height = float(row[5]) + 2.0  # addition of 2 meters  as the observer height 增加2米作为观察者高度
                pos3d = np.array([float(row[3]), float(row[4]), height])
                rec = {'grid_code': grid_code,
                       'pos3d': pos3d}
                recs.append(rec)
        print(f'Processed {line_count} lines.')
        return recs


# **********
# Main function
# **********
def do_it(image_name, features, pixel_x, pixel_y, output, scale):
    im = cv2.imread(image_name)
    im2 = np.copy(im)
    im[:, :, 0] = im2[:, :, 2]
    im[:, :, 1] = im2[:, :, 1]
    im[:, :, 2] = im2[:, :, 0]

    plt.figure(figsize=(11.69, 8.27))  # 40,20
    plt.imshow(im)

    recs = read_points_data(features, pixel_x, pixel_y, scale)
    print(442, recs)
    locations = read_camera_locations()
    pixels = []
    for rec in recs:
        symbol = rec['symbol']
        pixel = rec['pixel']
        if pixel[0] != 0 or pixel[1] != 0:
            plt.text(pixel[0], pixel[1], symbol, color='red', fontsize=38)
        pixels.append(pixel)

    num_matches12 = find_homographies(recs, locations, im, False, 120.0, output)
    num_matches2 = num_matches12[:, 1]
    # print(np.min(num_matches2[num_matches2 > 0]))
    # print(np.max(num_matches2[num_matches2 > 0]))

    num_matches2[num_matches2 == 0] = 1000000
    print(np.min(num_matches2))

    theloci = np.argmin(num_matches2)  # theloci contains the best location for the camera
    print('location id: ' + str(theloci) + ' - ' + str(locations[theloci]))

    find_homographies(recs, [locations[theloci]], im, True, -1, output)  # Orig = 120.0


# img = '5'
# img = '15'
# img = '7'
# img = '13'
# img = '11'
# img = '6'
# img = '4'
# img = '1'
# img = '3'
# img = '0539'
# img = '0518'
# img = 'Henn'
# img = 'Broyn'
img = 'Tirion'
# img = '16'
# img = '17'

camera_locations = ''
grid_code_min = 7

if img == '5':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(23)
    img = cv2.imread('5.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # un-distort
    cv2.imwrite('tmpp5.png', dst)

    image_name = 'tmpp5.png'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_5'
    pixel_y = 'Pixel_y_5'
    output = 'zOutput_5.png'
    scale = 1.0
elif img == '15':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(23)
    img = cv2.imread('15.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # un-distort
    cv2.imwrite('tmpp15.png', dst)

    image_name = 'tmpp15.png'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_15'
    pixel_y = 'Pixel_y_15'
    output = 'zOutput_15.png'
    scale = 1.0

elif img == '7':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(23)
    img = cv2.imread('7.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # un-distort
    cv2.imwrite('tmpp77.png', img)

    image_name = 'tmpp77.png'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_7'
    pixel_y = 'Pixel_y_7'
    output = 'zOutput_77.png'
    scale = 1.0

elif img == '13':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(23)
    img = cv2.imread('13.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # un-distort
    cv2.imwrite('tmpp13.png', dst)

    image_name = 'tmpp13.png'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_13'
    pixel_y = 'Pixel_y_13'
    output = 'zOutput_13.png'
    scale = 1.0

elif img == '11':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(23)
    img = cv2.imread('11.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # un-distort
    cv2.imwrite('tmpp11.png', dst)

    image_name = 'tmpp11.png'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_11'
    pixel_y = 'Pixel_y_11'
    output = 'zOutput_11.png'
    scale = 1.0

elif img == '6':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(23)
    img = cv2.imread('6.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # un-distort
    cv2.imwrite('tmpp6.png', dst)

    image_name = 'tmpp6.png'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_6'
    pixel_y = 'Pixel_y_6'
    output = 'zOutput_6.png'
    scale = 1.0

elif img == '4':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(23)
    img = cv2.imread('4.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # un-distort
    # cv2.imwrite('tmpp4.png', dst)

    image_name = '4.jpg'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_4'
    pixel_y = 'Pixel_y_4'
    output = 'zOutput_4.png'
    scale = 1.0

elif img == '1':
    # ret, mtx, dist, rvecs, tvecs = calibrate_camera(23)
    img = cv2.imread('1.jpg')
    h, w = img.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)   # un-distort
    # cv2.imwrite('tmpp11.png', dst)

    image_name = '1.jpg'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_1'
    pixel_y = 'Pixel_y_1'
    output = 'zOutput_1.png'
    scale = 1.0

elif img == '3':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(23)
    img = cv2.imread('3.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # un-distort
    cv2.imwrite('tmpp3.png', dst)

    image_name = 'tmpp3.png'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_3'
    pixel_y = 'Pixel_y_3'
    output = 'zOutput_3.png'
    scale = 1.0

elif img == '16':
    image_name = '16.jpg'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_16'
    pixel_y = 'Pixel_y_16'
    output = 'zOutput_16.png'
    scale = 1.0

elif img == '17':
    image_name = '17.JPG'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_17'
    pixel_y = 'Pixel_y_17'
    output = 'zOutput_17.png'
    scale = 1.0

elif img == '0539':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(23)
    img = cv2.imread('DSC_0539.tif')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # un-distort
    cv2.imwrite('tmpDSC_0539.png', dst)

    image_name = 'tmpDSC_0539.png'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_DSC_0539'
    pixel_y = 'Pixel_y_DSC_0539'
    output = 'zOutput_DSC_0539.png'
    scale = 1.0

elif img == '0518':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(23)
    img = cv2.imread('DSC_0518.tif')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # un-distort
    cv2.imwrite('tmpDSC_0518.png', dst)

    image_name = 'tmpDSC_0518.png'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_DSC_0518'
    pixel_y = 'Pixel_y_DSC_0518'
    output = 'zOutput_DSC_0518.png'
    scale = 1.0
elif img == 'Henn':
    image_name = 'NNL_Henniker.jpg'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_Henniker'
    pixel_y = 'Pixel_y_Henniker'
    output = 'zOutput_Henniker.png'
    scale = 1.0

elif img == 'Broyn':
    image_name = 'de-broyn-1698.tif'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_Broyin'
    pixel_y = 'Pixel_y_Broyin'
    output = 'zOutput_Broyin.png'
    scale = 1.0
elif img == 'Tirion':
    image_name = 'Tirion-1732.tif'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_Tirion'
    pixel_y = 'Pixel_y_Tirion'
    output = 'zOutput_Tirion.png'
    scale = 1.0
elif img == 'Laboard_b':
    image_name = 'laboard_before.tif'
    features = 'features_tiberias.csv'
    camera_locations = 'potential_camera_locations_tiberias_3D.csv'
    pixel_x = 'Pixel_x_Laboard_b'
    pixel_y = 'Pixel_y_Laboard_b'
    output = 'zOutput_Laboard_b.png'
    scale = 1.0
else:
    print('No file was selected')

do_it(image_name, features, pixel_x, pixel_y, output, scale)

print('**********************')
# print ('ret: ')
# print (ret)
# print ('mtx: ')
# print (mtx)
# print ('dist: ')
# print (dist)
# print('rvecs: ')
# print(rvecs)
# print ('tvecs: ')
# print(tvecs)

print('Done!')





