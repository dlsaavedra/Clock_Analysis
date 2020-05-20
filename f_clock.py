import cv2
import matplotlib.pyplot as plt
import numpy as np




def detect_cicles(path_img, ideal_size = 300, output_size = 200):

    #### Read and rescale image ####
    o_image= cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB)
    new_size = np.round(np.array(o_image.shape)[:2] * 1/(min(np.array(o_image.shape)[:2]) / 300))
    new_size = new_size.astype(int)
    #print(new_size)
    o_image = cv2.resize(o_image, (new_size[1], new_size[0]))

    # Convert to grayscale.
    gray = cv2.cvtColor(o_image, cv2.COLOR_RGB2HSV)[:,:,2]

    min_size = min(gray.shape)
    max_size = max(gray.shape)
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (5, 5))

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
                       cv2.HOUGH_GRADIENT, dp = 1, minDist = max_size/2, param1 = 200,
                   param2 = 30, minRadius = int(min_size/10), maxRadius = int(min_size/2))

    ## Double circle detection
    cx,cy,r  = np.uint16(detected_circles[0][0])  # utilizar solo un circulo detectado
    delta = 5
    r +=delta

    gray_blurred = gray_blurred[cy-r:cy+r,cx-r:cx+r]
    image_crop = o_image[cy-r:cy+r,cx-r:cx+r].copy()

    min_size = min(gray_blurred.shape)
    max_size = max(gray_blurred.shape)

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
                   cv2.HOUGH_GRADIENT, dp = 1, minDist = max_size/2, param1 = 200,
                   param2 = 30, minRadius = int(min_size/10), maxRadius = int(min_size/2))


    cx2,cy2,r2  = np.uint16(detected_circles[0][0])  # utilizar solo un circulo detectado
    delta = -5
    r2 +=delta

    image_center =image_crop[cy2 - r2: cy2 + r2, cx2-r2 : cx2 + r2].copy()

    for i in range(image_center.shape[0]):
        for j in range(image_center.shape[1]):
            if ((r2-i) ** 2 + (r2-j) ** 2) > r2**2 - delta*2 :
                image_center[i][j] = 255


    return cv2.resize(image_center, (output_size, output_size))


def image2polar(image_center, new_size = 200, border = 50):

    center = np.array(image_center.shape[:2])/2
    maxRadius = min(np.array(image_center.shape[:2])/2)

    polar_image = cv2.linearPolar(image_center, (center[0], center[1]), maxRadius, cv2.WARP_FILL_OUTLIERS)
    polar_image = cv2.resize(polar_image, (new_size, 360))
    polar_image = polar_image.astype(np.uint8)
    polar_image = polar_image[:, border:]

    return polar_image

def segmentate_clock(polar_image, otsu = False):

    ### Use V Chanel of HSV (detect only black with backgorund white)
    new_size = polar_image.shape[1]
    gray_image = cv2.cvtColor(polar_image, cv2.COLOR_RGB2HSV)[:,:,2]
    kernel = np.ones((1, int(new_size/3)),np.uint8)
    ## filter for segmentated lines  ##
    dilate = cv2.dilate(gray_image, kernel, iterations = 1)
    erosion = cv2.erode(dilate, kernel, iterations = 1)

    if otsu:
        ret,th = cv2.threshold(erosion,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    else:
        umbral = 240
        ret,th = cv2.threshold(erosion,umbral,255,cv2.THRESH_BINARY_INV)

        while (np.sum(th == 255)/th.size) * 100 > 5: # porcentaje de area de las manillas
            umbral /=1.3
            ret,th = cv2.threshold(erosion,umbral,255,cv2.THRESH_BINARY_INV)
            #print("Umbral fijo1",ret)

    return cv2.dilate(th, np.ones((3,3), np.uint8), iterations = 1)

def filter_segmetate(image_th, th_min_area = 10):

    epsilon = 0.0001

    ### Agrupate clock hands ###
    num_labels, labels_im = cv2.connectedComponents(image_th)
    labels_im -= 1
    contours, hierarchy = cv2.findContours(image_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    center_dict = {}
    for n, c in enumerate(contours):
       M = cv2.moments(c)
       cX = int(M["m10"] / (M["m00"] + epsilon))
       cY = int(M["m01"] / (M["m00"] + epsilon))
       max_X = max([a[0][0] for a in c])
       min_X = min([a[0][0] for a in c])
       if min_X == 0 and M["m00"] > th_min_area:
           center_dict[n] = (cX, cY, max_X, M["m00"]) # Centroides, distancia máxima de la barra y tamaño de la barra

    ### Join result clock hands ####

    center_dict_filter = {}
    n = 0
    for k in center_dict:

        cX,cY, max_X, masa = center_dict[k]
        for j in center_dict:

            if  k == j:
                continue
            #print(k,j)
            if np.abs(cY-center_dict[j][1])< 200 and np.abs(cY-center_dict[j][1]) > 160:
                if max_X > center_dict[j][2]:
                    center_dict_filter[n] = center_dict[k]
                    n +=1

    if center_dict_filter == {}: ## la barra no cruza el centro del reloj
        center_dict_filter = center_dict

    ### Return polar grade of clock hands, the zero grade is in the 12 o'clock.

    return center_dict_filter # dictionary with clock hands {cx, cy, Max_X, Area}

    #

    #for k in center_dict_filter:
    #    print('Existe una manilla en los ', (center_dict_filter[k][1] + 90) % 360, ' grado')

def degrees(path_img, ideal_size = 300, clock_size = 200, border = 50, otsu = False):

    ### Circle Detect ###
    image_center = detect_cicles(path_img, ideal_size, clock_size)

    ### Transform to polar image ###
    polar_image = image2polar(image_center, clock_size, border)

    ### Segmentate  clock hands ####
    th = segmentate_clock(polar_image,  otsu)

    ###  degrees of clock hands ####
    result_degrees = filter_segmetate(th)

    return result_degrees, image_center

def read_degrees(path_img, ideal_size = 300, clock_size = 200, border = 50, otsu = False):


    center_dict_filter, image_center = degrees(path_img, ideal_size, clock_size, border, otsu)
    ### List with grade of each clock hands detected
    return [(center_dict_filter[k][1] + 90) % 360 for k in center_dict_filter], image_center

def read_clock(path_img, ideal_size = 300, clock_size = 200, border = 50, otsu = False):

    center_dict_filter, image_center = degrees(path_img, ideal_size, clock_size, border, otsu)

    if len(center_dict_filter) == 3:
        aux_dict = center_dict_filter.copy()
        key_seg = list(center_dict_filter.keys())[np.argmin([center_dict_filter[k][3] for k in center_dict_filter])]
        del aux_dict[key_seg]
        key_min = list(aux_dict.keys())[np.argmax([aux_dict[k][2] for k in aux_dict])]
        del aux_dict[key_min]
        key_hr = list(aux_dict.keys())[0]
        seg = round (((center_dict_filter[key_seg][1] + 90) % 360)/360 * 60) % 60
        minu = round(((center_dict_filter[key_min][1] + 90) % 360)/360 * 60) % 60
        hr = round(((center_dict_filter[key_hr][1] + 90) % 360)/360 * 12) % 12
        return "La hora es: " + str(hr)  + ":" +  str(minu) + ":" + str(seg), image_center

    elif len(center_dict_filter) == 2:
        aux_dict = center_dict_filter.copy()
        key_min = list(aux_dict.keys())[np.argmax([aux_dict[k][2] for k in aux_dict])]
        del aux_dict[key_min]
        key_hr = list(aux_dict.keys())[0]
        minu = round(((center_dict_filter[key_min][1] + 90) % 360)/360 * 60) % 60
        hr = round(((center_dict_filter[key_hr][1] + 90) % 360)/360 * 12) % 12
        if hr == 0: hr = 12
        return "La hora es: " + str(hr) + ":" + str(minu), image_center
