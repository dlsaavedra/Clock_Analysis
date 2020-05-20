import cv2
import matplotlib.pyplot as plt
import numpy as np
from f_clock import read_degrees
import argparse

def _main_(args):

    path_img = args.path_img
    img1 = cv2.imread(path_img)
    degrees, img2 = read_degrees(path_img)
    img2= cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    degrees.sort()
    for d in degrees:  print('Existe una manilla en los ', d , ' grado')

    img1 = cv2.resize(img1, (img2.shape[0], img2.shape[1]))
    vis = np.concatenate((img1, img2), axis=1)
    cv2.imshow("Item_Measure", vis)
    cv2.waitKey()

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='read clock')
    argparser.add_argument('-i', '--path_img', help='path to clock image')

    args = argparser.parse_args()
    _main_(args)
