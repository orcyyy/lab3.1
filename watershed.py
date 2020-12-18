
import cv2 as cv
import numpy as np
import os
import random as rng
import time

start_time = time.time()


def cannyWatershed(inputfile):
    img = cv.imread("test.jpg")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    high_thresh, thresh_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    low_thresh = high_thresh * 0.5
    marker = cv.GaussianBlur(gray, (5, 5), 2)
    canny = cv.Canny(marker, low_thresh, high_thresh)
    contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    marker32 = np.int32(marker)
    watershed = cv.watershed(img, marker32)
    m = cv.convertScaleAbs(marker32)
    _, thresh = cv.threshold(m, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    thresh_inv = cv.bitwise_not(thresh)
    temp = cv.bitwise_and(img, img, mask=thresh)
    temp1 = cv.bitwise_and(img, img, mask=thresh_inv)
    result = cv.addWeighted(temp, 1, temp1, 1, 0)
    final = cv.drawContours(result, contours, -1, (0, 0, 255), 1)
    mask = np.zeros(img.shape, dtype=float)
    edgemap = cv.drawContours(mask, contours, -1, (255, 0, 0), 1)

    return edgemap

if __name__ == "__main__":
    print("Сегментация методом водораздела начата")
    edge_map = cannyWatershed("test.jpg")
    cv.imshow('edge map', edge_map)
    cv.imwrite("out_watershed.jpg", edge_map)
    #cv.waitKey(0)

    print("Сегментация методом водораздела закончена")
    print("Затраченное время на выполнения алгоритма собственной реализации: %s seconds ---" % (time.time() - start_time))