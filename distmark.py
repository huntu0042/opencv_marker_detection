import cv2
import os
from matplotlib import pyplot as plt
import numpy as np


TAKEOFF_ARR = [0, 1, 0, 1, 0, 1, 0, 1, 0]
LAND_ARR = [1, 0, 1, 0, 1, 0, 1, 0, 1]

DIV_NUM = 3

def show_img(img):
    cv2.imshow("TEST",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#########네모에서 마크 판단하기###############

def dist_img(img): #5*5로 쪼개어 셀 내부 검출
    h = len(img)
    w = len(img[0])
    unit_h = int(h/5)
    unit_w = int(w/5)

    roi = img[unit_h:h-unit_h,unit_w:w-unit_w] #내부 9칸만 뗌
    result_arr = []

    print(roi.shape)

    for i in range(0,DIV_NUM*unit_h,unit_h):
        for j in range(0, DIV_NUM*unit_w, unit_w):
            target = roi[i:i+unit_h,j:j+unit_w]
            result_arr.append(is_white(target))
    print(result_arr)
    dist_arr(result_arr)
    return 0

def is_white(img):
    h = len(img)
    w = len(img[0])
    ret, thr1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    #show_img(thr1)
    count = 0
    for i in range(0,h):
        for j in range(0,w):
            if thr1[i,j][0] == 255 and thr1[i,j][1] == 255 and thr1[i,j][2] == 255:
                count = count+1
    print(100*count/(h*w))
    if h*w/2 > count:
        print("BLACK")
        return 1
    else:
        print("WHITE")
        return 0

def dist_arr(arr):
    if arr == TAKEOFF_ARR:
        print("TAKE OFF MARK DETECT")
        return 1
    if arr == LAND_ARR:
        print("LAND MARK DETECT")
        return 2
    return -1

####################################################################################
#########마커 찾기###############


def find_marker(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    approx_arr = []

    '''
    th1 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 15, 2)
    th2 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 15, 2)
    '''
    ori_img = img
    img= gray_img
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #다양한 필터링을 해봐야 할것으로 보임. 이진 이미지 추출

    contours, hierarchy = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #for cnt in contours:
    #    cv2.drawContours(gray_img, [cnt], 0, (255, 0, 0), 3)  # blue

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 5000:

            approx_arr.append(approx)
            #cv2.drawContours(gray_img, [approx], 0, (0, 255, 255), 5)

    print(str(len(approx_arr)) + " 개의 사각형 검출 ")
    marker_approx =find_approx(approx_arr)
    cv2.drawContours(gray_img, [marker_approx], 0, (0, 255, 255), 5)

    marker_img = affine_marker(ori_img,marker_approx)
    show_img(gray_img)
    show_img(marker_img)
    return marker_img


def find_approx(approx_arr):

    for approx in approx_arr: #16/25 넓이 또는 25/16 넓이가 있는지 찾는다
        area = cv2.contourArea(approx)
        for second_approx in approx_arr:
            second_area = cv2.contourArea(second_approx)
            if second_area > 15/25 * area and second_area < 17/25 * area:
                marker_approx = approx
                return marker_approx

    return -1


def affine_marker(img,approx):

    #pts1 = np.float32(approx)
    #arr = np.float32([[236,297],[248,636],[633,276],[665,587]])
    arr = approx_to_list(approx)
    arr = sort_arr(arr)
    arr = np.float32(arr)

    pts2 = np.float32([[0, 0], [0, 300], [300, 0], [300, 300]])
    M = cv2.getPerspectiveTransform(arr, pts2) #원근법 변환
    #affine = cv2.getAffineTransform(arr,pts2)  #평행성 변환

    dst = cv2.warpPerspective(img, M, (300, 300))

    return dst

def sort_arr(arr):
    avg_x = (arr[0][0] + arr[1][0] + arr[2][0] + arr[3][0]) / 4
    avg_y = (arr[0][1] + arr[1][1] + arr[2][1] + arr[3][1]) / 4
    save_list = [0 for _ in range(4)]


    for point in arr: #조건 제대로 하려면 다시 짜야할듯 . 임시로
        if point[0] < avg_x and point[1] < avg_y:
            save_list[0] = point
        elif point[0] < avg_x and point[1] > avg_y:
            save_list[1] = point
        elif point[0] > avg_x and point[1] < avg_y:
            save_list[2] = point
        elif point[0] > avg_x and point[1] > avg_y:
            save_list[3] = point
    print("정렬결과")
    print(save_list)

    return save_list


def approx_to_list(approx):
    arr = []
    arr.append(approx[0][0].tolist())
    arr.append(approx[1][0].tolist())
    arr.append(approx[2][0].tolist())
    arr.append(approx[3][0].tolist())

    return arr



####################################################################################


natural_img = cv2.imread("img/takeoff_3.jpg")

marker_img = find_marker(natural_img)
dist_img(marker_img)

#img = cv2.imread("img/land_full.png")
