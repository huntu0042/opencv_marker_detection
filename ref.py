# -*- coding:utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy as np
import scipy.io as sio
import scipy.misc
from scipy.misc import imresize
import cv2
import json

FINAL_HEIGHT = 256
FINAL_WIDTH = 192
count = 0

###이미지 관련 ##
def _load_image(img_name_dir):
    img = cv2.imread(img_name_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


###마스크 관련 ##
def process_raw_mask(mask): #stage1 마스크 처리
    #gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, raw = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    return raw

def otsu_external_contour(image): #stage2 마스크를 하얗게
    print("image.shape:" + str(image.shape))
    otsu_thr, otsu_mask = cv2.threshold(image, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print('Estimated threshold (Otsu): ', otsu_thr)
    _, contours, hierarchy = cv2.findContours(otsu_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    image_external = np.zeros(image.shape, image.dtype)
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(image_external, contours, i, 255, -1)
    return image_external

def make_final_mask(mask_one,mask_two,segment):
    segment = process_raw_mask(segment)
    #show_img(segment)

    final_mask = cv2.add(segment, mask_one)
    final_mask = process_raw_mask(final_mask)
    final_mask = cv2.add(final_mask, mask_two)
    final_mask = process_raw_mask(final_mask)

    return final_mask


def _process_ratio(h, w, composition_mask):
    # height_ratio = h / composition_raw.shape[0]
    # width_ratio = w / composition_raw.shape[1]
    # update_h = int(h * height_ratio)
    # update_w = int(w * width_ratio)

    # resized_composition_raw = cv2.resize(composition_raw, (w, h), interpolation = cv2.INTER_CUBIC)
    resized_composition_mask = cv2.resize(composition_mask, (w, h), interpolation=cv2.INTER_CUBIC)
    return resized_composition_mask

##합성 처리##

def make_fg(comp_img,mask):
    comp_img = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)
    fg = cv2.bitwise_and(comp_img, comp_img, mask=mask)
    return fg

def make_bg(ori_img,mask): #1대1의 original image에 입힘

    len_img = len(ori_img)
    len_widht = len(ori_img[0])
    alpha = int((len_widht-FINAL_WIDTH)/2)
    roi = ori_img[:FINAL_HEIGHT, alpha:alpha + FINAL_WIDTH] #씌워질 영역
    final_mask_reverse = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(roi, roi, mask=final_mask_reverse)
    return bg

def make_bg_pants(ori_img,mask): #1대1의 original image에 입힘

    len_img = len(ori_img)
    len_width = len(ori_img[0])
    real_width = len(mask[1])
    alpha = int((len_width-real_width)/2)


    roi = ori_img[:, alpha:alpha + real_width]
    final_mask_reverse = cv2.bitwise_not(mask)
    print(alpha)
    print(ori_img.shape)
    print(final_mask_reverse.shape)
    print(roi.shape)
    bg = cv2.bitwise_and(roi, roi, mask=final_mask_reverse)
    return bg

def fg_plus_bg(fg,bg):

    coarse_image = cv2.add(fg, bg)
    return coarse_image

def bg_to_original(ori_img,bg): #잘린 bg 이미지를 다시 정상크기 이미지로 붙인다.
    height = len(bg)
    width = len(bg[0])

    len_img = len(ori_img)
    len_widht = len(ori_img[0])
    alpha = int((len_widht - width) / 2)

    ori_img[:height, alpha:alpha + width] = bg
    return ori_img




###출력##

def show_img(img):
    global count
    count = count+1

    cv2.imshow(str(count),img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''

USER_DIR = '../userdata/'
USER_ID = 'test'
IMAGE_ID = 'image'

DATA_LIST_DIR = USER_DIR + USER_ID + '/Input/image.txt'
ORI_IMG_DIR = USER_DIR + USER_ID + "/Input/original_images/"
ORI_SEG_DIR = USER_DIR + USER_ID + "/Input/original_segment/"
STD_IMG_DIR = USER_DIR + USER_ID + "/Input/standard_images/"
STD_SEG_DIR = USER_DIR + USER_ID + "/Input/standard_segment/"
BODY_IMG_DIR = USER_DIR + USER_ID + "/Input/body_image/"
BODY_SEG_DIR = USER_DIR + USER_ID + "/Input/body_segment/"
UPPER_IMG_DIR = USER_DIR + USER_ID + "/Input/upper_images/"
UPPER_SEG_DIR = USER_DIR + USER_ID + "/Input/upper_segment/"

RESULT_STG_DIR = USER_DIR + USER_ID + "/stage/"
RESULT_COMP_DIR = USER_DIR + USER_ID + "/Output/composed_images/"

INPUT_DIR = USER_DIR + USER_ID + "/Input"
OUTPUT_DIR = USER_DIR + USER_ID + "/Output"


image_id = "000001"
upper_id = '102001'
lower_id = '000000'
is_upper = 1
##input 받아야 할 것들

image_name = image_id + '_0'

upper_name = upper_id + "_1"
lower_name = lower_id + "_1"
upper_composition_name = image_name + "_" + upper_name + "_"
lower_composition_name = image_name + "_" + lower_name + "_"
full_composition_name = image_name + "_" + upper_name + "_" + lower_name + "_"

middle_composition_name = ""
if(is_upper == 0):
  print("Lower")
  middle_composition_name = lower_composition_name
  product_image_name = lower_id + '_1'
elif(is_upper == 1):
  print("Upper")
  middle_composition_name = upper_composition_name
  product_image_name = upper_id + '_1'


comp_img = _load_image(RESULT_COMP_DIR + middle_composition_name + "final.png")
#합성된 이미지


mask_comp_one = cv2.imread(RESULT_STG_DIR + middle_composition_name + "mask.png", 0)
mask_comp_two = cv2.imread(RESULT_COMP_DIR + middle_composition_name + "sel_mask.png", 0)
#show_img(mask_comp_one)
print(mask_comp_one.shape)

composition_mask1 = process_raw_mask(mask_comp_one)
composition_mask2 = otsu_external_contour(mask_comp_two)

segment = cv2.imread(UPPER_SEG_DIR + image_id + ".png", 0)
segment = cv2.resize(segment, (192, 256), interpolation=cv2.INTER_NEAREST)

final_mask = make_final_mask(composition_mask1,composition_mask2,segment)

#3개를 합친 마스크

#lowerid로 바지가 있는지 체크해야함



fg = make_fg(comp_img,final_mask)
#마스크로 FOREGROUND를 만들어야함, 저장해야함


model_img = cv2.imread(STD_IMG_DIR + image_id + ".png")

bg = make_bg(model_img,final_mask)
plus_img = fg_plus_bg(fg,bg)
show_img(bg_to_original(model_img,plus_img))
#전신 사진에 합성


'''