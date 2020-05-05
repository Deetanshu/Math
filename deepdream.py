# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 11:56:25 2018

@author: deept
"""

import style_utils as su
from PIL import Image
import cv2
style_path = 'Eheeta/10.jpeg'
content_path = 'tmp/nst/Vassily.jpg'

def make_painting_bw(PIL_img=None, img=None):
    if PIL_img is not None:
        img = su.conv_p2cv(PIL_img)
    _,xyz = su.denoise(img = img, multiFilter=True)
    img = su.RGB_G(Image.fromarray(cv2.bilateralFilter(cv2.GaussianBlur(xyz, (3,3), 0), 9, 75, 75)))
    su.imshow(img)
    return img

def make_painting(PIL_img=None, img=None):
    if PIL_img is not None:
        img = su.conv_p2cv(PIL_img)
    _,xyz = su.denoise(img = img, multiFilter=True)
    img = Image.fromarray(cv2.bilateralFilter(cv2.GaussianBlur(xyz, (3,3), 0), 9, 75, 75))
    su.imshow(img)
    return img
    

mix2,_,progress2 = su.run_style_transfer(content_path, style_path, isBW=True)
mix3,_,progress3 = su.run_style_transfer(content_path, style_path, isBW=True)
mix4,_,progress4 = su.run_style_transfer('try.png', style_path, isBW=False)
dn = su.RGB_G(Image.fromarray(bblur))

dn1 = su.denoise(PIL_img = dn, multiFilter=True)
blur = cv2.GaussianBlur(dn1[1],(3,3),0)
bblur = cv2.bilateralFilter(dn,9,75,75)


#dn2 = Image.fromarray(dn1[1])
blimg = Image.fromarray(bblur)
su.imshow(dn)
img = cv2.imread('tried2.jpg')
g_img = cv2.GaussianBlur(img, (3,3), 0)
b_img23 = cv2.bilateralFilter(g_img, 9, 75, 75)
p_img23 = su.RGB_G(Image.fromarray(b_img23))
su.imshow(p_img23)
p_img23.save('final.png')

imgs = make_painting(img = mix2)
imgs = make_painting(img)
img2 = make_painting(cv2.imread('output7.png'))
img4 = make_painting_bw(img = mix4)
xyz =cv2.addWeighted(su.conv_p2cv(img4),1.5,mix4,-0.5,0)
su.imshow(Image.fromarray(xyz))


# FOR FUTURE REFERENCE:
# For starry night, the best outcome was --> denoise(multi) --> guassian -->bilateral --> Convert to gray
# It is now make_painting.