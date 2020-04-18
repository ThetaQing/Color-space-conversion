# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 15:34:02 2020

@author: Administrator
"""

import cv2
import numpy as np

img = cv2.imread(filename="../img/1.png",flags=True)

img_red = np.zeros(img.shape, np.uint8)
img_YUV = np.zeros(img.shape,np.uint8)
height = img.shape[0]
width = img.shape[1]

print ("img.shape[1]:%d,img.shape[0]:%d" %(img.shape[1],img.shape[0]))

HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
YCrCb_img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
for i in range(height):
        for j in range(width):
            (b,g,r) =  img[i,j]
            y = 0.299*r+0.587*g+0.114*b
            u = -0.147*r-0.289*g+0.436*b
            v = 0.615*r-0.515*g-0.1*b
            img_YUV[i][j] = (y+1.14*v, y-0.39*u-0.58*v, y+2.03*u)
            img_red[i,j] = (255,255,255)
            if (r>b and r>g):               #我们假设(b,g,r)中r最大时为红色，当然也可设阈值
                img_red[i,j] = (b,g,r)      #cv2中读取RGB通道的顺序是b-g-r      
                
cv2.imwrite('bp_red.jpg', img_red)
cv2.imwrite("HSV_img.jpg",HSV_img)
cv2.imshow('image',img)
cv2.imshow('YCrCb_img',YCrCb_img)
cv2.imshow("HSV_img", HSV_img)
cv2.imshow("HUV_img",img_YUV)
cv2.waitKey(0)


