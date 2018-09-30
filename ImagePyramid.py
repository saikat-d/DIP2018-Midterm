# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 20:07:42 2018

@author: Saikat
"""

import numpy as np
import cv2
from cv2 import ml
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from os.path import isfile,join

        
DB_path = 'images\\Pyramid\\'
RES_path = 'images\\Pyramid\\res\\'
BOW_path = 'images\\SVM\\bow\\'    

siftdata = []
datalabel = []
for i in range(2,5):
    
    dbimname = str(i) + '.jpg'
    #testimname = str(i) + '.jpg'
    dbimgpath = join(DB_path,dbimname)
    #testimgpath = join(TEST_path,testimname)
    if  isfile(dbimgpath) :
        img2 = cv2.imread(dbimgpath,cv2.IMREAD_COLOR) # queryImage
        #img1 = cv2.imread(testimgpath,cv2.IMREAD_COLOR) # trainImage
        #cv2.cvtColor(img1,cv2.COLOR_BGR2RGB,img1)
        #cv2.cvtColor(img2,cv2.COLOR_BGR2RGB,img2)
    else :
        print("Worng Path : "  + dbimgpath)
    imgg = img2
    imgl = img2
    for j in range(0,5):    
        
        imgg = cv2.GaussianBlur(imgg,(5,5),0)
        imggt = np.uint8(imgg)
        ret2,th2 = cv2.threshold(imggt[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imwrite(join(RES_path,"gaussian\\"+str(i)+"_"+str(j)+"_t.jpg"),th2)
        cv2.imwrite(join(RES_path,"gaussian\\"+str(i)+"_"+str(j)+".jpg"),imgg)
        
        imgl = cv2.Laplacian(imgl,cv2.CV_32F,ksize=5)
        cv2.imwrite(join(RES_path,"laplacian\\"+str(i)+"_"+str(j)+".jpg"),imgl)
        imgl_ = np.array(imgl)  
        imgl_ *= 255.0/imgl_.max()  
        #cv2.imwrite(join(RES_path,"laplacian\\"+str(i)+"_"+str(j)+"_c.jpg"),imgl_)
        imglt = np.uint8(imgl_)
        ret1,th1 = cv2.threshold(imglt[:,:,1],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imwrite(join(RES_path,"laplacian\\"+str(i)+"_"+str(j)+"_t.jpg"),th1)
        
        
    
    
    
    
    