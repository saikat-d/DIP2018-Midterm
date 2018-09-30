# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 12:22:06 2018

@author: Saikat
"""


import numpy as np
import cv2
from cv2 import ml
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from os.path import isfile,join

        
DB_path = 'images\\SVM\\train\\'
TEST_path = 'images\\SVM\\test\\'
BOW_path = 'images\\SVM\\bow\\'    

siftdata = []
datalabel = []
for i in range(1,13):
    
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
        
    color_hist = [None] * 12

    ############################
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img2,None)
    for j in des1:
        siftdata.append(j)
        if i<=6:
            datalabel.append(0)
        else:
            datalabel.append(1)
    sift_np = np.float32(siftdata).reshape(-1,128)
    label_np = np.int32(datalabel).reshape(-1,1)
    #print(sift_np.shape)
    #print(label_np.shape)
print(label_np)    
svm = ml.SVM_create()
svm.setType(ml.SVM_C_SVC)
svm.setKernel(ml.SVM_POLY)
svm.setGamma(1.383)
svm.setC(1.67)
svm.setDegree(0.5)
svm.train(sift_np, cv2.ml.ROW_SAMPLE, label_np)

pred_score = [None]*8
for i in range(1,9):
    class1 = 0
    class0 = 0
    #dbimname = str(i) + '.jpg'
    testimname = str(i) + '.jpg'
    #dbimgpath = join(DB_path,dbimname)
    testimgpath = join(TEST_path,testimname)
    if  isfile(testimgpath) :
        img2 = cv2.imread(testimgpath,cv2.IMREAD_COLOR) # queryImage
        #img1 = cv2.imread(testimgpath,cv2.IMREAD_COLOR) # trainImage
        #cv2.cvtColor(img1,cv2.COLOR_BGR2RGB,img1)
        #cv2.cvtColor(img2,cv2.COLOR_BGR2RGB,img2)
    else :
        print("Worng Path : "  + testimgpath)
      
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    for j in des2:
        test_hist_np = np.float32(j).reshape(1,128)
        res = svm.predict(test_hist_np)
        if res[1][0]==1:
            class1 = class1+1
        else:
            class0 = class0+1
    print("class 0: "+str(class0)+ " class 1: "+str(class1))
    if class1>class0:
        pred_score[i-1] = 1
    else:
        pred_score[i-1] = 0
    print("Pred Score : "+ str(pred_score[i-1]))
    #hist_data.append(color_hist)

pred_np = np.float32(pred_score).reshape(1,8)  
true_label = np.float32([0,0,0,0,1,1,1,1])   
#print(pred_np.shape)
fpv, tpv, _ = roc_curve(true_label,pred_np[0,:])
roc_auc = auc(fpv, tpv)
lw =2
plt.plot(fpv, tpv, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Color Histogram')
plt.legend(loc="lower right")
plt.show()

        
        
