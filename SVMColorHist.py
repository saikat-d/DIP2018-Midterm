# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 09:34:30 2018

@author: Saikat
"""


import numpy as np
import cv2
from cv2 import ml
from matplotlib import pyplot as plt
from os.path import isfile,join
from sklearn.metrics import roc_curve, auc

class choices:
    def _init_(self):
        self.imgname = ' '
        self.match = np.zeros(1,1)
        self.count = 0
        


        
DB_path = 'images\\SVM\\train\\'
TEST_path = 'images\\SVM\\test\\'    

hist_data = []
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
    
    r_histogram = cv2.calcHist(img2,[0],None,[8],[0,256])
    g_histogram = cv2.calcHist(img2,[1],None,[8],[0,256])
    b_histogram = cv2.calcHist(img2,[2],None,[8],[0,256])
    
    color_hist = [None] * 24
    for j in range(0,8):
        color_hist[j] = r_histogram[j]
        color_hist[8+j] = g_histogram[j]
        color_hist[16+j] = b_histogram[j]
    
    hist_data.append(color_hist)
    
labels = [0,0,0,0,0,0,1,1,1,1,1,1]    
hist_np = np.float32(hist_data).reshape(-1,24)
label_np = np.int32(labels).reshape(-1,1)
print(hist_np.shape)
print(label_np.shape)

svm_params = dict( kernel_type = ml.SVM_LINEAR,
                    svm_type = ml.SVM_C_SVC,
                    C=2.67, gamma=5.383 )
svm = ml.SVM_create()
svm.setType(ml.SVM_C_SVC)
svm.setKernel(ml.SVM_POLY)
svm.setGamma(5.383)
svm.setC(2.67)
svm.setDegree(1)
svm.train(hist_np, cv2.ml.ROW_SAMPLE, label_np)

pred_score = [None]*8
for i in range(1,9):
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
        
    r_histogram = cv2.calcHist(img2,[0],None,[8],[0,256])
    g_histogram = cv2.calcHist(img2,[1],None,[8],[0,256])
    b_histogram = cv2.calcHist(img2,[2],None,[8],[0,256])
    
    test_hist = [None] * 24
    for j in range(0,8):
        test_hist[j] = r_histogram[j]
        test_hist[8+j] = g_histogram[j]
        test_hist[16+j] = b_histogram[j]
    
    #hist_data.append(color_hist)
    
    test_hist_np = np.float32(test_hist).reshape(1,24)      
    res = svm.predict(test_hist_np)
    pred_score[i-1] = res
    print(res)
    
#print(pred_score)
pred_np = np.float32(pred_score).reshape(2,8)  
true_label = np.float32([0,0,0,0,1,1,1,1])   
#print(pred_np.shape)
fpv, tpv, _ = roc_curve(true_label,pred_np[1,:])
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
    
    
        
        
        
