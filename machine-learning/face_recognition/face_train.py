import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random

IMG_PATH = '/Users/mike/SPRING2023/CPV301/Assignment_CPV/db_face'

class sift_model:
    def __init__(self):
        self.trainface = {}
        self.testface = {}
 
    def color_equalize(self,img):
        img2 = img.copy()
        blue, green, red = cv2.split(img2)
        img2[:,:,0] = cv2.equalizeHist(blue)
        img2[:,:,1] = cv2.equalizeHist(green)
        img2[:,:,2] = cv2.equalizeHist(red)
        return img2

    def find_angle(self,img):
        im = np.float32(img) / 255.0
        gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
        _, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        return angle

    def first_thresh(self,img):
        
        im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bimg = cv2.threshold(im, 80, 255, cv2.THRESH_BINARY)
        im_new = img.copy()
        im_new[bimg == 0] = 0
        return im_new

    def binary_transform(self,img):
        img2 = img.copy()
        _,bimg = cv2.threshold(img2, 120, 255, cv2.THRESH_BINARY)
        img2[bimg==0] = 0
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        return img2

    def find_sift(self,img):
        img2 = img.copy()
        img2 = img2.astype(np.uint8)
        sift = cv2.SIFT_create() #nfeatures=300
        kp, des = sift.detectAndCompute(img2, None)
        
        return kp, des

    def ransac(self,des1, des2):
        bf = cv2.BFMatcher()
        matches = bf.match(des1, des2)
        # sort them in ascending order of their distances so that best matches (with low distance) come to front
        matchesx = sorted(matches, key=lambda x: x.distance)
        return matchesx

    def show_result(self,img1, img2):
        # first thresh
        im1 = self.first_thresh(img1)
        im21 = self.color_equalize(img2)
        im2 = self.first_thresh(im21)

        # find angle
        angle1 = self.find_angle(im1)
        angle2 = self.find_angle(im2)

        # binary transform
        b_img1 = self.binary_transform(angle1)
        b_img2 = self.binary_transform(angle2)

        # find sift
        kp1, des1 = self.find_sift(b_img1)
        kp2, des2 = self.find_sift(b_img2)
        if des2 is None:
            return None
        else:
        #print(des1.shape, des2.shape)
        # ransac
            results = self.ransac(des1, des2)

            return results

    def find_total(self,lost):
        total = 0
        for x in lost:
            total += x.distance
        log2_total = np.log2(total)
        log10_total = np.log10(total)
        return total, log2_total, log10_total

    def find_total50(self,lost):
        total = 0
        for x in lost[:51]:
            total += x.distance
        log2_total = np.log2(total)
        log10_total = np.log10(total)
        return total, log2_total, log10_total

    def plot_distance(self,dis1, dis2):
        dist1 = []
        dist2 = []
        for x in dis1:
            dist1.append(x.distance)
        for y in dis2:
            dist2.append(y.distance)
        plt.plot(dist1, label = 'dist train')
        plt.plot(dist2, label = 'dist test')

    def plot_distance50(self,dis1, dis2):
        dist1 = []
        dist2 = []
        for x in dis1[:51]:
            dist1.append(x.distance)
        for y in dis2[:51]:
            dist2.append(y.distance)
        plt.plot(dist1, label = 'dist train')
        plt.plot(dist2, label = 'dist test')
        plt.legend()
    


if __name__ == "__main__":
    train = sift_model()
    ground_trues = list()
    base_log2 = list()

    for usr in os.listdir(IMG_PATH):
        list_usr = []
        for file in glob.glob(os.path.join(IMG_PATH, usr)+'/*.jpg'):
            img = cv2.imread(file)
            img = train.color_equalize(img)
            list_usr.append(img)
        random.Random(42).shuffle(list_usr)
        try:
            train.testface[usr] = list_usr[-1]
            train.trainface[usr] = list_usr[:-1]
        except: continue

    facematrix = []
    facelabel = []

    for key, value in train.trainface.items():
        std_meanface = np.mean(np.array(value), axis=0).astype(np.uint8)
        facematrix.append(std_meanface.reshape(-1))
        facelabel.append(key)

    testmatrix = []
    testlabel = []

    for key, value in train.testface.items():
        testmatrix.append(np.array(value).astype(np.uint8))
        testlabel.append(key)

    for i in range(len(facelabel)):
        distance1 = train.show_result(facematrix[i].reshape((160,160,3)), train.trainface[facelabel[i]][0])
        distance2 = train.show_result(facematrix[i].reshape((160,160,3)), testmatrix[i])
        score, log2, log10 = train.find_total50(distance1)
        score_test, log2_test, log10_test = train.find_total50(distance2)
        diff_log2 = np.abs(log2 - log2_test)
        diff_log10 = np.abs(log10-log10_test)
        threshold = np.abs(diff_log2) + np.abs(diff_log10*2)
        # threshold = np.abs(score_test - score)
        base_log2.append(log2)
        ground_trues.append(threshold)

    df = pd.DataFrame(data = facematrix, index = None)
    df.loc[:, 'std_name'] = facelabel
    df.loc[:, 'groundtrues'] = ground_trues
    df.loc[:, 'baselog2'] = base_log2
    df.to_csv("/Users/mike/SPRING2023/CPV301/Assignment_CPV/local_db.csv", index = False)