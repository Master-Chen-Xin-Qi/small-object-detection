import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import copy

# color map
COLOR_Background = (0,0,0)
COLOR_Chair = (0,0,128)
COLOR_Cylinder = (0,128,0)
COLOR_Foot = (0,128,128)
COLOR_Short_Box = (128,0,0)
COLOR_Tall_Box = (128,0,128)
COLOR_BG = (255,0,0)
COLOR_FG = (0,255,0)
global points_total 
global points 
global color_total 
global color
points_total = []
points = []
color_total = []
color = (0,0,0)# original color is black

def nothing(x):
    pass

# Everytime counters a mouse movement call this
def draw(e, x, y, flags, param):
    drag = False
    if e == cv2.EVENT_LBUTTONDOWN:
        # print("123")
        drag = True
    if drag == True :
        # print("456")
        points.append((x,y))
        print('point:',len(points))
        if(len(points)>=2):
            for i in range(len(points)-1):
                cv2.line(cur_img, points[i], points[i+1], color, 3, 8, 0)
    if e == cv2.EVENT_LBUTTONUP:
        drag = False
    if e == cv2.EVENT_MOUSEWHEEL:
        # print(points)
        if len(points)!=0:
            tmp = copy.deepcopy(points)
            points_total.append(tmp)
            cv2.fillConvexPoly(cur_img, np.array(tmp), color)
            #print(points_total)
            print("fill finish,select another area")
        points.clear()
    if e == cv2.EVENT_RBUTTONDOWN:
        if len(points)!=0:
            print("line finish,select another area")
            points.clear()

rgb_dir = "/home/xinqichen/Desktop/label_solver/rgb_forlabel"
img_dir = "/home/xinqichen/Desktop/label_solver/img_forlabel"
label_save_dir = "/home/xinqichen/Desktop/label_solver/label_npy"
pic_save_dir = "/home/xinqichen/Desktop/label_solver/afterlabel"

imglist = sorted([x for x in os.listdir(img_dir) if '.png' in x or '.jpg' in x])
# for i in imglist:
#     print(i)
rgblist = sorted([x for x in os.listdir(rgb_dir) if '.png' in x or '.jpg' in x])
print("================= Label Solver =================")
print("Press left mouse to draw points")
print("Press rifht mouse to finish drawing line")
print("Press b to draw background(You can use black color to erase your prior operation as well)")
print("Press c to draw chair")
print("Press y to draw cylinder")
print("Press f to draw foot")
print("Press s to draw short_box")
print("Press t to draw tall_box")
print("Press n to jump to next picture")
print("Press m to save current picture and labels")
print("Press Esc to terminate this program")
print("Move your mousewheel to start a new area")


for idx in range(len(imglist)):
    img = imglist[idx]
    rgb = rgblist[idx]
    cur_rgb = cv2.imread(os.path.join(rgb_dir,rgb))
    cur_img = cv2.imread(os.path.join(img_dir,img))
    while(1):
        flag = 0
        cv2.namedWindow(img)
        cv2.setMouseCallback(img,draw)
        merge = cur_img+cur_rgb
        imghstack = np.hstack((merge, cur_rgb, cur_img))
        cv2.imshow(img,imghstack)
        key = cv2.waitKey(100)
        if key == 27:
            print('You have already quit')
            exit()
        if key == ord('n'):
            print('Next picture')
            break
        if key == ord('p'):
            flag = 1
            print('Save current picture')
            break
        if key == ord('b'):
            color = COLOR_Background
            print("Background Mode")
            #color_total.append(color)
        elif key == ord('c'):
            color = COLOR_Chair
            print("Chair Mode")
            #color_total.append(color)
        elif key == ord('y'):
            color = COLOR_Cylinder
            print("Cylinder Mode")
            #color_total.append(color)
        elif key == ord('f'):
            color = COLOR_Foot
            print("Foot Mode")
            #color_total.append(color)
        elif key == ord('s'):
            color = COLOR_Short_Box
            print("Short_Box Mode")
            #color_total.append(color)
        elif key == ord('t'):
            color = COLOR_Tall_Box
            print("Tall_Box Mode")
            #color_total.append(color)
    cv2.destroyAllWindows()
    if flag == 1:
        print(points_total)
        # for i in range(len(points_total)):
        #     points = points_total[i]
        #     points = np.array(points)
        #     cv2.fillConvexPoly(cur_img, points, color_total[i])
        #     points = list()

        pic_save_path = pic_save_dir+'/'+img
        cv2.imwrite(pic_save_path, cur_img)

        #picture size
        m = len(cur_img)
        n = len(cur_img[0])

        label = [[0 for i in range(n)]for j in range(m)]

        for i in range(m):
            for j in range(n):
                if tuple(cur_img[i][j]) == COLOR_Background:
                    label[i][j] = 0
                elif tuple(cur_img[i][j]) == COLOR_Chair:
                    label[i][j] = 1
                elif tuple(cur_img[i][j]) == COLOR_Cylinder:
                    label[i][j] = 2
                elif tuple(cur_img[i][j]) == COLOR_Foot:
                    label[i][j] = 3
                elif tuple(cur_img[i][j]) == COLOR_Short_Box:
                    label[i][j] = 4
                elif tuple(cur_img[i][j]) == COLOR_Tall_Box:
                    label[i][j] = 5
        # save labels as .npy
        label = np.array(label)
        label_name = img.replace('.png','')
        save_path = os.path.join(label_save_dir, label_name)
        np.save(save_path, label)
        print("%s Finish!"%img)
