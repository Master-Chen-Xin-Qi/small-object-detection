from PIL import Image
import numpy as np
import os
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import glob
from skimage import transform,data
list_=list()

# color map
COLOR_Background = (0,0,0)
COLOR_Chair = (0,0,128)
COLOR_Cylinder = (0,128,0)
COLOR_Foot = (0,128,128)
COLOR_Short_Box = (128,0,0)
COLOR_Tall_Box = (128,0,128)
COLOR_BG = (255,0,0)
COLOR_FG = (0,255,0)

# img = cv2.imread("/home/xinqichen/projects/d430_demo/data/ir/00010.png")
# img1 = cv2.imread("/home/xinqichen/00010_json/label.png")
# #print(min(map(min,img1)))
# plt.imshow(img)
# plt.show()
# #read_directory(array_of_depth_img,"home/xinqichen/projects/d430_demo/data/depth")
# #read_directory(array_of_ir_img,"home/xinqichen/projects/d430_demo/data/ir")

# with open("/home/xinqichen/Desktop/out/label.txt", "r") as f:  # 打开文件
#     data = np.float(f.read())  # 读取文件
#     img = Image.fromarray(data)
#     cv2.imshow(img)
#     #plt.show()

# best = 0
# index = -1
best_iou = [0.56,0.24,0.56,0.456,0.89]
# for k in range(len(best_iou)):
#     print("Cross:%d iou:%f" %(k+1, best_iou[k]))
#     if(best_iou[k]>best):
#         best = best_iou[k]
#         index = k+1
# print("Best iou:%f "%best)
# print("Cross:%d" %index)

# a = torch.randn(480,240)
# a.resize_(120,120)
# print(a)

# file_handle=open('1.txt',mode='w')
# for i in range(len(best_iou)):
#     s = str(best_iou[i])
#     file_handle.write(s)
#     file_handle.write('\n')
# file_handle.close()

# a = Image.open('/home/xinqichen/Desktop/small/3.png')
# a = transforms.ToTensor()(a)
# print(a.shape)



path_save = "/home/xinqichen/Desktop/240*320/label_png" #resized pictures are saved here
path = '/home/xinqichen/Desktop/label_solver/afterlabel'#put your pictures here, which you want to resize
label_save_dir = "/home/xinqichen/Desktop/240*320/labels_resize"
pic_save_dir = "/home/xinqichen/Desktop/240*320/pic_resize"
filelist = os.listdir(path)
resize_filelist = os.listdir(path_save)
# for file in filelist:
#     save_name = os.path.join(path_save, file)
#     name = os.path.join(path, file)
#     im = Image.open(name)
#     print(im.format, im.size, im.mode)
#     out = im.resize((320, 240),Image.ANTIALIAS)
#     print(out.format, out.size, out.mode)
#     out.save(save_name,'png')


for file in resize_filelist:
    name = os.path.join(path_save, file)
    cur_img = cv2.imread(name)
    #picture size
    m = len(cur_img)
    n = len(cur_img[0])

    label = [[0 for i in range(n)]for j in range(m)]

    for i in range(m):
        for j in range(n):
            if tuple(cur_img[i][j]) == COLOR_Background:
                label[i][j] = 0
            elif tuple(cur_img[i][j]) == COLOR_Chair or (cur_img[i][j][0]==0 and cur_img[i][j][1]==0 and 5<=cur_img[i][j][2]<=149):
                label[i][j] = 1
            elif tuple(cur_img[i][j]) == COLOR_Cylinder or (cur_img[i][j][0]==0 and 3<cur_img[i][j][1]<=149 and cur_img[i][j][2]==0):
                label[i][j] = 2
            elif tuple(cur_img[i][j]) == COLOR_Foot or (cur_img[i][j][0]==0  and 5<cur_img[i][j][1]<=149 and 5<cur_img[i][j][2]<=149):
                label[i][j] = 3
            elif tuple(cur_img[i][j]) == COLOR_Short_Box or (1<=cur_img[i][j][0]<=149 and cur_img[i][j][1]==0 and cur_img[i][j][2]==0):
                label[i][j] = 4
            elif tuple(cur_img[i][j]) == COLOR_Tall_Box or (9<cur_img[i][j][0]<=149 and cur_img[i][j][1]==0 and 9<cur_img[i][j][2]<=149):
                label[i][j] = 5
    # save labels as .npy
    label = np.array(label)
    label_name = file.replace('.png','')
    save_path = os.path.join(label_save_dir, label_name)
    pic_save_path = os.path.join(pic_save_dir, file)
    np.save(save_path, label)
    im = Image.fromarray(np.uint8(label)*40)
    im.save(pic_save_path)
    print("%s Finish!"%file)


# a = Image.open('/home/xinqichen/Desktop/label_png/549.png')
# b = np.load('/home/xinqichen/Desktop/labels_resize/549.npy')
# plt.subplot(2,1,1)
# plt.imshow(a)
# plt.subplot(2,1,2)
# plt.imshow(b)
# plt.show()
