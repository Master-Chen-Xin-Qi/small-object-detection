import os
import numpy as np
import h5py
import torch
import torch.utils.data as data
import torch
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CreateData_Cross(Dataset):
    def __init__(self, rgb_images_path, depth_images_path, labels_path, i, data_sort):
        self.rgb_images_path = rgb_images_path
        self.depth_images_path = depth_images_path
        self.labels_path = labels_path
        self.k = i
        
        self.rgb_images = self.read_data(self.rgb_images_path, self.k, data_sort)
        self.depth_images = self.read_data(self.depth_images_path, self.k, data_sort)
        self.labels = self.read_data(self.labels_path, self.k, data_sort)
    def __getitem__(self, index):

        #根据index取得相应的一幅图像，一幅标签的路径
        depth_image = self.depth_images[index]
        rgb_image = self.rgb_images[index]
        label = self.labels[index]
        
        #将图片和label读出。
        
        rgb_image = Image.open(rgb_image)
        rgb_image = rgb_image.resize((320, 240),Image.ANTIALIAS)
        rgb_image = transforms.ToTensor()(rgb_image)*255
        depth_image = Image.open(depth_image)
        depth_image = depth_image.resize((320, 240),Image.ANTIALIAS)
        #距离在0.1到5米间的数值翻转
        # depth_image = np.array(depth_img)/1000
        # for i in range(len(depth_image)):
        #     for j in range(len(depth_image[0])):
        #         if 0.1<depth_image[i][j]<5:
        #             depth_image[i][j] = 1/depth_image[i][j] 
        #             print(depth_image[i][j])
        #         else:
        #             depth_image[i][j] = 0

        depth_image = transforms.ToTensor()(depth_image)
        depth_image = depth_image.float()
        label = np.load(label,allow_pickle=True)+1
        label = torch.from_numpy(label)
        label = label.long()


        dataset_list = [rgb_image,depth_image,label]
        return dataset_list

    def read_data(self, data_path, k, data_sort):
        data_list = os.listdir(data_path)
        data_list.sort()
        l = len(data_list)
        cross_len = int(l/5)
        if data_sort == 'test':
            test_path_list = [os.path.join(data_path,data_list[i]) for i in range((k-1)*cross_len, k*cross_len)]
            test_path_list.sort()
            print(len(test_path_list))
            return test_path_list
        if data_sort == 'train':
            train_path_list = [os.path.join(data_path, data_list[i]) for i in range((k-1)*cross_len)]
            train_path_list += [os.path.join(data_path, data_list[i]) for i in range(k*cross_len,l)]
            train_path_list.sort()
            print(len(train_path_list))
            return train_path_list
    def __len__(self):
        return len(self.rgb_images)

def dataset_generate(rgb_images_path, depth_images_path, labels_path, i, data_sort):
    dataset = CreateData_Cross(rgb_images_path, depth_images_path, labels_path, i, data_sort)
    return dataset


def get_cross_data(opt, use_train, use_test, i):
    if os.path.exists(opt.dataroot):
        path = opt.dataroot
    else:
        raise Exception('Wrong datasets requested.')

    rgb_images_path = path + '/rgb'
    depth_images_path = path + '/depth'
    #labels_path = path + '/label/label/labels/SegmentationClass'
    labels_path = path + '/labels_resize'
    
    train_dataset_generator = None
    test_dataset_generator = None

    if use_train:
        train_dataset_generator = dataset_generate(rgb_images_path, depth_images_path, labels_path, i, 'train')
        print('[INFO] Training set generator has been created')
    if use_test:
        test_dataset_generator = dataset_generate(rgb_images_path, depth_images_path, labels_path, i, 'test')
        print('[INFO] Test set generator has been created')
    return train_dataset_generator, test_dataset_generator
