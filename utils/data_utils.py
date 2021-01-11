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


# class CreateData(data.Dataset):
#     def __init__(self, dataset_dict):
#         self.len_dset_dict = len(dataset_dict)
#         self.rgb = dataset_dict['rgb']
#         self.depth = dataset_dict['depth']
#         self.seg_label = dataset_dict['seg_label']

#         if self.len_dset_dict > 3:
#             self.class_label = dataset_dict['class_label']
#             self.use_class = True

#     def __getitem__(self, index):
#         rgb_img = self.rgb[index]
#         depth_img = self.depth[index]
#         seg_label = self.seg_label[index]

#         rgb_img = torch.from_numpy(rgb_img)
#         depth_img = torch.from_numpy(depth_img)

#         dataset_list = [rgb_img, depth_img, seg_label]

#         if self.len_dset_dict > 3:
#             class_label = self.class_label[index]
#             dataset_list.append(class_label)
#         return dataset_list

#     def __len__(self):
#         return len(self.seg_label)


# def get_data(opt, use_train=True, use_test=True):
#     """
#     Load NYU_v2 or SUN rgb-d dataset in hdf5 format from disk and prepare
#     it for classifiers.
#     """
#     # Load the chosen datasets path
#     if os.path.exists(opt.dataroot):
#         path = opt.dataroot
#     else:
#         raise Exception('Wrong datasets requested. Please choose either "NYU" or "SUN"')
    
#     h5file = h5py.File(path, 'r')

#     train_dataset_generator = None
#     test_dataset_generator = None

#     # Create python dicts containing numpy arrays of training samples
#     if use_train:
#         train_dataset_generator = dataset_generator(h5file, 'train', opt.use_class)
#         print('[INFO] Training set generator has been created')

#     # Create python dicts containing numpy arrays of test samples
#     if use_test:
#         test_dataset_generator = dataset_generator(h5file, 'test', opt.use_class)
#         print('[INFO] Test set generator has been created')
#     h5file.close()
#     return train_dataset_generator, test_dataset_generator


# def dataset_generator(h5file, dset_type, use_class):
#     """
#     Move h5 dictionary contents to python dict as numpy arrays and create dataset generator
#     """
#     dataset_dict = dict()
#     # Create numpy arrays of given samples
#     dataset_dict['rgb'] = np.array(h5file['rgb_' + dset_type],  dtype=np.float32)
#     dataset_dict['depth'] = np.array(h5file['depth_' + dset_type], dtype=np.float32)
#     dataset_dict['seg_label'] = np.array(h5file['label_' + dset_type], dtype=np.int64)

#     # If classification loss is included in training add the classification labels to the dataset as well
#     if use_class:
#         dataset_dict['class_label'] = np.array(h5file['class_' + dset_type], dtype=np.int64)
#     return CreateData(dataset_dict)


class CreateData(Dataset):
    def __init__(self , rgb_images_path , depth_images_path , labels_path):
        
        # 所有图片和标签的路径
        # depth_images_path_list 
        # rgb_images_path_list 
        # labels_path_list 
        
        # 所有图片和标签
        # depth_images
        # rgb_images
        # labels

        self.depth_images_path_list = depth_images_path
        self.rgb_images_path_list = rgb_images_path
        self.labels_path_list = labels_path

        self.depth_images = self.read_file(self.depth_images_path_list)
        self.rgb_images = self.read_file(self.rgb_images_path_list)
        self.labels = self.read_file(self.labels_path_list)

       
    def __getitem__(self,index):
    
        #根据index取得相应的一幅图像，一幅标签的路径
        depth_image = self.depth_images[index]
        rgb_image = self.rgb_images[index]
        label = self.labels[index]
        
        #将图片和label读出。
        
        rgb_image = Image.open(rgb_image)
        rgb_image = rgb_image.resize((320, 240),Image.ANTIALIAS)
        rgb_image = transforms.ToTensor()(rgb_image)*255
        #rgb_image = rgb_image.float()
        depth_image = Image.open(depth_image)
        depth_image = depth_image.resize((320, 240),Image.ANTIALIAS)
        depth_image = transforms.ToTensor()(depth_image)
        depth_image = depth_image.float()
        label = np.load(label,allow_pickle=True)+1
        label = torch.from_numpy(label)
        label = label.long()
        #label.resize(1, 240,320)
        dataset_list = [rgb_image,depth_image,label]
        return dataset_list

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        #print(len(file_path_list))
        return file_path_list
        
    def __len__(self):
        return len(self.rgb_images)

def get_data(opt, use_train=True, use_test=True):
    """
    Load self-made dat from disk and prepare it for classifiers.
    """
    # Load the chosen datasets path
    if os.path.exists(opt.dataroot):
        path = opt.dataroot
    else:
        raise Exception('Wrong datasets requested.')
    
    train_rgb_images_path = path + '/rgb'
    train_depth_images_path = path + '/depth'
    train_labels_path = path + '/label/label/labels/SegmentationClass'

    # test_rgb_images_path = path + '/test_rgb'
    # test_depth_images_path = path + '/test_depth'
    # test_labels_path = path + '/test_label/SegmentationClass'

    test_rgb_images_path = path + '/rgb'
    test_depth_images_path = path + '/depth'
    test_labels_path = path + '/labels_resize'
    #test_labels_path = path + '/label/label/labels/SegmentationClass'

    train_dataset_generator = None
    test_dataset_generator = None

    # Create python dicts containing numpy arrays of training samples
    if use_train:
        train_dataset_generator = dataset_generator(train_rgb_images_path, train_depth_images_path, train_labels_path)
        print('[INFO] Training set generator has been created')

    # Create python dicts containing numpy arrays of test samples
    if use_test:
        test_dataset_generator = dataset_generator(test_rgb_images_path, test_depth_images_path, test_labels_path)
        print('[INFO] Test set generator has been created')
    return train_dataset_generator, test_dataset_generator


class CreateData_nolabel(Dataset):
    def __init__(self , rgb_images_path , depth_images_path):
        
        # 所有图片的路径
        # depth_images_path_list 
        # rgb_images_path_list 
        
        # 所有图片
        # depth_images
        # rgb_images

        self.depth_images_path_list = depth_images_path
        self.rgb_images_path_list = rgb_images_path

        self.depth_images = self.read_file(self.depth_images_path_list)
        self.rgb_images = self.read_file(self.rgb_images_path_list)
       
    def __getitem__(self,index):
    
        #根据index取得相应的一幅图像
        depth_image = self.depth_images[index]
        rgb_image = self.rgb_images[index]
        
        #将图片读出。“L”表示灰度图，也可以填“RGB”
        
        rgb_image = Image.open(rgb_image)
        rgb_image = transforms.ToTensor()(rgb_image)*255
        #rgb_image = rgb_image.float()
        depth_image = Image.open(depth_image)
        depth_image = transforms.ToTensor()(depth_image)*100
        depth_image = depth_image.float()
        dataset_list = [rgb_image,depth_image]
        return dataset_list

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img.replace('depth','')) for img in files_list]
        file_path_list.sort()
        #print(len(file_path_list))
        return file_path_list
        
    def __len__(self):
        return len(self.rgb_images)

# without labels,just for data generation 
def get_data_nolabel(opt, use_train=True, use_test=True):
    """
    Load self-made dat from disk and prepare it for classifiers.
    """
    # Load the chosen datasets path
    if os.path.exists(opt.dataroot):
        path = opt.dataroot
    else:
        raise Exception('Wrong datasets requested.')
    
    train_rgb_images_path = path + '/rgb'
    train_depth_images_path = path + '/depth'
    train_labels_path = path + '/label/label/labels/SegmentationClass'

    test_rgb_images_path = path + '/test_rgb'
    test_depth_images_path = path + '/test_depth'
    #test_labels_path = path + '/test_label/SegmentationClass'


    train_dataset_generator = None
    test_dataset_generator = None

    # Create python dicts containing numpy arrays of training samples
    if use_train:
        train_dataset_generator = dataset_generator(train_rgb_images_path, train_depth_images_path, train_labels_path)
        print('[INFO] Training set generator has been created')

    # Create python dicts containing numpy arrays of test samples
    if use_test:
        test_dataset_generator = dataset_generator_nolabel(test_rgb_images_path, test_depth_images_path)
        print('[INFO] Test set generator has been created')
    return train_dataset_generator, test_dataset_generator

def dataset_generator(rgb_images_path, depth_images_path, labels_path):
    dataset = CreateData(rgb_images_path,depth_images_path,labels_path)
    return dataset

def dataset_generator_nolabel(rgb_images_path, depth_images_path):
    dataset = CreateData_nolabel(rgb_images_path,depth_images_path)
    return dataset