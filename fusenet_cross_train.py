from __future__ import print_function
import os
import datetime
import torch
from torch.utils import data
from fusenet_solver import Solver
from utils.data_utils import get_data
from utils.fusenet_cross_validate import get_cross_data
from utils.loss_utils import cross_entropy_2d
from options.train_options import TrainOptions
from utils.utils import print_time_info
import numpy as np

if __name__ == '__main__':
    opt = TrainOptions().parse()
    print("Success")
    dset_name = os.path.basename(opt.dataroot)
    # if dset_name.lower().find('nyu') is not -1:
    #     dset_info = {'NYU': 40}
    # elif dset_name.lower().find('sun') is not -1:
    #     dset_info = {'SUN': 37}
    # else:
    #     raise NameError('Name of the dataset file should accordingly contain either nyu or sun in it')
    dset_info = {'selfmade': 6}
    #print('[INFO] %s dataset is being processed' % list(dset_info.keys())[0])
    print('[INFO] %s dataset is being processed' % "self-made dataset")
    #train_data, test_data = get_data(opt, use_train=True, use_test=True)
    best_iou = []
    for i in range(2,6):
        train_data, test_data = get_cross_data(opt, True,True, i)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
        print("[INFO] Data loaders for %s dataset have been created" % "self-made dataset")

        # Run an individual training session
        start_date_time = datetime.datetime.now().replace(microsecond=0)
        solver = Solver(opt, dset_info, i, loss_func=cross_entropy_2d)
        best_iou.append(solver.train_model(train_loader, test_loader, num_epochs=opt.num_epochs, log_nth=opt.print_freq))
        end_date_time = datetime.datetime.now().replace(microsecond=0)
        print_time_info(start_date_time, end_date_time)
    best = 0
    index = -1
    for k in range(len(best_iou)):
        print("Cross:%d iou:%f" %(k+1, best_iou[k]))
        if(best_iou[k]>best):
            best = best_iou[k]
            index = k+1

    print("Best iou:%f "%best)
    print("Cross:%d" %index)
    file_handle=open('resize_iou.txt',mode='w')
    for i in range(len(best_iou)):
        s = str(best_iou[i])
        file_handle.write(s)
        file_handle.write('\n')
    file_handle.close()
