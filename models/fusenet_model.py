import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

class FuseNet(nn.Module):
    def __init__(self, num_labels, gpu_device=0, use_class=True):
        super(FuseNet, self).__init__()

        # Load pre-trained VGG-16 weights to two separate variables.
        # They will be used in defining the depth and RGB encoder sequential layers.
        bn_moment = 0.1
        feats = list()
        #0
        #feats.append(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1))
        feats.append(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3))
        #1
        feats.append(nn.Conv2d(3, 64, kernel_size=1))
        #2
        feats.append(nn.ReLU())
        #3
        #feats.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
        feats.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64))
        #4
        feats.append(nn.Conv2d(64, 64, kernel_size=1))
        #5
        feats.append(nn.ReLU())
        #6
        feats.append(nn.MaxPool2d(kernel_size=2, stride=2, dilation=1))
        #7
        #feats.append(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))
        feats.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64))
        #8
        feats.append(nn.Conv2d(64, 128, kernel_size=1))
        #9
        feats.append(nn.ReLU())
        #10
        #feats.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        feats.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128))
        #11
        feats.append(nn.Conv2d(128, 128, kernel_size=1))
        #12
        feats.append(nn.ReLU())
        #13
        feats.append(nn.MaxPool2d(kernel_size=2, stride=2, dilation=1))
        #14
        #feats.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        feats.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128))
        #15
        feats.append(nn.Conv2d(128, 256, kernel_size=1))
        #16
        feats.append(nn.ReLU())
        #17
        #feats.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256))
        #18
        feats.append(nn.Conv2d(256, 256, kernel_size=1))
        #19
        feats.append(nn.ReLU())
        #20
        #feats.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256))
        #21
        feats.append(nn.Conv2d(256, 256, kernel_size=1))
        #22
        feats.append(nn.ReLU())
        #23
        feats.append(nn.MaxPool2d(kernel_size=2, stride=2, dilation=1))
        #24
        #feats.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256))
        #25
        feats.append(nn.Conv2d(256, 512, kernel_size=1))
        #26
        feats.append(nn.ReLU())
        #27
        #feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        feats.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512))
        #28
        feats.append(nn.Conv2d(512, 512, kernel_size=1))
        #29
        feats.append(nn.ReLU())
        #30
        #feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        feats.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512))
        #31
        feats.append(nn.Conv2d(512, 512, kernel_size=1))
        #32
        feats.append(nn.ReLU())
        #33
        feats.append(nn.MaxPool2d(kernel_size=2, stride=2, dilation=1))
        #34
        #feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        feats.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512))
        #35
        feats.append(nn.Conv2d(512, 512, kernel_size=1))
        #36
        feats.append(nn.ReLU())
        #37
        #feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        feats.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512))
        #38
        feats.append(nn.Conv2d(512, 512, kernel_size=1))

        #39
        feats.append(nn.ReLU())
        #40
        #feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        feats.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512))
        #41
        feats.append(nn.Conv2d(512, 512, kernel_size=1))
        #42
        feats.append(nn.ReLU())
        #43
        feats.append(nn.MaxPool2d(kernel_size=2, stride=2, dilation=1))

        feats2 = feats

        # Average the first layer of feats variable, the input-layer weights of VGG-16,
        # over the channel dimension, as depth encoder will be accepting one-dimensional
        # inputs instead of three.
        # avg = torch.mean(feats[0].cuda(gpu_device).weight.data, dim=1)
        # avg = avg.unsqueeze(1)
        bn_moment = 0.1
        self.use_class = use_class


        # DEPTH ENCODER
        self.conv11d = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1).cuda(gpu_device)

        #self.conv11d.weight.data = avg

        self.CBR1_D = nn.Sequential(
            nn.BatchNorm2d(64).cuda(gpu_device),
            feats[2].cuda(gpu_device),
            feats[3].cuda(gpu_device),
            feats[4].cuda(gpu_device),
            nn.BatchNorm2d(64).cuda(gpu_device),
            feats[5].cuda(gpu_device),
        )

        self.CBR2_D = nn.Sequential(
            feats[7].cuda(gpu_device),
            feats[8].cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),
            feats[9].cuda(gpu_device),
            feats[10].cuda(gpu_device),
            feats[11].cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),
            feats[12].cuda(gpu_device),
        )


        self.CBR3_D = nn.Sequential(
            feats[14].cuda(gpu_device),
            feats[15].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats[16].cuda(gpu_device),
            feats[17].cuda(gpu_device),
            feats[18].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats[19].cuda(gpu_device),
            feats[20].cuda(gpu_device),
            feats[21].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats[22].cuda(gpu_device),
        )
        self.dropout3_d = nn.Dropout(p=0.5).cuda(gpu_device)

        self.CBR4_D = nn.Sequential(
            feats[24].cuda(gpu_device),
            feats[25].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[26].cuda(gpu_device),
            feats[27].cuda(gpu_device),
            feats[28].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[29].cuda(gpu_device),
            feats[30].cuda(gpu_device),
            feats[31].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[32].cuda(gpu_device),
        )
        self.dropout4_d = nn.Dropout(p=0.5).cuda(gpu_device)

        self.CBR5_D = nn.Sequential(
            feats[34].cuda(gpu_device),
            feats[35].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[36].cuda(gpu_device),
            feats[37].cuda(gpu_device),
            feats[38].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[39].cuda(gpu_device),
            feats[40].cuda(gpu_device),
            feats[41].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[42].cuda(gpu_device),
        )

        # RGB ENCODER
        self.CBR1_RGB = nn.Sequential(
            feats2[0].cuda(gpu_device),
            feats2[1].cuda(gpu_device),
            nn.BatchNorm2d(64).cuda(gpu_device),
            feats2[2].cuda(gpu_device),
            feats2[3].cuda(gpu_device),
            feats2[4].cuda(gpu_device),
            nn.BatchNorm2d(64).cuda(gpu_device),
            feats2[5].cuda(gpu_device),
        )

        self.CBR2_RGB = nn.Sequential(
            feats2[7].cuda(gpu_device),
            feats2[8].cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),
            feats2[9].cuda(gpu_device),
            feats2[10].cuda(gpu_device),
            feats2[11].cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),
            feats2[12].cuda(gpu_device),
        )


        self.CBR3_RGB = nn.Sequential(
            feats2[14].cuda(gpu_device),
            feats2[15].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats2[16].cuda(gpu_device),
            feats2[17].cuda(gpu_device),
            feats2[18].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats2[19].cuda(gpu_device),
            feats2[20].cuda(gpu_device),
            feats2[21].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats2[22].cuda(gpu_device),
        )
        self.dropout3 = nn.Dropout(p=0.5).cuda(gpu_device)

        self.CBR4_RGB = nn.Sequential(
            feats2[24].cuda(gpu_device),
            feats2[25].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[26].cuda(gpu_device),
            feats2[27].cuda(gpu_device),
            feats2[28].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[29].cuda(gpu_device),
            feats2[30].cuda(gpu_device),
            feats2[31].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[32].cuda(gpu_device),
        )
        self.dropout4 = nn.Dropout(p=0.5).cuda(gpu_device)

        self.CBR5_RGB = nn.Sequential(
            feats2[34].cuda(gpu_device),
            feats2[35].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[36].cuda(gpu_device),
            feats2[37].cuda(gpu_device),
            feats2[38].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[39].cuda(gpu_device),
            feats2[40].cuda(gpu_device),
            feats2[41].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[42].cuda(gpu_device),
        )
        self.dropout5 = nn.Dropout(p=0.5).cuda(gpu_device)

        # RGB DECODER
        self.CBR5_Dec = nn.Sequential(
            #nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512).cuda(gpu_device),
            nn.Conv2d(512, 512, kernel_size=1).cuda(gpu_device),
            nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            #nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512).cuda(gpu_device),
            nn.Conv2d(512, 512, kernel_size=1).cuda(gpu_device),
            nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            #nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512).cuda(gpu_device),
            nn.Conv2d(512, 512, kernel_size=1).cuda(gpu_device),
            nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Dropout(p=0.5).cuda(gpu_device),
        )

        self.CBR4_Dec = nn.Sequential(
            #nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512).cuda(gpu_device),
            nn.Conv2d(512, 512, kernel_size=1).cuda(gpu_device),
            nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            #nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512).cuda(gpu_device),
            nn.Conv2d(512, 512, kernel_size=1).cuda(gpu_device),
            nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            #nn.Conv2d(256, 128, kernel_size=3, padding=1).cuda(gpu_device),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512).cuda(gpu_device),
            nn.Conv2d(512, 256, kernel_size=1).cuda(gpu_device),
            nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Dropout(p=0.5).cuda(gpu_device),
        )

        self.CBR3_Dec = nn.Sequential(
            #nn.Conv2d(128, 128, kernel_size=3, padding=1).cuda(gpu_device),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256).cuda(gpu_device),
            nn.Conv2d(256, 256, kernel_size=1).cuda(gpu_device),
            nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            #nn.Conv2d(128, 128, kernel_size=3, padding=1).cuda(gpu_device),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256).cuda(gpu_device),
            nn.Conv2d(256, 256, kernel_size=1).cuda(gpu_device),
            nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            #nn.Conv2d(128, 64, kernel_size=3, padding=1).cuda(gpu_device),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256).cuda(gpu_device),
            nn.Conv2d(256, 128, kernel_size=1).cuda(gpu_device),
            nn.BatchNorm2d(128, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Dropout(p=0.5).cuda(gpu_device),
        )

        self.CBR2_Dec = nn.Sequential(
            #nn.Conv2d(64, 64, kernel_size=3, padding=1).cuda(gpu_device),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128).cuda(gpu_device),
            nn.Conv2d(128, 128, kernel_size=1).cuda(gpu_device),
            nn.BatchNorm2d(128, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            #nn.Conv2d(64, 32, kernel_size=3, padding=1).cuda(gpu_device),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128).cuda(gpu_device),
            nn.Conv2d(128, 64, kernel_size=1).cuda(gpu_device),
            nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )

        self.CBR1_Dec = nn.Sequential(
            #nn.Conv2d(32, 32, kernel_size=3, padding=1).cuda(gpu_device),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64).cuda(gpu_device),
            nn.Conv2d(64, 64, kernel_size=1).cuda(gpu_device),
            nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            #nn.Conv2d(32, num_labels, kernel_size=3, padding=1).cuda(gpu_device),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64).cuda(gpu_device),
            nn.Conv2d(64, num_labels, kernel_size=1).cuda(gpu_device),
        )

        print('[INFO] FuseNet model has been created')
        self.initialize_weights()

    # He Initialization for the linear layers in the classification head
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_out = size[0]  # number of rows
                fan_in = size[1]  # number of columns
                variance = np.sqrt(4.0/(fan_in + fan_out))
                m.weight.data.normal_(0.0, variance)

    def forward(self, rgb_inputs, depth_inputs):
        #DEPTH
        #Stage 1
        x = self.conv11d(depth_inputs)
        x_1 = self.CBR1_D(x)
        pool1_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        x, id1_d = pool1_d(x_1)

        # Stage 2
        x_2 = self.CBR2_D(x)
        pool2_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        x, id2_d = pool2_d(x_2)
       
        # Stage 3
        x_3 = self.CBR3_D(x)
        pool3_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        x, id3_d = pool3_d(x_3)
        x = self.dropout3_d(x)

        # Stage 4
        x_4 = self.CBR4_D(x)
        pool4_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        x, id3_d = pool4_d(x_4)
        x = self.dropout4_d(x)

        # Stage 5
        x_5 = self.CBR5_D(x)

        # RGB ENCODER
        # Stage 1
        y = self.CBR1_RGB(rgb_inputs)
        y = torch.add(y, x_1)
        pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        y, id1 = pool1(y)

        # Stage 2
        y = self.CBR2_RGB(y)
        y = torch.add(y, x_2)
        pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        y, id2 = pool2(y)
      
        # Stage 3
        y = self.CBR3_RGB(y)
        y = torch.add(y, x_3)
        pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        y, id3 = pool3(y)
        y = self.dropout3(y)

        # Stage 4
        y = self.CBR4_RGB(y)
        y = torch.add(y,x_4)
        pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        y, id4 = pool4(y)
        y = self.dropout4(y)

        # Stage 5
        y = self.CBR5_RGB(y)
        y = torch.add(y, x_5)
        y_size = y.size()


        pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        y, id5 = pool5(y)
        y = self.dropout5(y)


        # DECODER
        # Stage 5 dec
        unpool5 = nn.Upsample(size=[15,20], mode='bilinear',align_corners=True)
        #y = unpool5(y, id5, output_size=torch.Size(y_size))
        y = unpool5(y)
        y = self.CBR5_Dec(y)

        # Stage 4 dec
        # unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # y = unpool4(y, id4)
        unpool4 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        y = unpool4(y)
        y = self.CBR4_Dec(y)

        # Stage 3 dec
        # unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # y = unpool3(y, id3)
        unpool3 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        y = unpool3(y)
        y = self.CBR3_Dec(y)

        # Stage 2 dec
        # unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # y = unpool2(y, id2)
        unpool2 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        y = unpool2(y)
        y = self.CBR2_Dec(y)

        # Stage 1 dec
        # unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # y = unpool1(y, id1)
        unpool1 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        y = unpool1(y)
        y = self.CBR1_Dec(y)

        return y
#depthconv
# class FuseNet(nn.Module):
#     def __init__(self, num_labels, gpu_device=0, use_class=True):
#         super(FuseNet, self).__init__()

#         # Load pre-trained VGG-16 weights to two separate variables.
#         # They will be used in defining the depth and RGB encoder sequential layers.
#         bn_moment = 0.1
#         feats = list()
#         #0
#         #feats.append(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1))
#         feats.append(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3))
#         #1
#         feats.append(nn.Conv2d(3, 64, kernel_size=1))
#         #2
#         feats.append(nn.ReLU())
#         #3
#         #feats.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
#         feats.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64))
#         #4
#         feats.append(nn.Conv2d(64, 64, kernel_size=1))
#         #5
#         feats.append(nn.ReLU())
#         #6
#         feats.append(nn.MaxPool2d(kernel_size=2, stride=2, dilation=1))
#         #7
#         #feats.append(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))
#         feats.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64))
#         #8
#         feats.append(nn.Conv2d(64, 128, kernel_size=1))
#         #9
#         feats.append(nn.ReLU())
#         #10
#         #feats.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
#         feats.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128))
#         #11
#         feats.append(nn.Conv2d(128, 128, kernel_size=1))
#         #12
#         feats.append(nn.ReLU())
#         #13
#         feats.append(nn.MaxPool2d(kernel_size=2, stride=2, dilation=1))
#         #14
#         #feats.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
#         feats.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128))
#         #15
#         feats.append(nn.Conv2d(128, 256, kernel_size=1))
#         #16
#         feats.append(nn.ReLU())
#         #17
#         #feats.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
#         feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256))
#         #18
#         feats.append(nn.Conv2d(256, 256, kernel_size=1))
#         #19
#         feats.append(nn.ReLU())
#         #20
#         #feats.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
#         feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256))
#         #21
#         feats.append(nn.Conv2d(256, 256, kernel_size=1))
#         #22
#         feats.append(nn.ReLU())
#         #23
#         feats.append(nn.MaxPool2d(kernel_size=2, stride=2, dilation=1))
#         #24
#         #feats.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
#         feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256))
#         #25
#         feats.append(nn.Conv2d(256, 512, kernel_size=1))
#         #26
#         feats.append(nn.ReLU())
#         #27
#         #feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
#         feats.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512))
#         #28
#         feats.append(nn.Conv2d(512, 512, kernel_size=1))
#         #29
#         feats.append(nn.ReLU())
#         #30
#         #feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
#         feats.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512))
#         #31
#         feats.append(nn.Conv2d(512, 512, kernel_size=1))
#         #32
#         feats.append(nn.ReLU())
#         #33
#         feats.append(nn.MaxPool2d(kernel_size=2, stride=2, dilation=1))
#         #34
#         #feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
#         feats.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512))
#         #35
#         feats.append(nn.Conv2d(512, 512, kernel_size=1))
#         #36
#         feats.append(nn.ReLU())
#         #37
#         #feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
#         feats.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512))
#         #38
#         feats.append(nn.Conv2d(512, 512, kernel_size=1))

#         #39
#         feats.append(nn.ReLU())
#         #40
#         #feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
#         feats.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512))
#         #41
#         feats.append(nn.Conv2d(512, 512, kernel_size=1))
#         #42
#         feats.append(nn.ReLU())
#         #43
#         feats.append(nn.MaxPool2d(kernel_size=2, stride=2, dilation=1))

#         feats2 = feats

#         # Average the first layer of feats variable, the input-layer weights of VGG-16,
#         # over the channel dimension, as depth encoder will be accepting one-dimensional
#         # inputs instead of three.
#         # avg = torch.mean(feats[0].cuda(gpu_device).weight.data, dim=1)
#         # avg = avg.unsqueeze(1)
#         bn_moment = 0.1
#         self.use_class = use_class


#         # DEPTH ENCODER
#         self.conv11d = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1).cuda(gpu_device)

#         #self.conv11d.weight.data = avg

#         self.CBR1_D = nn.Sequential(
#             nn.BatchNorm2d(64).cuda(gpu_device),
#             feats[2].cuda(gpu_device),
#             feats[3].cuda(gpu_device),
#             feats[4].cuda(gpu_device),
#             nn.BatchNorm2d(64).cuda(gpu_device),
#             feats[5].cuda(gpu_device),
#         )

#         self.CBR2_D = nn.Sequential(
#             feats[7].cuda(gpu_device),
#             feats[8].cuda(gpu_device),
#             nn.BatchNorm2d(128).cuda(gpu_device),
#             feats[9].cuda(gpu_device),
#             feats[10].cuda(gpu_device),
#             feats[11].cuda(gpu_device),
#             nn.BatchNorm2d(128).cuda(gpu_device),
#             feats[12].cuda(gpu_device),
#         )


#         self.CBR3_D = nn.Sequential(
#             feats[14].cuda(gpu_device),
#             feats[15].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats[16].cuda(gpu_device),
#             feats[17].cuda(gpu_device),
#             feats[18].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats[19].cuda(gpu_device),
#             feats[20].cuda(gpu_device),
#             feats[21].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats[22].cuda(gpu_device),
#         )
#         self.dropout3_d = nn.Dropout(p=0.5).cuda(gpu_device)

#         self.CBR4_D = nn.Sequential(
#             feats[24].cuda(gpu_device),
#             feats[25].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats[26].cuda(gpu_device),
#             feats[27].cuda(gpu_device),
#             feats[28].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats[29].cuda(gpu_device),
#             feats[30].cuda(gpu_device),
#             feats[31].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats[32].cuda(gpu_device),
#         )
#         self.dropout4_d = nn.Dropout(p=0.5).cuda(gpu_device)

#         self.CBR5_D = nn.Sequential(
#             feats[34].cuda(gpu_device),
#             feats[35].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats[36].cuda(gpu_device),
#             feats[37].cuda(gpu_device),
#             feats[38].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats[39].cuda(gpu_device),
#             feats[40].cuda(gpu_device),
#             feats[41].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats[42].cuda(gpu_device),
#         )

#         # RGB ENCODER
#         self.CBR1_RGB = nn.Sequential(
#             feats2[0].cuda(gpu_device),
#             feats2[1].cuda(gpu_device),
#             nn.BatchNorm2d(64).cuda(gpu_device),
#             feats2[2].cuda(gpu_device),
#             feats2[3].cuda(gpu_device),
#             feats2[4].cuda(gpu_device),
#             nn.BatchNorm2d(64).cuda(gpu_device),
#             feats2[5].cuda(gpu_device),
#         )

#         self.CBR2_RGB = nn.Sequential(
#             feats2[7].cuda(gpu_device),
#             feats2[8].cuda(gpu_device),
#             nn.BatchNorm2d(128).cuda(gpu_device),
#             feats2[9].cuda(gpu_device),
#             feats2[10].cuda(gpu_device),
#             feats2[11].cuda(gpu_device),
#             nn.BatchNorm2d(128).cuda(gpu_device),
#             feats2[12].cuda(gpu_device),
#         )


#         self.CBR3_RGB = nn.Sequential(
#             feats2[14].cuda(gpu_device),
#             feats2[15].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats2[16].cuda(gpu_device),
#             feats2[17].cuda(gpu_device),
#             feats2[18].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats2[19].cuda(gpu_device),
#             feats2[20].cuda(gpu_device),
#             feats2[21].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats2[22].cuda(gpu_device),
#         )
#         self.dropout3 = nn.Dropout(p=0.5).cuda(gpu_device)

#         self.CBR4_RGB = nn.Sequential(
#             feats2[24].cuda(gpu_device),
#             feats2[25].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats2[26].cuda(gpu_device),
#             feats2[27].cuda(gpu_device),
#             feats2[28].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats2[29].cuda(gpu_device),
#             feats2[30].cuda(gpu_device),
#             feats2[31].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats2[32].cuda(gpu_device),
#         )
#         self.dropout4 = nn.Dropout(p=0.5).cuda(gpu_device)

#         self.CBR5_RGB = nn.Sequential(
#             feats2[34].cuda(gpu_device),
#             feats2[35].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats2[36].cuda(gpu_device),
#             feats2[37].cuda(gpu_device),
#             feats2[38].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats2[39].cuda(gpu_device),
#             feats2[40].cuda(gpu_device),
#             feats2[41].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats2[42].cuda(gpu_device),
#         )
#         self.dropout5 = nn.Dropout(p=0.5).cuda(gpu_device)

#         # RGB DECODER
#         self.CBR5_Dec = nn.Sequential(
#             #nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512).cuda(gpu_device),
#             nn.Conv2d(512, 512, kernel_size=1).cuda(gpu_device),
#             nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             #nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512).cuda(gpu_device),
#             nn.Conv2d(512, 512, kernel_size=1).cuda(gpu_device),
#             nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             #nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512).cuda(gpu_device),
#             nn.Conv2d(512, 512, kernel_size=1).cuda(gpu_device),
#             nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Dropout(p=0.5).cuda(gpu_device),
#         )

#         self.CBR4_Dec = nn.Sequential(
#             #nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512).cuda(gpu_device),
#             nn.Conv2d(512, 512, kernel_size=1).cuda(gpu_device),
#             nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             #nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512).cuda(gpu_device),
#             nn.Conv2d(512, 512, kernel_size=1).cuda(gpu_device),
#             nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             #nn.Conv2d(256, 128, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512).cuda(gpu_device),
#             nn.Conv2d(512, 256, kernel_size=1).cuda(gpu_device),
#             nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Dropout(p=0.5).cuda(gpu_device),
#         )

#         self.CBR3_Dec = nn.Sequential(
#             #nn.Conv2d(128, 128, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256).cuda(gpu_device),
#             nn.Conv2d(256, 256, kernel_size=1).cuda(gpu_device),
#             nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             #nn.Conv2d(128, 128, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256).cuda(gpu_device),
#             nn.Conv2d(256, 256, kernel_size=1).cuda(gpu_device),
#             nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             #nn.Conv2d(128, 64, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256).cuda(gpu_device),
#             nn.Conv2d(256, 128, kernel_size=1).cuda(gpu_device),
#             nn.BatchNorm2d(128, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Dropout(p=0.5).cuda(gpu_device),
#         )

#         self.CBR2_Dec = nn.Sequential(
#             #nn.Conv2d(64, 64, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128).cuda(gpu_device),
#             nn.Conv2d(128, 128, kernel_size=1).cuda(gpu_device),
#             nn.BatchNorm2d(128, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             #nn.Conv2d(64, 32, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128).cuda(gpu_device),
#             nn.Conv2d(128, 64, kernel_size=1).cuda(gpu_device),
#             nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#         )

#         self.CBR1_Dec = nn.Sequential(
#             #nn.Conv2d(32, 32, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64).cuda(gpu_device),
#             nn.Conv2d(64, 64, kernel_size=1).cuda(gpu_device),
#             nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             #nn.Conv2d(32, num_labels, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64).cuda(gpu_device),
#             nn.Conv2d(64, num_labels, kernel_size=1).cuda(gpu_device),
#         )

#         print('[INFO] FuseNet model has been created')
#         self.initialize_weights()

#     # He Initialization for the linear layers in the classification head
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 size = m.weight.size()
#                 fan_out = size[0]  # number of rows
#                 fan_in = size[1]  # number of columns
#                 variance = np.sqrt(4.0/(fan_in + fan_out))
#                 m.weight.data.normal_(0.0, variance)

#     def forward(self, rgb_inputs, depth_inputs):
#         # DEPTH ENCODER
#         # Stage 1
#         # print(rgb_inputs.size())
#         # print(depth_inputs.size())

#         # x = self.conv11d(depth_inputs)
#         # x_1 = self.CBR1_D(x)
#         # x, id1_d = F.max_pool2d(x_1, kernel_size=2, stride=2, return_indices=True)

#         # # Stage 2
#         # x_2 = self.CBR2_D(x)
#         # x, id2_d = F.max_pool2d(x_2, kernel_size=2, stride=2, return_indices=True)
       
#         # # Stage 3
#         # x_3 = self.CBR3_D(x)
#         # x, id3_d = F.max_pool2d(x_3, kernel_size=2, stride=2, return_indices=True)
#         # x = self.dropout3_d(x)

#         # # Stage 4
#         # x_4 = self.CBR4_D(x)
#         # x, id4_d = F.max_pool2d(x_4, kernel_size=2, stride=2, return_indices=True)
#         # x = self.dropout4_d(x)

#         # # Stage 5
#         # x_5 = self.CBR5_D(x)

#         # # RGB ENCODER
#         # # Stage 1
#         # y = self.CBR1_RGB(rgb_inputs)
#         # y = torch.add(y, x_1)
#         # y, id1 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

#         # # Stage 2
#         # y = self.CBR2_RGB(y)
#         # y = torch.add(y, x_2)
#         # y, id2 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
      
#         # # Stage 3
#         # y = self.CBR3_RGB(y)
#         # y = torch.add(y, x_3)
#         # y, id3 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
#         # y = self.dropout3(y)

#         # # Stage 4
#         # y = self.CBR4_RGB(y)
#         # y = torch.add(y,x_4)
#         # y, id4 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
#         # y = self.dropout4(y)

#         # # Stage 5
#         # y = self.CBR5_RGB(y)
#         # y = torch.add(y, x_5)
#         # y_size = y.size()

#         # y, id5 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
#         # y = self.dropout5(y)

#         # # DECODER
#         # # Stage 5 dec
#         # y = F.max_unpool2d(y, id5, kernel_size=2, stride=2, output_size=y_size)
#         # y = self.CBR5_Dec(y)

#         # # Stage 4 dec
#         # y = F.max_unpool2d(y, id4, kernel_size=2, stride=2)
#         # y = self.CBR4_Dec(y)

#         # # Stage 3 dec
#         # y = F.max_unpool2d(y, id3, kernel_size=2, stride=2)
#         # y = self.CBR3_Dec(y)

#         # # Stage 2 dec
#         # y = F.max_unpool2d(y, id2, kernel_size=2, stride=2)
#         # y = self.CBR2_Dec(y)

#         # # Stage 1 dec
#         # y = F.max_unpool2d(y, id1, kernel_size=2, stride=2)
#         # y = self.CBR1_Dec(y)


#         #DEPTH
#         #Stage 1
#         x = self.conv11d(depth_inputs)
#         x_1 = self.CBR1_D(x)
#         pool1_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
#         x, id1_d = pool1_d(x_1)

#         # Stage 2
#         x_2 = self.CBR2_D(x)
#         pool2_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
#         x, id2_d = pool2_d(x_2)
       
#         # Stage 3
#         x_3 = self.CBR3_D(x)
#         pool3_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
#         x, id3_d = pool3_d(x_3)
#         x = self.dropout3_d(x)

#         # Stage 4
#         x_4 = self.CBR4_D(x)
#         pool4_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
#         x, id3_d = pool4_d(x_4)
#         x = self.dropout4_d(x)


#         # Stage 5
#         x_5 = self.CBR5_D(x)

#         # RGB ENCODER
#         # Stage 1
#         y = self.CBR1_RGB(rgb_inputs)
#         y = torch.add(y, x_1)
#         pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
#         y, id1 = pool1(y)

#         # Stage 2
#         y = self.CBR2_RGB(y)
#         y = torch.add(y, x_2)
#         pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
#         y, id2 = pool2(y)
      
#         # Stage 3
#         y = self.CBR3_RGB(y)
#         y = torch.add(y, x_3)
#         pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
#         y, id3 = pool3(y)
#         y = self.dropout3(y)

#         # Stage 4
#         y = self.CBR4_RGB(y)
#         y = torch.add(y,x_4)
#         pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
#         y, id4 = pool4(y)
#         y = self.dropout4(y)

#         # Stage 5
#         y = self.CBR5_RGB(y)
#         y = torch.add(y, x_5)
#         y_size = y.size()


#         pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
#         y, id5 = pool5(y)
#         y = self.dropout5(y)


#         # DECODER
#         # Stage 5 dec
#         unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
#         #deconv = nn.ConvTranspose2d(in_channels=y.shape[1], out_channels=y.shape[1], kernel_size=2, stride=1, padding=1).cuda(0)
#         y = unpool5(y, id5, output_size=torch.Size(y_size))
#         y = self.CBR5_Dec(y)

#         # Stage 4 dec
#         unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
#         y = unpool4(y, id4)
#         y = self.CBR4_Dec(y)

#         # Stage 3 dec
#         unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
#         y = unpool3(y, id3)
#         y = self.CBR3_Dec(y)

#         # Stage 2 dec
#         unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
#         y = unpool2(y, id2)
#         y = self.CBR2_Dec(y)

#         # Stage 1 dec
#         unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
#         y = unpool1(y, id1)
#         y = self.CBR1_Dec(y)

#         return y
#一半通道
# class FuseNet(nn.Module):
#     def __init__(self, num_labels, gpu_device=0, use_class=True):
#         super(FuseNet, self).__init__()

#         # Load pre-trained VGG-16 weights to two separate variables.
#         # They will be used in defining the depth and RGB encoder sequential layers.
#         bn_moment = 0.1
#         feats = list()
#         #0
#         feats.append(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1))
#         #1
#         feats.append(nn.ReLU())
#         #2
#         feats.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
#         #3
#         feats.append(nn.ReLU())
#         #4
#         feats.append(nn.MaxPool2d(kernel_size=2, stride=2, dilation=1))
#         #5
#         feats.append(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))
#         #6
#         feats.append(nn.ReLU())
#         #7
#         feats.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
#         #8
#         feats.append(nn.ReLU())
#         #9
#         feats.append(nn.MaxPool2d(kernel_size=2, stride=2, dilation=1))
#         #10
#         feats.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
#         #11
#         feats.append(nn.ReLU())
#         #12
#         feats.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
#         #13
#         feats.append(nn.ReLU())
#         #14
#         feats.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
#         #15
#         feats.append(nn.ReLU())
#         #16
#         feats.append(nn.MaxPool2d(kernel_size=2, stride=2, dilation=1))
#         #17
#         feats.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
#         #18
#         feats.append(nn.ReLU())
#         #19
#         feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
#         #20
#         feats.append(nn.ReLU())
#         #21
#         feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
#         #22
#         feats.append(nn.ReLU())
#         #23
#         feats.append(nn.MaxPool2d(kernel_size=2, stride=2, dilation=1))
#         #24
#         feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
#         #25
#         feats.append(nn.ReLU())
#         #26
#         feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
#         #27
#         feats.append(nn.ReLU())
#         #28
#         feats.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
#         #29
#         feats.append(nn.ReLU())
#         #30
#         feats.append(nn.MaxPool2d(kernel_size=2, stride=2, dilation=1))

#         feats2 = feats

#         # Average the first layer of feats variable, the input-layer weights of VGG-16,
#         # over the channel dimension, as depth encoder will be accepting one-dimensional
#         # inputs instead of three.
#         avg = torch.mean(feats[0].cuda(gpu_device).weight.data, dim=1)
#         avg = avg.unsqueeze(1)
#         bn_moment = 0.1
#         self.use_class = use_class


#         # DEPTH ENCODER
#         self.conv11d = nn.Conv2d(1, 32, kernel_size=3, padding=1).cuda(gpu_device)
#         self.conv11d.weight.data = avg

#         self.CBR1_D = nn.Sequential(
#             nn.BatchNorm2d(32).cuda(gpu_device),
#             feats[1].cuda(gpu_device),
#             feats[2].cuda(gpu_device),
#             nn.BatchNorm2d(32).cuda(gpu_device),
#             feats[3].cuda(gpu_device),
#         )

#         self.CBR2_D = nn.Sequential(
#             feats[5].cuda(gpu_device),
#             nn.BatchNorm2d(64).cuda(gpu_device),
#             feats[6].cuda(gpu_device),
#             feats[7].cuda(gpu_device),
#             nn.BatchNorm2d(64).cuda(gpu_device),
#             feats[8].cuda(gpu_device),
#         )


#         self.CBR3_D = nn.Sequential(
#             feats[10].cuda(gpu_device),
#             nn.BatchNorm2d(128).cuda(gpu_device),
#             feats[11].cuda(gpu_device),
#             feats[12].cuda(gpu_device),
#             nn.BatchNorm2d(128).cuda(gpu_device),
#             feats[13].cuda(gpu_device),
#             feats[14].cuda(gpu_device),
#             nn.BatchNorm2d(128).cuda(gpu_device),
#             feats[15].cuda(gpu_device),
#         )
#         self.dropout3_d = nn.Dropout(p=0.5).cuda(gpu_device)

#         self.CBR4_D = nn.Sequential(
#             feats[17].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats[18].cuda(gpu_device),
#             feats[19].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats[20].cuda(gpu_device),
#             feats[21].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats[22].cuda(gpu_device),
#         )
#         self.dropout4_d = nn.Dropout(p=0.5).cuda(gpu_device)

#         self.CBR5_D = nn.Sequential(
#             feats[24].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats[25].cuda(gpu_device),
#             feats[26].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats[27].cuda(gpu_device),
#             feats[28].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats[29].cuda(gpu_device),
#         )

#         # RGB ENCODER
#         self.CBR1_RGB = nn.Sequential(
#             feats2[0].cuda(gpu_device),
#             nn.BatchNorm2d(32).cuda(gpu_device),
#             feats2[1].cuda(gpu_device),
#             feats2[2].cuda(gpu_device),
#             nn.BatchNorm2d(32).cuda(gpu_device),
#             feats2[3].cuda(gpu_device),
#         )

#         self.CBR2_RGB = nn.Sequential(
#             feats2[5].cuda(gpu_device),
#             nn.BatchNorm2d(64).cuda(gpu_device),
#             feats2[6].cuda(gpu_device),
#             feats2[7].cuda(gpu_device),
#             nn.BatchNorm2d(64).cuda(gpu_device),
#             feats2[8].cuda(gpu_device),
#         )


#         self.CBR3_RGB = nn.Sequential(
#             feats2[10].cuda(gpu_device),
#             nn.BatchNorm2d(128).cuda(gpu_device),
#             feats2[11].cuda(gpu_device),
#             feats2[12].cuda(gpu_device),
#             nn.BatchNorm2d(128).cuda(gpu_device),
#             feats2[13].cuda(gpu_device),
#             feats2[14].cuda(gpu_device),
#             nn.BatchNorm2d(128).cuda(gpu_device),
#             feats2[15].cuda(gpu_device),
#         )
#         self.dropout3 = nn.Dropout(p=0.5).cuda(gpu_device)

#         self.CBR4_RGB = nn.Sequential(
#             feats2[17].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats2[18].cuda(gpu_device),
#             feats2[19].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats2[20].cuda(gpu_device),
#             feats2[21].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats2[22].cuda(gpu_device),
#         )
#         self.dropout4 = nn.Dropout(p=0.5).cuda(gpu_device)

#         self.CBR5_RGB = nn.Sequential(
#             feats2[24].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats2[25].cuda(gpu_device),
#             feats2[26].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats2[27].cuda(gpu_device),
#             feats2[28].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats2[29].cuda(gpu_device),
#         )
#         self.dropout5 = nn.Dropout(p=0.5).cuda(gpu_device)

#         # RGB DECODER
#         self.CBR5_Dec = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Dropout(p=0.5).cuda(gpu_device),
#         )

#         self.CBR4_Dec = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Conv2d(256, 128, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(128, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Dropout(p=0.5).cuda(gpu_device),
#         )

#         self.CBR3_Dec = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(128, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(128, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Conv2d(128, 64, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Dropout(p=0.5).cuda(gpu_device),
#         )

#         self.CBR2_Dec = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Conv2d(64, 32, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(32, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#         )

#         self.CBR1_Dec = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(32, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Conv2d(32, num_labels, kernel_size=3, padding=1).cuda(gpu_device),
#         )

#         print('[INFO] FuseNet model has been created')
#         self.initialize_weights()

#     # He Initialization for the linear layers in the classification head
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 size = m.weight.size()
#                 fan_out = size[0]  # number of rows
#                 fan_in = size[1]  # number of columns
#                 variance = np.sqrt(4.0/(fan_in + fan_out))
#                 m.weight.data.normal_(0.0, variance)

#     def forward(self, rgb_inputs, depth_inputs):
#         # DEPTH ENCODER
#         # Stage 1
#         # print(rgb_inputs.size())
#         # print(depth_inputs.size())
#         x = self.conv11d(depth_inputs)
#         x_1 = self.CBR1_D(x)
#         x, id1_d = F.max_pool2d(x_1, kernel_size=2, stride=2, return_indices=True)

#         # Stage 2
#         x_2 = self.CBR2_D(x)
#         x, id2_d = F.max_pool2d(x_2, kernel_size=2, stride=2, return_indices=True)
       
#         # Stage 3
#         x_3 = self.CBR3_D(x)
#         x, id3_d = F.max_pool2d(x_3, kernel_size=2, stride=2, return_indices=True)
#         x = self.dropout3_d(x)

#         # Stage 4
#         x_4 = self.CBR4_D(x)
#         x, id4_d = F.max_pool2d(x_4, kernel_size=2, stride=2, return_indices=True)
#         x = self.dropout4_d(x)

#         # Stage 5
#         x_5 = self.CBR5_D(x)

#         # RGB ENCODER
#         # Stage 1
#         y = self.CBR1_RGB(rgb_inputs)
#         y = torch.add(y, x_1)
#         y, id1 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

#         # Stage 2
#         y = self.CBR2_RGB(y)
#         y = torch.add(y, x_2)
#         y, id2 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
      
#         # Stage 3
#         y = self.CBR3_RGB(y)
#         y = torch.add(y, x_3)
#         y, id3 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
#         y = self.dropout3(y)

#         # Stage 4
#         y = self.CBR4_RGB(y)
#         y = torch.add(y,x_4)
#         y, id4 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
#         y = self.dropout4(y)

#         # Stage 5
#         y = self.CBR5_RGB(y)
#         y = torch.add(y, x_5)
#         y_size = y.size()

#         y, id5 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
#         y = self.dropout5(y)

#         if self.use_class:
#             # FC Block for Scene Classification
#             y_class = y.view(y.size(0), -1)
#             y_class = self.ClassHead(y_class)

#         # DECODER
#         # Stage 5 dec
#         y = F.max_unpool2d(y, id5, kernel_size=2, stride=2, output_size=y_size)
#         y = self.CBR5_Dec(y)

#         # Stage 4 dec
#         y = F.max_unpool2d(y, id4, kernel_size=2, stride=2)
#         y = self.CBR4_Dec(y)

#         # Stage 3 dec
#         y = F.max_unpool2d(y, id3, kernel_size=2, stride=2)
#         y = self.CBR3_Dec(y)

#         # Stage 2 dec
#         y = F.max_unpool2d(y, id2, kernel_size=2, stride=2)
#         y = self.CBR2_Dec(y)

#         # Stage 1 dec
#         y = F.max_unpool2d(y, id1, kernel_size=2, stride=2)
#         y = self.CBR1_Dec(y)

#         if self.use_class:
#             return y, y_class
#         return y

#原始的

# class FuseNet(nn.Module):
#     def __init__(self, num_labels, gpu_device=0, use_class=True):
#         super(FuseNet, self).__init__()

#         # Load pre-trained VGG-16 weights to two separate variables.
#         # They will be used in defining the depth and RGB encoder sequential layers.

#         feats = list(models.vgg16(pretrained=True).features.children())
#         feats2 = list(models.vgg16(pretrained=True).features.children())

#         # feats = list(models.densenet121(pretrained=True).features.children())
#         # feats2 = list(models.densenet121(pretrained=True).features.children())

#         # Average the first layer of feats variable, the input-layer weights of VGG-16,
#         # over the channel dimension, as depth encoder will be accepting one-dimensional
#         # inputs instead of three.
#         avg = torch.mean(feats[0].cuda(gpu_device).weight.data, dim=1)
#         avg = avg.unsqueeze(1)
#         bn_moment = 0.1
#         self.use_class = use_class

#         if use_class:
#             num_classes = 10

#         # DEPTH ENCODER
#         self.conv11d = nn.Conv2d(1, 64, kernel_size=3, padding=1).cuda(gpu_device)
#         self.conv11d.weight.data = avg

#         self.CBR1_D = nn.Sequential(
#             nn.BatchNorm2d(64).cuda(gpu_device),
#             feats[1].cuda(gpu_device),
#             feats[2].cuda(gpu_device),
#             nn.BatchNorm2d(64).cuda(gpu_device),
#             feats[3].cuda(gpu_device),
#         )
#         self.CBR2_D = nn.Sequential(
#             feats[5].cuda(gpu_device),
#             nn.BatchNorm2d(128).cuda(gpu_device),
#             feats[6].cuda(gpu_device),
#             feats[7].cuda(gpu_device),
#             nn.BatchNorm2d(128).cuda(gpu_device),
#             feats[8].cuda(gpu_device),
#         )
#         self.CBR3_D = nn.Sequential(
#             feats[10].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats[11].cuda(gpu_device),
#             feats[12].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats[13].cuda(gpu_device),
#             feats[14].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats[15].cuda(gpu_device),
#         )
#         self.dropout3_d = nn.Dropout(p=0.5).cuda(gpu_device)

#         self.CBR4_D = nn.Sequential(
#             feats[17].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats[18].cuda(gpu_device),
#             feats[19].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats[20].cuda(gpu_device),
#             feats[21].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats[22].cuda(gpu_device),
#         )
#         self.dropout4_d = nn.Dropout(p=0.5).cuda(gpu_device)

#         self.CBR5_D = nn.Sequential(
#             feats[24].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats[25].cuda(gpu_device),
#             feats[26].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats[27].cuda(gpu_device),
#             feats[28].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats[29].cuda(gpu_device),
#         )

#         # RGB ENCODER
#         self.CBR1_RGB = nn.Sequential(
#             feats2[0].cuda(gpu_device),
#             nn.BatchNorm2d(64).cuda(gpu_device),
#             feats2[1].cuda(gpu_device),
#             feats2[2].cuda(gpu_device),
#             nn.BatchNorm2d(64).cuda(gpu_device),
#             feats2[3].cuda(gpu_device),
#         )

#         self.CBR2_RGB = nn.Sequential(
#             feats2[5].cuda(gpu_device),
#             nn.BatchNorm2d(128).cuda(gpu_device),
#             feats2[6].cuda(gpu_device),
#             feats2[7].cuda(gpu_device),
#             nn.BatchNorm2d(128).cuda(gpu_device),
#             feats2[8].cuda(gpu_device),
#         )

#         self.CBR3_RGB = nn.Sequential(
#             feats2[10].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats2[11].cuda(gpu_device),
#             feats2[12].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats2[13].cuda(gpu_device),
#             feats2[14].cuda(gpu_device),
#             nn.BatchNorm2d(256).cuda(gpu_device),
#             feats2[15].cuda(gpu_device),
#         )
#         self.dropout3 = nn.Dropout(p=0.5).cuda(gpu_device)

#         self.CBR4_RGB = nn.Sequential(
#             feats2[17].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats2[18].cuda(gpu_device),
#             feats2[19].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats2[20].cuda(gpu_device),
#             feats2[21].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats2[22].cuda(gpu_device),
#         )
#         self.dropout4 = nn.Dropout(p=0.5).cuda(gpu_device)

#         self.CBR5_RGB = nn.Sequential(
#             feats2[24].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats2[25].cuda(gpu_device),
#             feats2[26].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats2[27].cuda(gpu_device),
#             feats2[28].cuda(gpu_device),
#             nn.BatchNorm2d(512).cuda(gpu_device),
#             feats2[29].cuda(gpu_device),
#         )
#         self.dropout5 = nn.Dropout(p=0.5).cuda(gpu_device)

#         if use_class:
#             self.ClassHead = nn.Sequential(
#                 # classifier[0].cuda(gpu_device),
#                 nn.Linear(35840, 4096).cuda(gpu_device),
#                 nn.ReLU(),
#                 nn.Dropout(p=0.5).cuda(gpu_device),
#                 nn.Linear(4096, 4096).cuda(gpu_device),
#                 # classifier[3].cuda(gpu_device),
#                 nn.ReLU(),
#                 nn.Dropout(p=0.5).cuda(gpu_device),
#                 nn.Linear(4096, num_classes).cuda(gpu_device)
#             )

#         # RGB DECODER
#         self.CBR5_Dec = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Dropout(p=0.5).cuda(gpu_device),
#         )

#         self.CBR4_Dec = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Conv2d(512, 256, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Dropout(p=0.5).cuda(gpu_device),
#         )

#         self.CBR3_Dec = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Conv2d(256,  128, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(128, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Dropout(p=0.5).cuda(gpu_device),
#         )

#         self.CBR2_Dec = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(128, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Conv2d(128, 64, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#         )

#         self.CBR1_Dec = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1).cuda(gpu_device),
#             nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
#             nn.ReLU().cuda(gpu_device),
#             nn.Conv2d(64, num_labels, kernel_size=3, padding=1).cuda(gpu_device),
#         )

#         print('[INFO] FuseNet model has been created')
#         self.initialize_weights()

#     # He Initialization for the linear layers in the classification head
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 size = m.weight.size()
#                 fan_out = size[0]  # number of rows
#                 fan_in = size[1]  # number of columns
#                 variance = np.sqrt(4.0/(fan_in + fan_out))
#                 m.weight.data.normal_(0.0, variance)

#     def forward(self, rgb_inputs, depth_inputs):
#         # DEPTH ENCODER
#         # Stage 1
#         # print(rgb_inputs.size())
#         # print(depth_inputs.size())
#         x = self.conv11d(depth_inputs)
#         x_1 = self.CBR1_D(x)
#         x, id1_d = F.max_pool2d(x_1, kernel_size=2, stride=2, return_indices=True)

#         # Stage 2
#         x_2 = self.CBR2_D(x)
#         x, id2_d = F.max_pool2d(x_2, kernel_size=2, stride=2, return_indices=True)

#         # Stage 3
#         x_3 = self.CBR3_D(x)
#         x, id3_d = F.max_pool2d(x_3, kernel_size=2, stride=2, return_indices=True)
#         x = self.dropout3_d(x)

#         # Stage 4
#         x_4 = self.CBR4_D(x)
#         x, id4_d = F.max_pool2d(x_4, kernel_size=2, stride=2, return_indices=True)
#         x = self.dropout4_d(x)

#         # Stage 5
#         x_5 = self.CBR5_D(x)

#         # RGB ENCODER
#         # Stage 1
#         y = self.CBR1_RGB(rgb_inputs)
#         y = torch.add(y, x_1)
#         y, id1 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

#         # Stage 2
#         y = self.CBR2_RGB(y)
#         y = torch.add(y, x_2)
#         y, id2 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

#         # Stage 3
#         y = self.CBR3_RGB(y)
#         y = torch.add(y, x_3)
#         y, id3 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
#         y = self.dropout3(y)

#         # Stage 4
#         y = self.CBR4_RGB(y)
#         y = torch.add(y,x_4)
#         y, id4 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
#         y = self.dropout4(y)

#         # Stage 5
#         y = self.CBR5_RGB(y)
#         y = torch.add(y, x_5)
#         y_size = y.size()

#         y, id5 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
#         y = self.dropout5(y)

#         if self.use_class:
#             # FC Block for Scene Classification
#             y_class = y.view(y.size(0), -1)
#             y_class = self.ClassHead(y_class)

#         # DECODER
#         # Stage 5 dec
#         y = F.max_unpool2d(y, id5, kernel_size=2, stride=2, output_size=y_size)
#         y = self.CBR5_Dec(y)

#         # Stage 4 dec
#         y = F.max_unpool2d(y, id4, kernel_size=2, stride=2)
#         y = self.CBR4_Dec(y)

#         # Stage 3 dec
#         y = F.max_unpool2d(y, id3, kernel_size=2, stride=2)
#         y = self.CBR3_Dec(y)

#         # Stage 2 dec
#         y = F.max_unpool2d(y, id2, kernel_size=2, stride=2)
#         y = self.CBR2_Dec(y)

#         # Stage 1 dec
#         y = F.max_unpool2d(y, id1, kernel_size=2, stride=2)
#         y = self.CBR1_Dec(y)

#         if self.use_class:
#             return y, y_class
#         return y
