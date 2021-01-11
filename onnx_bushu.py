import onnx
import onnxruntime as rt
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import torch
import matplotlib.pyplot as plt

def output(rgb_path, depth_path):
    dummy_input1 = Image.open(rgb_path)
    dummy_input2 = Image.open(depth_path)
    dummy_input1 = dummy_input1.resize([240, 320],Image.ANTIALIAS)
    dummy_input1 = transforms.ToTensor()(dummy_input1)*255
    dummy_input1 = dummy_input1.unsqueeze(0)
    dummy_input1 = dummy_input1.numpy()
    dummy_input2 = dummy_input2.resize((240, 320),Image.ANTIALIAS)
    dummy_input2 = transforms.ToTensor()(dummy_input2)
    dummy_input2 = dummy_input2.unsqueeze(0)
    dummy_input2 = dummy_input2.float()
    dummy_input2 = dummy_input2.numpy()
    pred_onnx = sess.run([seg_name], {input_name1:dummy_input1, input_name2:dummy_input2})
    pred_onnx = torch.Tensor(pred_onnx)
    pred_onnx = pred_onnx.squeeze(0)
    return pred_onnx
    
def seg_predict(pred_onnx, label_path):
    _, train_seg_preds = torch.max(pred_onnx, 1)
    train_seg_labels = np.load(label_path)
    train_seg_labels = torch.from_numpy(train_seg_labels)
    with open('/home/xinqichen/Desktop/FuseNet_code/FuseNet_PyTorch-master/utils/text/visualization_palette.txt', 'r') as f:
        lines = f.read().splitlines()
        palette = []
    for line in lines:
        colors = line.split(', ')
        for color in colors:
            palette.append(float(color))
    palette = np.uint8(np.multiply(255, palette))
    train_seg_preds = train_seg_preds[0]
    image = np.hstack((np.uint8(train_seg_preds),np.uint8(train_seg_labels)))
    image = Image.fromarray(image)
    image.convert("P")
    image.putpalette(palette)
    image.save(os.path.join('/home/xinqichen/Desktop','prediction.png'))
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    # load your onnx model
    model = onnx.load('/home/xinqichen/projects/ncnn_demo/src/mobile_sim.onnx')
    sess = rt.InferenceSession("/home/xinqichen/projects/ncnn_demo/src/mobile_sim.onnx")
    input_name1 = sess.get_inputs()[0].name
    input_name2 = sess.get_inputs()[1].name
    seg_name = sess.get_outputs()[0].name
    # get the output
    pred_onnx = output("/home/xinqichen/Downloads/Firefox-Downloads/rgbd_cap/out/rgb/1.png", "/home/xinqichen/Downloads/Firefox-Downloads/rgbd_cap/out/depth/1.png")
    seg_predict(pred_onnx, "/home/xinqichen/Desktop/240*320/labels_resize/0.npy")
    print('Finish')
