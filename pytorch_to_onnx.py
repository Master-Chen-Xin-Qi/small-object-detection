import io
import torch
import torch.onnx
from models.fusenet_model import FuseNet

 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test():
  #model = FuseNet(num_labels=6)
  
  pthfile = './checkpoints/mobile_best_model_cpu_320x240.pth.tar'
  model = torch.load(pthfile)

  model.load_state_dict(loaded_model['state_dict'])
  # model = model.to(device)

  #data type nchw
  dummy_input1 = torch.randn(1, 3, 240, 320)
  dummy_input2 = torch.randn(1, 1, 240, 320)
  # dummy_input2 = torch.randn(1, 3, 64, 64)
  # dummy_input3 = torch.randn(1, 3, 64, 64)
  input_names = [ "rgb", "depth"]
  output_names = [ "output_class" ]

  # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
  torch.onnx._export(model, (dummy_input1,dummy_input2), "FuseNet_test.onnx", opset_version = 9, verbose=True, input_names=input_names, output_names=output_names)

if __name__ == "__main__":
  test()
