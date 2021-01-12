import io
import torch
import torch.onnx
from models.fusenet_model import FuseNet
import torchvision
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def test():
  
  pthfile = './checkpoints/upsample_best_model_1.pth.tar'
  model = torch.load(pthfile, map_location='cpu')
  fusenet = FuseNet(num_labels = 6)
  fusenet.load_state_dict(model['state_dict'],strict=True)
  modnet = fusenet.cpu()
  modnet.eval()
  print('end!\n')
  #convert_model(modnet)
  rgb_inputs= torch.randn(1, 3, 320, 240)
  depth_inputs = torch.randn(1, 1, 320, 240)
  input_names = [ "rgb_inputs" ] + [ "depth_inputs" ]
  output_names = [ "y"]


  inputdata = (rgb_inputs, depth_inputs)
  torch_out = torch.onnx.export(modnet, inputdata, "./mobile_best.onnx", export_params=True,opset_version=11, verbose=False, 
                                input_names=input_names, output_names=output_names, keep_initializers_as_inputs=True,training=False)
  #operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK

  print('done!!')

if __name__ == "__main__":
  test()

