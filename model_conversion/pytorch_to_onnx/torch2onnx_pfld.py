import os
import argparse
import onnx
from models.pfld import PFLDInference
from torch.autograd import Variable
import torch
import onnxsim

parser = argparse.ArgumentParser(description='pytorch2onnx')
parser.add_argument('--torch_model', default="./pytorch_model/checkpoint.pth.tar")
parser.add_argument('--onnx_model', default="./pfld.onnx")
parser.add_argument('--onnx_model_sim', help='Output ONNX model', default="./pfld-sim.onnx")
args = parser.parse_args()

print("=====> load pytorch checkpoint...")
checkpoint = torch.load(args.torch_model, map_location=torch.device('cpu'))
plfd_backbone = PFLDInference()
plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
print("PFLD bachbone:", plfd_backbone)

print("=====> convert pytorch model to onnx...")
dummy_input = Variable(torch.randn(1, 3, 112, 112)) 
input_names = ["input_1"]
output_names = [ "output_1" ]
torch.onnx.export(plfd_backbone, dummy_input, args.onnx_model, verbose=True, input_names=input_names, output_names=output_names)


print("====> check onnx model...")
model = onnx.load(args.onnx_model)
onnx.checker.check_model(model)
print("checking success!!!!!!1")

print("====> Simplifying...")
model_opt = onnxsim.simplify(args.onnx_model)
# print("model_opt", model_opt)
onnx.save(model_opt, args.onnx_model_sim)
print("onnx model simplify Ok!")
