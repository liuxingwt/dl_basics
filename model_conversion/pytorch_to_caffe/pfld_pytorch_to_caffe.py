import sys
sys.path.insert(0,'.')
sys.path.append("../")

import torch
from torch.autograd import Variable
from model import PFLDInference
import pytorch_to_caffe

if __name__=='__main__':
    name='pfld'
   
    # if cuda is avaliable, cancle the annotation
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    model_path = './checkpoint.pth'
    checkpoint = torch.load(model_path, map_location=device)
    net = PFLDInference().to(device)
    net.load_state_dict(checkpoint['plfd_backbone'])
    net.eval()

    input=Variable(torch.ones([1,3,112,112])).to(device)
    pytorch_to_caffe.trans_net(net,input,name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
