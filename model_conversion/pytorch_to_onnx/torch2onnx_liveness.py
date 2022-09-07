import os
import argparse
import onnx
from torch.autograd import Variable
import torch
from termcolor import cprint
from collections import OrderedDict

#import onnxsim
#from models.pfld import PFLDInference
from quality_mobilenet_v2 import quality_mobilenet_v2


def load_dict(args, model, insert_args=None):
    if os.path.isfile(args.resume):
        cprint('=> loading pth from {} ...'.format(args.resume), 'grey')
        checkpoint = torch.load(args.resume)
        _state_dict = clean_dict(model, checkpoint['state_dict'], insert_args=insert_args)
        model.load_state_dict(_state_dict)
        # delete to release more space
        del checkpoint
        del _state_dict
    else:
        print("=> No checkpoint found at '{}'".format(args.resume))
    return model

def clean_dict(model, state_dict, insert_args=None):
    _state_dict = OrderedDict()
    print(list(state_dict.items())[0][0])
    print(list(model.state_dict().items())[0][0] )
    #exit(1)
    
    for k, v in state_dict.items():
        k_ = k.split('.')
        assert k_[0] == 'module'
        #k[1] = blob_name
        if insert_args is not None:
            k_.insert(insert_args[0], insert_args[1])
        
        k = '.'.join(k_[2:])

        if k in model.state_dict().keys() and \
           v.size() == model.state_dict()[k].size():
            _state_dict[k] = v
            cprint(' : successfully load {} - {}'.format(k, v.shape), 'green')
        else:
            try:
                _state_dict[k] = model.state_dict()[k]
                cprint(' : ignore {} - {}'.format(k, v.shape), 'yellow')
            except:
                cprint(' : delete {} - {}'.format(k, v.shape), 'yellow')

    for k, v in model.state_dict().items():
        if k in _state_dict.keys():
            continue
        _state_dict[k] = v
    return _state_dict


def main():
    print("=====> load pytorch checkpoint...")
    net = quality_mobilenet_v2()
    net.eval()
    net = load_dict(args, net)
    print("bachbone:", net)

    print("=====> convert pytorch model to onnx...")
    dummy_input = Variable(torch.randn(1, 3, 224, 448)) 
    input_names = ["input_1"]
    output_names = [ "output_1" ]
    torch.onnx.export(net, dummy_input, args.onnx_model, verbose=True, input_names=input_names, output_names=output_names)


    print("====> check onnx model...")
    model = onnx.load(args.onnx_model)
    onnx.checker.check_model(model)
    print("checking success!!!!!!1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pytorch2onnx')
    parser.add_argument('--resume', default="./iter_0000507890.pth")
    parser.add_argument('--onnx_model', default="./liveness.onnx")
    parser.add_argument('--onnx_model_sim', help='Output ONNX model', default="./pfld-sim.onnx")
    args = parser.parse_args()
    main()
