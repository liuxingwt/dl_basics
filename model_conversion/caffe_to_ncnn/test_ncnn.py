import sys
import cv2
import numpy as np
import ncnn
# from ncnn.model_zoo import get_model
# from ncnn.utils import print_topk
from pfld_model import pfld

use_gpu = False
if ncnn.build_with_gpu():
    use_gpu = True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: %s [imagepath]\n"%(sys.argv[0]))
        sys.exit(0)

    imagepath = sys.argv[1]

    # m = cv2.imread(imagepath)
    # if m is None:
    #     print("cv2.imread %s failed\n"%(imagepath))
    #     sys.exit(0)

    if use_gpu:
        ncnn.create_gpu_instance()

    # net = get_model('squeezenet', num_threads=4, use_gpu=use_gpu)
    net = pfld(num_threads=4, use_gpu=use_gpu)
    result = net(imagepath)

    if use_gpu:
        ncnn.destroy_gpu_instance()
        
    # print_topk(result, 5)
