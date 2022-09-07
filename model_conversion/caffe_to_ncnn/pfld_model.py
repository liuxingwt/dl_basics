import time
import numpy as np
import cv2
import ncnn
# from .model_store import get_model_file

class pfld:
    def __init__(self, target_size=112, num_threads=1, use_gpu=False):
        self.target_size = target_size
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        # Important: (original_value - mean_vals) * norm_vals, 0.003921=1/255
        self.mean_vals = [0, 0, 0]
        self.norm_vals = [0.003921, 0.003921, 0.003921 ]

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu
        
        # the ncnn model https://github.com/caishanli/pyncnn-assets/tree/master/models
        # set ncnn-fromat model file path: 
        self.net.load_param("./pfld.param")
        self.net.load_model("./pfld.bin")
            
    def __del__(self):
        self.net = None

    def __call__(self, image_path):
        # img_h = img.shape[0]
        # img_w = img.shape[1]
        # mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR, img.shape[1], img.shape[0], self.target_size, self.target_size)
        # mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)
        # img = cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),-1)

        img = cv2.imread(image_path)
        img = cv2.resize(img, (112,112))
        img_h, img_w = img.shape[:2]
        print("original image, input shape:",img.shape,  img)
        
        # switch channel, and add dimension
        mat_in = ncnn.Mat.from_pixels(img, ncnn.Mat.PixelType.PIXEL_BGR, img_w, img_h)
        # minus zero, and divide 255
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)
        print("After processing:", np.array(mat_in))

        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)
        # remember to change input name
        ex.input("blob0", mat_in)
        
        start = time.time()
        mat_out = ncnn.Mat()
        # remember to change ouput name
        ex.extract("fc_blob131", mat_out)
        end = time.time()
        print("time cost:", end- start)
        #printf("%d %d %d\n", mat_out.w, mat_out.h, mat_out.c)
        out = np.array(mat_out)
        print(out)
        return out
