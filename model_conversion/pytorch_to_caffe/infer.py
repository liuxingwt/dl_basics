import argparse
import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from model import PFLDInference

cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# load backbone and parameters
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    # torch.module keys and values 
    # print(checkpoint['plfd_backbone'].keys())
    
    # print(pfld_backbone.state_dict().keys())
    pfld_backbone = PFLDInference().to(device)
    pfld_backbone.load_state_dict(checkpoint['plfd_backbone'])
    return pfld_backbone

# infer by pfld
def infer(pfld_backbone, image_path, save_path):    
    pfld_backbone.eval()
    with torch.no_grad():
        img_origin = cv2.imread(image_path)
        img = transforms.Compose([transforms.ToTensor()])(img_origin)
        img = img.reshape((1,3,112,112))
        img = img.to(device)
        
        print("input image: ", img[0,:,:5,:5])
        _, landmarks = pfld_backbone(img)
        landmarks = landmarks.cpu().numpy()
        landmarks = landmarks.reshape(landmarks.shape[0], -1, 2) 
        print("landmarks:", landmarks)
        # convert pre_landmark to numpy.ndarray type, shape=(98,2), element is type int
        # draw landmark in original image and save
        pre_landmark = landmarks[0] * [112, 112]
        pre_landmark = pre_landmark.astype(np.int32)
        img_clone = img_origin.copy()
        for (x, y) in pre_landmark:
            cv2.circle(img_clone, (x, y), 1, (255,0,0),-1)
        cv2.imwrite(save_path, img_clone)
    return

if __name__ == "__main__":
    model_path = './checkpoint.pth.tar'
    pfld_backbone = load_model(model_path)

    image_path='./0_37_Soccer_soccer_ball_37_45_0.png'
    save_path='./result_landmark.jpg'
    infer(pfld_backbone, image_path, save_path)
