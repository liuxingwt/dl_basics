import onnxruntime
import cv2
import numpy as np

img = cv2.imdecode(np.fromfile('example.jpg',dtype=np.uint8),-1)
img = cv2.resize(img, (112,112))
img = np.expand_dims(img,axis=0).astype(np.float32)/255
img = img.transpose(0,3,1,2) #格式 Batch, Chanel, Height, Width:

ort_session = onnxruntime.InferenceSession('pfld.onnx')
ort_inputs = {ort_session.get_inputs()[0].name: (img),} #类似tensorflow的传入数据，有几个输入就写几个

print(ort_session.get_inputs()[0].name)
for name in ort_session.get_outputs():
    print(name.name)
ort_outs = ort_session.run(None, ort_inputs)

#ort_outs = np.array(ort_outs)
print(ort_outs[0][0])
#mask = np.argmax(ort_outs[0], 1).squeeze().astype(np.int8)
#cv2.imwrite("result.jpg",mask*255)
