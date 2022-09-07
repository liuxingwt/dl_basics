import caffe
import numpy as np

caffe.set_mode_cpu()

model_def = "./pfld.prototxt"
model_weights = "./pfld.caffemodel"

net = caffe.Net(model_def, model_weights, caffe.TEST)

# blob0 is model input name, you can get it from netscope
transformer = caffe.io.Transformer({'data':net.blobs['blob0'].data.shape})

# transpose input image: [h,w,c]->[c,h,w]
transformer.set_transpose('data', [2,0,1])
# rescale image to: [0 - 1]
transformer.set_raw_scale('data', 1)
# swap image channel: RGB->BGR
transformer.set_channel_swap('data', [2,1,0])

net.blobs['blob0'].reshape(1, 3, 112, 112) 
 
img = caffe.io.load_image('./0_37_Soccer_soccer_ball_37_45_0.png')
transformer_img = transformer.preprocess('data',img)

# print(transformer_img.shape)
# print(type(transformer_img))
# print(transformer_img[:,:5,:5])
net.blobs['blob0'].data[...] = transformer_img
output = net.forward()

print(output)
