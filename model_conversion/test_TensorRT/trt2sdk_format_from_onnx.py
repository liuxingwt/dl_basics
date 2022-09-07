import struct
# import tensorrt as trt

# TRT_LOGGER = trt.Logger()
# batch_size = 1
# deploy_file = 'net_deploy.prototxt'
# model_file = 'model_weight.caffemodel'
input_layer = 'input_1'
output_layers = ['output_1']

# builder = trt.Builder(TRT_LOGGER)
# network = builder.create_network()
# parser = trt.CaffeParser()

# builder.max_batch_size = batch_size
# #builder.max_workspace_size = common.GiB(1)
# #builder.int8_mode = True
# #builder.int8_calibrator = calib
# # Parse Caffe model
# model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=trt.float32)
# for output_layer in output_layers:
#     network.mark_output(model_tensors.find(output_layer))
# # Build engine and do int8 calibration.
# engine = builder.build_cuda_engine(network)

f = open("liveness_mnv2_0921_trt7130_cuda102_nx_fp16_sdk.trt", "wb")
# batchSize
f.write(struct.pack("I", 1))

# inputChannel
f.write(struct.pack("I", 3))

# inputHeight
f.write(struct.pack("I", 224))

# inputWidth
f.write(struct.pack("I", 448))

# inputTensorNum
f.write(struct.pack("I", 1))
f.write(struct.pack("I", len(input_layer)))
f.write(bytes(input_layer, "utf-8"))

# outputTensorNum
f.write(struct.pack("I", len(output_layers)))
for output_layer in output_layers:
    f.write(struct.pack("I", len(output_layer)))
    f.write(bytes(output_layer, "utf-8"))

# size
# s = engine.serialize()

target = "liveness_mnv2_0921_trt7130_cuda102_nx_fp16.trt"
s = open(target, 'rb').read()
f.write(struct.pack("I", len(s)))
f.write(s)
