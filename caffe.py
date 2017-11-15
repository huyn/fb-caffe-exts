#coding=utf-8
#加载必要的库
import numpy as np
import caffe
import sys,os
from scipy import misc
from PIL import Image


import sys
def print_all(obj):
    modulelist = dir(obj)
    length = len(modulelist)
    print('=================1')
    for i in range(0,length,1):
        print(modulelist[i])
    print('=================2')

#设置当前目录
caffe_root = '/home/huyaonan/caffe/python'
model_path = '/home/huyaonan/fb-caffe-exts/'
sys.path.insert(0, caffe_root)

net_file=model_path + 's528_nn/s528_nn.prototxt'
caffe_model=model_path + 's528_nn/s528_nn.caffemodel'

net = caffe.Net(net_file,caffe_model,caffe.TEST)



image = Image.open('test.jpg')


image = np.array(image)
image = np.array(image[..., ::-1])        # RGB -> BGR
image = image.transpose(2, 0, 1)            # (H, W, C) -> (C, H, W)
image = image.reshape((1, ) + image.shape)  # (C, H, W) -> (B, C, H, W)
#image = torch.from_numpy(image.float()
image = image.astype(np.float)

print('=================1')
print(image)
print('=================2')
image = image / 128 - 1
print(image)
print('=================3')

#img = img.transpose((2,0,1))
net.blobs['data'].data[...]=image



#net.blobs['data'].data[...] = transformer.preprocess('data',img)
print('forward start')
out = net.forward()
print('forward done!')
out = out['TanhBackward20'];
print(out)
#print(out.dtype)
#image = out.array.numpy()

#output_img = out[0].transpose((1, 2, 0))
#output_img = output_img[..., ::-1]
##output_img = np.asarray(output_img)
#output_img = np.reshape(output_img, [512, 512, 3])
##print(output_img)
#output_img = Image.fromarray(output_img)
#output_img.save('2.jpg')

output_img = (out[0] + 1) * 128
output_img = (output_img.transpose((1, 2, 0)) + 1)
output_img = output_img[..., ::-1]
print(output_img)
output_img = output_img.astype(np.uint8)
print(output_img)
output_img = Image.fromarray(output_img)
output_img.save('2.jpg')
print('reshape ok1!!')


#enhanced_image = np.reshape(out*255, [512, 512, 3])
#misc.imsave('2.jpg', enhanced_image)
#misc.imsave(caffe_root+'2.jpg', out)



