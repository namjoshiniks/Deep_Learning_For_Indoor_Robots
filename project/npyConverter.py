import numpy as np
caffe_root = '/home/nvidia/caffe' # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( '/home/nvidia/project/meanfile.binaryproto' , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob) )
np.save('/home/nvidia/project/meanfile.npy', arr[0])
arr.shape
