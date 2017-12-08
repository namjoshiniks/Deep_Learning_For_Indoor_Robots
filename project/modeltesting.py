import numpy as np
import matplotlib.pyplot as plt
import sys

# Make sure that caffe is on the python path:
caffe_root = '/home/nvidia/caffe' # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
#MODEL_FILE = './examples/cifar10/cifar10_quick.prototxt' //
#deploy.prototxt file for our model

MODEL_FILE = '/home/nvidia/caffe/models/bvlc_reference_caffenet/deploy.prototxt' 
PRETRAINED = '/home/nvidia/caffe/models/bvlc_reference_caffenet/caffenet_train_iter_6000.caffemodel'

caffe.set_mode_gpu()
#caffe.set_device(0)
net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=np.load('/home/nvidia/project/meanfile.npy').mean(1).mean(1),
			channel_swap=(2,1,0),
			raw_scale=255,
			image_dims=(227, 227))

print "Classifier Loaded\n"

#IMAGE_FILE = '/home/nvidia/project/dataset/val/closedlift.38.jpg'
#IMAGE_FILE = '/home/nvidia/project/dataset/finaltesting/chair/chair.test4.jpg'
#IMAGE_FILE = '/home/nvidia/project/dataset/finaltesting/openlift/openlift.test3.jpg'
#IMAGE_FILE = '/home/nvidia/project/dataset/finaltesting/closedlift/closedlift.test3.jpg'   #testinaccurate1 works here
IMAGE_FILE = '/home/nvidia/project/dataset/finaltesting/opendoor/opendoor.test3.jpg'
#IMAGE_FILE = '/home/nvidia/project/dataset/finaltesting/closeddoor/closeddoor.test3.jpg'
input_image = caffe.io.load_image(IMAGE_FILE)
plt.imshow(input_image)

prediction = net.predict([input_image]) 
# predict takes any number of
#images, and formats them for the Caffe net automatically
print 'prediction shape:', prediction[0].shape

plt.plot(prediction[0])
print 'predicted class:', prediction[0].argmax()
classNumber = int(prediction[0].argmax())
if classNumber == 0:
	print 'It is a Chair'
elif classNumber == 1:
	print 'It is an Open Lift'
elif classNumber == 2:
	print 'It is a Closed Lift'
elif classNumber == 3:
	print 'It is an Open Door'
elif classNumber == 4:
	print 'It is a Closed Door'
plt.show()
