from skimage import data, segmentation, color, io
from minisom import MiniSom
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, help='input image path')
parser.add_argument('num_superpixel', type=int, help='number of segments')
parser.add_argument('compactness', type=int, help='compactness param of SLIC')
args = parser.parse_args()


#img = data.coffee()
img = io.imread(args.input_image)
labels = segmentation.slic(img, n_segments=args.num_superpixel, compactness=args.compactness)
out1 = color.label2rgb(labels, img, kind='avg')

io.imshow(out1)
io.show()

pixels = np.reshape(out1, (out1.shape[0] * out1.shape[1], 3)) / 255

print('training...')
som = MiniSom(2, 1, 3, sigma=1., learning_rate=0.2, neighborhood_function='bubble')
som.random_weights_init(pixels)
starting_weights = som.get_weights().copy()  # saving the starting weights
som.train_random(pixels, 5000, verbose=True)

print('quantization...')
qnt = som.quantization(pixels)  # quantize each pixels of the image
print('building new image...')
clustered = np.zeros(img.shape)
for i, q in enumerate(qnt):  # place the quantized values into a new image
    clustered[np.unravel_index(i, dims=(out1.shape[0], out1.shape[1]))] = q
print('done.')

io.imshow(clustered)
io.show()
io.imshow(starting_weights)
io.show()
io.imshow(som.get_weights())
io.show()
