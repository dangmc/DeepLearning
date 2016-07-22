import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

pixel = 255.0

path_image = "/home/dangmc/Documents/DeepLearning/NotMNIST/notMNIST_small/A/MDEtMDEtMDAudHRm.png"

image = mpimg.imread(path_image)

print image.shape
plt.imshow(image, cmap="Greys")

a = 5//2
print a

plt.show()