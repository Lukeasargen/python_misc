
from math import sqrt, ceil
import skimage.segmentation
import matplotlib.pyplot as plt
from skimage import io

img = io.imread("frog4.jpg")

k_list = [50, 100, 200, 400, 600, 800, 1000, 1200, 1400]
# k_list = [100, 200, 300, 400]
scale = 4

n = ceil(sqrt(len(k_list)))
fig = plt.figure(figsize=(scale*n, scale*n))

for i in range(len(k_list)):
    k = k_list[i]
    mask = skimage.segmentation.felzenszwalb(img, scale=k)

    ax = fig.add_subplot(n, n, i+1)
    ax.imshow(mask, cmap='rainbow')
    ax.set_xlabel(f"k={k}")

fig.suptitle("Felsenszwalb Graph Based Segmentation")
plt.tight_layout()
plt.show()