import numpy as np
from PIL import Image

def min_max(x):
    return "min : {}. max : {}".format( np.min(x), np.max(x) )

def normalize(x):
    return ( x - np.min(x) ) / ( np.max(x) - np.min(x) )


filename = "aerial_image.jpg"

image = Image.open(filename)
img = np.asarray(image) / 255.0

hsv = image.convert("HSV")
hsv = np.asarray(hsv) / 255.0

h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

print("h :", min_max(h))
print("s :", min_max(s))
print("v :", min_max(v))

total = np.sum(img, axis=2)

r = img[:,:,0] / total
g = img[:,:,1] / total
b = img[:,:,2] / total

print("r :", min_max(r))
print("g :", min_max(g))
print("b :", min_max(b))

exg = 2*g - r - b
print("exg :", min_max(exg))

exr = 1.4*r - g
print("exr :", min_max(exr))

civ = -0.881*g + 0.441*r + 0.385*b + 18.78745
print("civ :", min_max(civ))

exgr = exg - exr
print("exgr :", min_max(exgr))

ndi = (g-r) / (g+r)
print("ndi :", min_max(ndi))

grey = (0.2898*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]) / 255
print("grey :", min_max(grey))

# row1 = np.hstack((r, g, b))
row1 = np.hstack((h, s, v))
row2 = np.hstack((normalize(exg), normalize(exr), normalize(civ)))
row3 = np.hstack((normalize(exgr), normalize(ndi), normalize(grey)))

grid = np.vstack((row1, row2, row3))

im = Image.fromarray(grid*255)

im.show()

