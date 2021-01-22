import os
import psutil

print("psutil.virtual_memory() :", psutil.virtual_memory())  # physical memory usage
print('memory % used:', psutil.virtual_memory()[2])
