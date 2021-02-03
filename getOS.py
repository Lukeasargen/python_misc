
# the path
import time
import os.path
import platform as pl
import pkg_resources
import socket
import getpass
import multiprocessing
import os
import platform
import sys
for p in sys.path:
    print(p)


print("System name : ", os.name)
print("System : ", platform.system())
print("Version : ", platform.release())
print("Current File Name : ", os.path.realpath(__file__))

print("CPU usage : ", multiprocessing.cpu_count())

print("Username : ", getpass.getuser())

print("IP Address : ", [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2]
                                     if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)),
                                                                           s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET,
                                                                                                                                  socket.SOCK_DGRAM)]][0][1]]) if l][0][0])

#import cProfile
# cProfile.run('main()')

print("Last modified: %s" % time.ctime(os.path.getmtime("getOS.py")))
print("Created: %s" % time.ctime(os.path.getctime("getOS.py")))

#import os
file_size = os.path.getsize("getOS.py")
print("\nThe size of getOS.py is : ", file_size, " Bytes")

#import time
print("System time : ", time.ctime())

#import os.path
#import time

print('File         :', __file__)
print('Access time  :', time.ctime(os.path.getatime(__file__)))
print('Modified time:', time.ctime(os.path.getmtime(__file__)))
print('Change time  :', time.ctime(os.path.getctime(__file__)))
print('Size         :', os.path.getsize(__file__))

str1 = "Python"
str2 = "Python"

print("\nMemory location of str1 =", hex(id(str1)))
print("Memory location of str2 =", hex(id(str2)))

print("\nLocally Installed Python Modules")
installed_packages = pkg_resources.working_set
installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
                                  for i in installed_packages])
for m in installed_packages_list:
    print(m)

print("\nOS Information")
os_profile = [
    'architecture',
    'linux_distribution',
    'mac_ver',
    'machine',
    'node',
    'platform',
    'processor',
    'python_build',
    'python_compiler',
    'python_version',
    'release',
    'system',
    'uname',
    'version',
]
for key in os_profile:
    if hasattr(pl, key):
        print(key + ": " + str(getattr(pl, key)()))
