
import random
from queue import Queue

def make_telem():
    data = {
        "lat": random.random()*90,
        "lon": random.random()*90,
        "alt": random.random()*90
    }
    return data

q = Queue()

amount = 5

for i in range(amount):
    out_filename = '{:04d}.jpg'.format(i)
    data = make_telem()
    item = [out_filename, data]
    q.put(item)

while not q.empty():
    item = q.get()
    print(item, type(item))