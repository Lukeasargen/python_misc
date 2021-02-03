
import time
from threading import Thread, Event


class ClassA(object):
    def __init__(self):
        print("Class A enter")
        super(ClassA, self).__init__()
        print("Class A exit")  

    def startB(self):
        c = 0
        while True:
            c +=1
            try:
                print("ClassA :", c)
            except KeyboardInterrupt:
                break
            time.sleep(0.5)

class ClassB(ClassA, Thread):
    def __init__(self, *args, **kwargs):
        print("Class B enter")
        super(ClassB, self).__init__(*args, **kwargs)
        Thread.__init__(self)
        print("Class B exit")
        self.__flag = Event()  # is used to suspend the thread's identity
        self.__flag.set()  # Start true
        self.__running = Event() # to stop the thread's identity
        self.__running.set ()   # set true

    def run(self):
        print("run")
        while self.__running.isSet():
            print("wait :", self.__flag.wait(), time.time())
            print("flag :", self.__flag.isSet())
            self.__flag.wait()   # returns immediately if true
            time.sleep(0.25)

    def pause(self):
        print("pause")
        self.__flag.clear() # set false, block thread
        print("flag :", self.__flag.isSet())

    def resume(self):
        print("resume")
        self.__flag.set()  # set true, stop block
        print("flag :", self.__flag.isSet())

    def stop(self):
        print("stop")
        self.__flag.set() # sto block to close thread
        self.__running.clear() # set false  


if __name__ == "__main__":
    x = ClassB()
    x.run()
    time.sleep(2)
    x.pause()
    time.sleep(2)
    x.resume(2)
    time.sleep(2)
    x.stop()
    time.sleep()
    print("exit main")
