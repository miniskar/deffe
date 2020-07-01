## Copyright 2020 UT-Battelle, LLC.  See LICENSE.txt for more information.
###
# @author Narasinga Rao Miniskar, Frank Liu, Dwaipayan Chakraborty, Jeffrey Vetter
#         miniskarnr@ornl.gov
#
# Modification:
#              Baseline code
# Date:        Apr, 2020
# **************************************************************************
###
import threading, queue
import time
from queue import *
import pdb

class DeffeThreadData:
    def __init__(self, data=None, thread_end=False):
        self._thread_end = thread_end
        self._data = data

    def IsThreadEnd(self):
        return self._data

    def GetData(self):
        return self._data

class DeffeThread:
    """ 
    The RunThread() method will run in the background forever until the StopThread() method is called
    """

    def __init__(self):
        print("DeffeThread init")
        None

    def InitThread(self, method, arguments, start_flag=False):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.INIT=0
        self.RUN=1
        self.END=2
        self.in_ports = {}
        self.out_ports = {}
        self.stop_thread = False
        self.method = method
        self.arguments = arguments
        self.thread = threading.Thread(target=self.RunThread, args=())
        self.thread.daemon = True  # Daemonize thread
        self.thread_status = self.INIT
        if start_flag:
            self.StartThread()

    def Connect(name, source, dest):
        queue = Queue()
        source.out_ports[name] = queue
        dest.in_ports[name] = queue

    def StartThread(self):
        self.thread_status = self.RUN
        self.thread.start()

    def RunThread(self):
        """ Thread runs forever """
        while not self.stop_thread:
            # Do something
            print("Doing something imporant in the background")
            self.method(*self.arguments)
        self.thread_status = self.END

    def JoinThread(self):
        if self.thread_status == self.RUN:
            self.thread.join()

    def StopThread(self, join_flag=True):
        self.stop_thread = True
        if join_flag:
            self.JoinThread()

class Producer(DeffeThread):
    def __init__(self, tag):
        self.tag = tag
        DeffeThread.InitThread(self, self.run, ())

    def run(self):
        for i in range(10):
            self.out_ports['samples'].put(i+100, True)
            print(self.tag+": Sent data:"+str(i+100))

class Consumer(DeffeThread):
    def __init__(self, tag):
        self.tag = tag
        DeffeThread.InitThread(self, self.run, ())

    def run(self):
        for i in range(10):
            data = self.in_ports['samples'].get(True)
            print(self.tag+": Received data:"+str(data))


if __name__ == "__main__":
    example1 = Producer("Prod")
    example2 = Consumer("Cons")
    DeffeThread.Connect("samples", example1, example2)
    example2.StartThread()
    example1.StartThread()
    time.sleep(3)
    print("Checkpoint")
    time.sleep(4)
    print("Bye")
    example1.StopThread()
    example2.StopThread()
