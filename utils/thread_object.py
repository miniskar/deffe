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
import threading
import time
from queue import *
import pdb

class ThreadObject:
    """ 
    The Run() method will run in the background forever until the Stop() method is called
    """

    def __init__(self):
        print("ThreadObject init")
        None

    def Init(self, method, arguments, start_flag=False):
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
        self.thread = threading.Thread(target=self.Run, args=())
        self.thread.daemon = True  # Daemonize thread
        self.thread_status = self.INIT
        if start_flag:
            self.Start()

    def Connect(name, source, dest):
        queue = Queue()
        source.out_ports[name] = queue
        dest.in_ports[name] = queue

    def Start(self):
        self.thread_status = self.RUN
        self.thread.start()

    def Run(self):
        """ Thread runs forever """
        while not self.stop_thread:
            # Do something
            print("Doing something imporant in the background")
            self.method(*self.arguments)
        self.thread_status = self.END

    def Join(self):
        if self.thread_status == self.RUN:
            self.thread.join()

    def Stop(self, join_flag=True):
        self.stop_thread = True
        if join_flag:
            self.Join()

class Producer(ThreadObject):
    def __init__(self, tag):
        self.tag = tag
        ThreadObject.Init(self, self.run, ())

    def run(self):
        for i in range(10):
            self.out_ports['samples'].Enqueue(i+100)
            print(self.tag+": Sent data:"+str(i+100))

class Consumer(ThreadObject):
    def __init__(self, tag):
        self.tag = tag
        ThreadObject.Init(self, self.run, ())

    def run(self):
        for i in range(10):
            data = self.in_ports['samples'].Dequeue()
            print(self.tag+": Received data:"+str(data))


if __name__ == "__main__":
    example1 = Producer("Prod")
    example2 = Consumer("Cons")
    ThreadObject.Connect("samples", example1, example2)
    example2.Start()
    example1.Start()
    time.sleep(3)
    print("Checkpoint")
    time.sleep(4)
    print("Bye")
    example1.Stop()
    example2.Stop()
