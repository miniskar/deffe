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
import sys
import pdb

def RaiseException():
    exc_type, exc_value = sys.exc_info()[:2]
    print('Handling %s exception with message "%s" in %s' % \
          (exc_type.__name__, exc_value, 
           threading.current_thread().name))

class DeffeThreadData:
    def __init__(self, data=None, thread_end=False):
        self._thread_end = thread_end
        self._data = data

    def IsThreadEnd(self):
        return self._data

    def Get(self):
        return (self._data, self._thread_end)

    def GetData(self):
        return self._data

class DeffeThread:
    """ 
    The RunThread() method will run in the background forever until the StopThread() method is called
    """

    def __init__(self, method=None, arguments=None, single_run_flag=True, start_flag=False):
        #print("DeffeThread init")
        if method != None:
            self.InitThread(method, arguments, single_run_flag, start_flag)

    def InitThread(self, method, arguments, single_run_flag=True, start_flag=False):
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
        self.single_run_flag = single_run_flag
        self.method = method
        self.arguments = arguments
        self.thread = threading.Thread(target=self.RunThread, args=())
        self.thread.daemon = True  # Daemonize thread
        self.thread_status = self.INIT
        if start_flag:
            self.StartThread()


    def Connect(source, dest_list, name_list):
        if type(name_list) != list:
            name_list = [name_list]
        if type(dest_list) != list:
            dest_list = [dest_list]
        for name in name_list:
            for dest in dest_list:
                queue = Queue()
                if name not in source.out_ports:
                    source.out_ports[name] = []
                source.out_ports[name].append(queue)
                dest.in_ports[name] = queue

    def PutAll(self, data_hash):
        for k, v in data_hash.items():
            self.Put(k, v)

    def IsEmpty(self, port):
        return self.in_ports[port].empty()

    def GetAll(self, block_flag=True):
        data_hash = {}
        for k in self.in_ports.keys():
            #print("Getting data for k:"+k)
            data_hash[k] = self.Get(k, block_flag)
        return data_hash

    def SendEnd(self):
        for k in self.out_ports.keys():
            self.Put(k, DeffeThreadData(None, True))

    def Put(self, port_name, data, block_flag=True):
        caller_name = sys._getframe().f_back.f_code.co_name
        for port in self.out_ports[port_name]:
            #print(caller_name+": ********* Placing data:"+port_name)
            port.put(data, block_flag)
            #print(caller_name+": ********* Placed data successfully:"+port_name)

    def Get(self, port, block_flag=True):
        caller_name = sys._getframe().f_back.f_code.co_name
        #print(caller_name+": ********** Getting data:"+port)
        data = self.in_ports[port].get(block_flag)
        #print(caller_name+": ********** Got data successfully:"+port)
        return data

    def StartThread(self):
        self.thread_status = self.RUN
        self.thread.start()

    def RunThread(self):
        """ Thread runs forever """
        try:
            while not self.stop_thread:
                # Do something
                #print("Doing something imporant in the background")
                self.method(*self.arguments)
                if self.single_run_flag:
                    self.stop_thread = True
            self.thread_status = self.END
        except:
            RaiseException()

    def JoinThread(self):
        if self.thread_status == self.RUN:
            self.thread.join()

    def StopThread(self, join_flag=False):
        self.stop_thread = True
        #print("**** Joining thread *****"+str(join_flag))
        if join_flag:
            self.JoinThread()

class Producer(DeffeThread):
    def __init__(self, tag, count):
        self.tag = tag
        self.count = count
        DeffeThread.InitThread(self, self.run, ())

    def run(self):
        for i in range(self.count):
            self.Put('samples', i+100)
            #print(self.tag+": Sent data:"+str(i+100))
        self.StopThread(False)

class Consumer(DeffeThread):
    def __init__(self, tag, count):
        self.tag = tag
        self.count = count
        DeffeThread.InitThread(self, self.run, ())

    def run(self):
        for i in range(self.count):
            data = self.Get('samples')
            #print(self.tag+": Received data:"+str(data))
        self.StopThread(False)

def test1(count=100):
    example1 = Producer("Prod", count)
    example2 = Consumer("Cons", count)
    DeffeThread.Connect(example1, example2, "samples")
    example2.StartThread()
    example1.StartThread()
    print("Bye")
    example1.StopThread(True)
    example2.StopThread(True)

class ProdCons:
    def __init__(self, tag, count=10):
        self.tag = tag
        self.count = count

    def ConsRun(self):
        for i in range(self.count):
            data = self.cons_obj.Get('samples')
            print(self.tag+": Received data:"+str(data))
        self.cons_obj.StopThread(False)
        
    def ProdRun(self):
        for i in range(self.count):
            self.prod_obj.Put('samples', i+100)
            print(self.tag+": Sent data:"+str(i+100))
        self.prod_obj.StopThread(False)

    def RunSameObject(self):
        self.prod_obj = DeffeThread(self.ProdRun, ())
        self.cons_obj = DeffeThread(self.ConsRun, ())
        DeffeThread.Connect(self.prod_obj, self.cons_obj, "samples")
        self.prod_obj.StartThread()
        self.cons_obj.StartThread()
        self.prod_obj.JoinThread()
        self.cons_obj.JoinThread()

def test2(count):
    example = ProdCons("example", count)
    example.RunSameObject()

if __name__ == "__main__":
    #test1(10)
    test2(10)
