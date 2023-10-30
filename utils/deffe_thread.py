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
import re

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

all_queues = []
def GetQueues():
    return all_queues

class DeffeThread:
    """ 
    The RunThread() method will run in the background forever until the StopThread() method is called
    """

    def __init__(self, method=None, arguments=None, single_run_flag=True, start_flag=False, tag=''):
        #print("DeffeThread init")
        if method != None:
            self.InitThread(method, arguments, single_run_flag, start_flag, tag)

    def InitThread(self, method, arguments, single_run_flag=True, start_flag=False, tag=''):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.INIT=0
        self.RUN=1
        self.END=2
        self.tag = tag
        self.in_ports = {}
        self.out_ports = {}
        self.stop_thread = False
        self.single_run_flag = single_run_flag
        self.method = method
        self.arguments = arguments
        self.thread = threading.Thread(target=self.RunThread, args=())
        self.thread.setName(method.__name__)
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
            source_out_name = name
            dest_out_name = name
            if re.search(r'::', name):
                name_split = name.split(r'::')
                source_out_name = name_split[0]
                dest_out_name = name_split[1]
            source_th_name = source.method.__name__
            for dest in dest_list:
                queue = Queue(maxsize=2)
                if source_out_name not in source.out_ports:
                    source.out_ports[source_out_name] = []
                source.out_ports[source_out_name].append(queue)
                dest.in_ports[dest_out_name] = queue
                dest_th_name = dest.method.__name__
                all_queues.append([queue, source_th_name, dest_th_name, source_out_name, dest_out_name])

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

    def SendEnd(self, key=None, out_data=None):
        if key == None:
            for k in self.out_ports.keys():
                self.Put(k, DeffeThreadData(out_data, True))
        else:
            self.Put(key, DeffeThreadData(out_data, True))

    def Put(self, port_name, data, block_flag=True):
        caller_name = sys._getframe().f_back.f_code.co_name
        for port in self.out_ports[port_name]:
            print(caller_name+f":{self.tag}: ********* Placing data:{port_name}")
            port.put(data, block_flag)
            print(caller_name+f":{self.tag}: ********* Placed data successfully:{port_name}")

    def Get(self, port, block_flag=True):
        caller_name = sys._getframe().f_back.f_code.co_name
        print(caller_name+f":{self.tag}: ********** Getting data:"+port)
        data = self.in_ports[port].get(block_flag)
        print(caller_name+f":{self.tag} ********** Got data successfully:"+port)
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

class ProdConsBase:
    def __init__(self, tag, count=10):
        self.tag = tag
        self.count = count

    def Initialize(self, tag, count=10):
        self.tag = tag
        self.count = count

    def ConsRun(self):
        for i in range(self.count):
            data = self.cons_obj.Get('samples')
            print(self.tag+": Received data1:"+str(data))
        self.cons_obj.StopThread(False)
        
    def ProdRun(self):
        for i in range(self.count):
            self.prod_obj.Put('samples', i+100)
            print(self.tag+": Sent data1:"+str(i+100))
        self.prod_obj.StopThread(False)

    def RunSameObject(self):
        self.prod_obj = DeffeThread(self.ProdRun, ())
        self.cons_obj = DeffeThread(self.ConsRun, ())
        DeffeThread.Connect(self.prod_obj, self.cons_obj, "samples")
        self.prod_obj.StartThread()
        self.cons_obj.StartThread()
        self.prod_obj.JoinThread()
        self.cons_obj.JoinThread()

class ProdCons(ProdConsBase):
    def __init__(self, tag, count=10):
        ProdConsBase.Initialize(self, tag, count)
    
    def ConsRun(self):
        for i in range(self.count):
            data = self.cons_obj.Get('samples')
            print(self.tag+": Received data2:"+str(data))
        self.cons_obj.StopThread(False)
        
    def ProdRun(self):
        for i in range(self.count):
            self.prod_obj.Put('samples', i+100)
            print(self.tag+": Sent data2:"+str(i+100))
        self.prod_obj.StopThread(False)


def test2(count):
    example = ProdCons("example", count)
    example.RunSameObject()

if __name__ == "__main__":
    #test1(10)
    test2(10)
