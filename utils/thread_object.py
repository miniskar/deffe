## Copyright 2020 UT-Battelle, LLC.  See LICENSE.txt for more information.
###
 # @author Narasinga Rao Miniskar, Frank Liu, Dwaipayan Chakraborty, Jeffrey Vetter
 #         miniskarnr@ornl.gov
 # 
 # Modification:
 #              Baseline code
 # Date:        Apr, 2020
 #**************************************************************************
###
import threading
import time

class ThreadObject:
    """ 
    The Run() method will run in the background forever until the Stop() method is called
    """

    def __init__(self, method, arguments, start_flag=True):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.stop_thread = False
        self.method = method
        self.arguments = arguments
        self.thread = threading.Thread(target=self.Run, args=())
        self.thread.daemon = True                            # Daemonize thread
        if start_flag:
            self.Start()

    def Start(self):
        self.thread.start()

    def Run(self):
        """ Thread runs forever """
        while not self.stop_thread:
            # Do something
            print('Doing something imporant in the background')
            self.method(*self.arguments)

    def Join(self):
        self.thread.join()

    def Stop(self, join_flag=True):
        self.stop_thread = True
        if join_flag:
            self.Join()

def example_run(arg1, arg2):
    time.sleep(1)
    
if __name__ == "__main__":
    example = ThreadObject(example_run, ("Hello", "World"))
    time.sleep(3)
    print('Checkpoint')
    time.sleep(4)
    print('Bye')
    example.Stop()
