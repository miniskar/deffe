class Queue:
    def __init__(self, size=2):
        self.rear_ = 0
        self.front_ = 0
        self.size_ = size+1
        self.data_ = [None for i in range(self.size_)]

    def Increment(self, val):
        val = (val + 1)%self.size_
        return val

    def Enqueue(self, data, blocking=True):
        while self.IsFull():
            if not blocking:
                return False
        rear = self.rear_
        front = self.front_
        self.data_[rear] = data
        rear = self.Increment(rear)
        self.rear_ = rear
        return True

    def Dequeue(self, blocking=True):
        while self.IsEmpty():
            if not blocking:
                return None
        rear = self.rear_
        front = self.front_
        data = self.data_[front]
        front = self.Increment(front)
        self.front_ = front
        return data

    def IsFull(self):
        rear = self.rear_
        front = self.front_
        if self.Increment(rear) == front:
            return True
        return False

    def IsEmpty(self): 
        rear = self.rear_
        front = self.front_
        if rear == front:
            return True
        return False
    
