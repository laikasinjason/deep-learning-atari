
import random
class RingBuf:
    def __init__(self, size):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY
        # whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. 
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        self.size = size
        
    def append(self, element):
        self.data[self.end] = element
        
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
        
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def isFull(self):
        return len(self) == self.size
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
            
    def sample_batch(self, batch_size):
        result = []
        for x in range(batch_size):
            # generate random number between 1 and self.size-2
            result.append(self[random.randint(1,self.size-2)])
        return (np.asarray(result)).transpose()