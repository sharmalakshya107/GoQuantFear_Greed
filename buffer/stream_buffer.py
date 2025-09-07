from queue import Queue
from threading import Lock

class StreamBuffer:
    def __init__(self, max_size=500):
        self.buffer = Queue(maxsize=max_size)
        self.lock = Lock()

    def push(self, data):
        with self.lock:
            if self.buffer.full():
                self.buffer.get()  # Remove oldest
            self.buffer.put(data)

    def pull_all(self):
        with self.lock:
            items = []
            while not self.buffer.empty():
                items.append(self.buffer.get())
            return items

    def size(self):
        with self.lock:
            return self.buffer.qsize()
