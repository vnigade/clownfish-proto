import threading
import time


class FrameReader(threading.Thread):
    def __init__(self, data_loader, timer=30, max_limit=64):
        threading.Thread.__init__(self)
        self._data_loader = data_loader
        self._data_list = {}
        self._data_len = len(data_loader)
        self.timer = timer * 1e-3  # ms to sec
        self.max_limit = max_limit
        self._lock = threading.Lock()
        self._cur_index = 0

    def run(self):
        while self._cur_index < self._data_len:
            with self._lock:
                if len(self._data_list) < self.max_limit:
                    data = self._data_loader[self._cur_index]
                    self._data_list[self._cur_index] = data
                    self._cur_index += 1
            time.sleep(self.timer)

    def __getitem__(self, index):
        if index >= self._data_len:
            raise IndexError
        while True:
            with self._lock:
                if index in self._data_list:
                    data = self._data_list[index]
                    del self._data_list[index]
                    # print("Returning frame data", index)
                    return data
            time.sleep(0.001)

    def __len__(self):
        return self._data_len
