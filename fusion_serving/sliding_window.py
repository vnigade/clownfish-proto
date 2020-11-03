from collections import namedtuple

WindowMeta = namedtuple('WindowMeta', 'start_frame end_frame id')


class SlidingWindow():
    def __init__(self, opt):
        self.window_size = opt.window_size
        self.window_stride = opt.window_stride
        self.window = []
        self.frame_ids = []

    def reset(self):
        self.window.clear()
        self.frame_ids.clear()

    def append(self, frame_id, frame):
        assert len(self.window) < self.window_size, "Window is already full"
        self.window.append(frame)
        self.frame_ids.append(frame_id)

    def next(self):
        window = None
        window_meta = None
        if len(self.window) == self.window_size:
            window = self.window
            window_id = self.frame_ids[0] // self.window_stride
            window_meta = WindowMeta(
                self.frame_ids[0], self.frame_ids[-1], window_id)
            self.window = self.window[self.window_stride:]
            self.frame_ids = self.frame_ids[self.window_stride:]
            assert len(self.window) == (self.window_size - self.window_stride)
            assert len(self.frame_ids) == (
                self.window_size - self.window_stride)

        return window, window_meta
