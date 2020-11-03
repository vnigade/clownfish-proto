
class FrameCache():
    def __init__(self):
        self.cached_data = None
        self.cached_meta = None

    def reset(self):
        self.cached_data = None
        self.cached_meta = None

    def update(self, meta, data):
        self.cached_data = data
        self.cached_meta = meta

    def get_overlap_frames(self, meta):
        if self.cached_meta is None:
            return None
        cached_start_frame = self.cached_meta[0]
        cached_end_frame = self.cached_meta[1]
        start_frame = meta[0]
        end_frame = meta[1]
        # Should always be the same or next window
        assert start_frame >= cached_start_frame
        assert end_frame >= cached_end_frame
        if start_frame > cached_end_frame:
            return None
        return (start_frame, cached_end_frame)

    @staticmethod
    def frame2array_index(frame_index, array_meta):

        array_start_index = frame_index[0] - array_meta[0]
        array_end_index = frame_index[1] - array_meta[0]
        # print("frame2array:", frame_index, array_meta, array_start_index, array_end_index)
        assert array_start_index >= 0 and array_start_index <= array_end_index, "Array start " \
            "index is not correct ({},{})".format(
                array_start_index, array_end_index)
        assert array_end_index <= array_meta[1], "Array end index " \
            "is not correcti {}, {}".format(array_end_index, array_meta[1])
        return (array_start_index, array_end_index)

    def get_cached_data(self, overlap_frames):
        # overlap_frames = _overlap(meta)
        tensor_index = self.frame2array_index(overlap_frames, self.cached_meta)
        # print("Tensor index:", tensor_index)
        tensor_size = self.cached_data.shape[2]
        # print("Tensor size:", tensor_size)
        assert tensor_index[0] <= tensor_index[1], "Tensor index {}".format(
            tensor_index)
        assert tensor_index[0] >= 0 and tensor_index[0] < tensor_size
        assert tensor_index[1] >= 0 and tensor_index[1] < tensor_size

        return self.cached_data[:, :, tensor_index[0]:(tensor_index[1]+1), :, :]
