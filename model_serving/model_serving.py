import grpc
from protos import predict_pb2
from protos import predict_pb2_grpc
from concurrent import futures
import timeit
import time
import pickle
import numpy as np
import torch
import cv2
import PIL
import io
from cache import FrameCache
import threading
import os

_MAX_MSG_LENGTH = 100 * 1024 * 1024
START_FRAME, END_FRAME, ID = 0, 1, 2


class ModelServingServicer(predict_pb2_grpc.ModelServingServicer):
    """
    gRPC server for ModelServing Service
    """

    def __init__(self, model, port_number=9999, max_workers=10, transforms=None, stats=None, cache=None, lock=None, cv_lock=None):
        self.server_port = port_number
        self.max_workers = max_workers
        self.model = model
        self.lock = lock
        self.cv_lock = cv_lock
        self.transforms = transforms
        self.stats = stats
        self.cache = cache
        torch.manual_seed(1)

    def unserialize(self, input_data, input_size, start_index):
        def _unserialize(_bytes):
            _bytes = io.BytesIO(_bytes)
            img = PIL.Image.open(_bytes)
            return img

        assert len(input_data) == input_size[0]
        frames = [_unserialize(input_data[i])
                  for i in range(start_index, len(input_data))]
        # frames = self._mp_pool.map(_unserialize, input_data[start_index:])
        return frames

    def serialize(self, tensor):
        return tensor.cpu().data.numpy().tobytes()

    def _wait_next_window(self, input_size, window_meta):
        # print("Cache address", hex(id(self.cache)), "my pid", os.getpid())
        if self.cache.cached_meta is None:
            return

        window_size = window_meta[END_FRAME] - window_meta[START_FRAME] + 1
        if input_size[0] == window_size:
            # no overlapping
            # print("No overlapping:")
            return
        try:
            self.cv_lock.acquire()
            while (self.cache.cached_meta[ID] + 1) != window_meta[ID]:
                # This is next window. Process it.
                self.cv_lock.wait()
            self.cv_lock.notify()
        finally:
            self.cv_lock.release()

    def pre_process(self, input, input_size, window_meta):
        # print("Preprocessing:", input_size, window_meta)
        window_size = window_meta[END_FRAME] - window_meta[START_FRAME] + 1
        if self.transforms is not None:
            overlap_frames = self.cache.get_overlap_frames(window_meta)
            # print("Overlap frames:", overlap_frames)
            if overlap_frames:
                overlap_data = self.cache.get_cached_data(overlap_frames)
                # print("Overlap data:", overlap_data.shape)
                window_overlap_frames = FrameCache.frame2array_index(
                    overlap_frames, window_meta)
                # print("Window overlap frames:", window_overlap_frames)
                new_start_frame = window_overlap_frames[1] + 1
                input_data = overlap_data
                # if new_start_frame < input.shape[0]:
                # if new_start_frame < len(input):
                if new_start_frame < window_size:
                    start_time = timeit.default_timer()
                    # input = self.unserialize(input, input_size, start_index=new_start_frame)
                    input = self.unserialize(input, input_size, start_index=0)
                    unserialize_time = (
                        timeit.default_timer() - start_time) * 1000
                    # new_input = self.transforms(input[new_start_frame:,:,:,:])
                    start_time = timeit.default_timer()
                    new_input = self.transforms(input)
                    transform_time = (
                        timeit.default_timer() - start_time) * 1000
                    # print("Stats:", "Unserialize:", unserialize_time, "Transform time:", transform_time)
                    input_data = torch.cat([overlap_data, new_input], dim=2)
                else:
                    print("Complete overlapping windows..")
                    raise NotImplemented()
            else:
                input = self.unserialize(input, input_size, start_index=0)
                input_data = self.transforms(input)

            self.cache.update(window_meta, input_data)
        return input_data

    def predict(self, request, context):
        """
        Implementation of the rpc predict declared in the proto
        file above.
        """
        # Input unserialization. Get the window_meta and input_data
        start_time = timeit.default_timer()
        window_meta_pb = request.window_meta
        input_size = pickle.loads(request.input_size)
        # print("Received window id", window_meta_pb.id)
        # Preprocess input
        window_meta = (window_meta_pb.start_frame,
                       window_meta_pb.end_frame, window_meta_pb.id)
        self._wait_next_window(input_size, window_meta)
        try:
            self.cv_lock.acquire()
            input_data = self.pre_process(
                request.input_data, input_size, window_meta)
            self.cv_lock.notify_all()
        finally:
            self.cv_lock.release()

        input_data = input_data.cuda(non_blocking=True)
        preprocessing_time = (timeit.default_timer() - start_time) * 1000

        # Model prediction
        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        with torch.no_grad():
            output = self.model(input_data)
        torch.cuda.synchronize()
        prediction_time = (timeit.default_timer() - start_time) * 1000

        # Output serialization
        start_time = timeit.default_timer()
        output_size = pickle.dumps(output.shape)
        output = self.serialize(output)
        result = {'window_meta': window_meta_pb,
                  'output_data': output, 'output_size': output_size}
        postprocessing_time = (timeit.default_timer() - start_time) * 1000

        if self.stats:
            self.stats.update({"preprocessing_time": preprocessing_time,
                               "prediction_time": prediction_time,
                               "postprocessing_time": postprocessing_time})
        # print("Sending window id", window_meta_pb.id)
        return predict_pb2.ModelOutput(**result)

    def reset(self, request, context):
        print("Reset..")
        self.cache.reset()
        return predict_pb2.Empty()

    def start_server(self):
        """
        Function which actually starts the gRPC server, and preps
        it for serving incoming connections
        """
        # declare a server object with desired number
        # of thread pool workers.
        options = [('grpc.max_send_message_length', _MAX_MSG_LENGTH),
                   ('grpc.max_receive_message_length', _MAX_MSG_LENGTH)]
        model_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers), options=options)
        # compression=grpc.Compression.Gzip)

        # Shared data structure
        self.cache = FrameCache()
        self.lock = threading.Lock()
        self.cv_lock = threading.Condition(self.lock)

        # This line can be ignored
        predict_pb2_grpc.add_ModelServingServicer_to_server(
            ModelServingServicer(model=self.model,
                                 port_number=self.server_port,
                                 transforms=self.transforms,
                                 stats=self.stats,
                                 cache=self.cache,
                                 lock=self.lock,
                                 cv_lock=self.cv_lock), model_server)

        # bind the server to the port defined above
        model_server.add_insecure_port('[::]:{}'.format(self.server_port))

        # start the server
        model_server.start()
        print('Model Serving Server running ...')

        try:
            while True:
                time.sleep(60*60*60)
        except KeyboardInterrupt:
            model_server.stop(0)
            print('Model Serving Server Stopped ...')
