import grpc

import model_serving.protos.predict_pb2 as predict_pb2
import model_serving.protos.predict_pb2_grpc as predict_pb2_grpc
import torch
import pickle
import numpy as np
import cv2
from common import utils
import timeit
import PIL
import io
import threading
import queue
import time

_MAX_MSG_LENGTH = 100 * 1024 * 1024

START_FRAME, END_FRAME, ID = 0, 1, 2


class ModelServingClient(object):
    """
    Client for connecting to ModelServingServer
    """

    def __init__(self, host="localhost", port_number=9999, callback=None, stats=False, stats_name="model_client", compression_level=6):
        # configure the host and the
        # the port to which the client should connect
        # to.
        self.host = host
        self.server_port = port_number

        # instantiate a communication channel
        # options = [('grpc.max_receive_message_length', _MAX_MSG_LENGTH)]
        options = [('grpc.max_send_message_length', _MAX_MSG_LENGTH),
                   ('grpc.max_receive_message_length', _MAX_MSG_LENGTH)]
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port), options=options)
        # compression=grpc.Compression.Gzip)

        # bind the client to the server channel
        self.stub = predict_pb2_grpc.ModelServingStub(self.channel)

        # stats
        self.stats = None
        if stats:
            self.stats = utils.Stats(["preprocessing_time", "prediction_time",
                                      "postprocessing_time"], stats_name=stats_name)

        # cache
        self.window_meta_cache = None
        self.encoded_cache_list = None

        # Handle async prediction calls
        self._callback = callback
        self.outstanding_windows = []
        self.response_dict = {}
        self.response_lock = threading.Lock()
        self.cv_response_lock = threading.Condition(self.response_lock)
        self.throttle_limit = 16  # In windows.

        # data
        self.compression_level = compression_level

    def register_callback(self, callback):
        self._callback = callback

    def set_compression_level(self, compression_level):
        self.compression_level = compression_level

    def _get_encoded_cache_list(self, window_meta, inputs):
        byte_list = []
        if self.window_meta_cache is None:
            return byte_list
        assert (window_meta.end_frame - window_meta.start_frame) == \
            (self.window_meta_cache.end_frame -
             self.window_meta_cache.start_frame), print("Assertion in Cache list:1")
        assert window_meta.start_frame > self.window_meta_cache.start_frame, print(
            "Assertion in Cache list:2", window_meta, self.window_meta_cache)
        start_idx = window_meta.start_frame - \
            self.window_meta_cache.start_frame
        if start_idx > self.window_meta_cache.end_frame:
            # no overlapping frames
            return byte_list
        byte_list = self.encoded_cache_list[start_idx:]
        return byte_list

    def encode_numpy2img(self, window_meta, inputs):

        byte_list = self._get_encoded_cache_list(window_meta, inputs)
        # size_before =  inputs.nbytes
        size_after = 0
        for i in range(len(byte_list), len(inputs)):
            img = inputs[i]
            # size_after += buf.nbytes
            buf = io.BytesIO()
            img.save(buf, "PNG", compress_level=self.compression_level)
            byte_list.append(buf.getvalue())
        # print("Input size: {}/{}".format(size_after, size_before))

        return byte_list

    def serialize(self, window_meta, inputs):
        byte_list = self.encode_numpy2img(window_meta, inputs)
        start_idx = 0
        if self.window_meta_cache is not None and \
                window_meta.start_frame >= self.window_meta_cache.start_frame and \
                self.window_meta_cache.end_frame >= window_meta.start_frame:
            # Overlapping window
            start_idx = self.window_meta_cache.end_frame - \
                window_meta.start_frame + 1

        # update cache
        self.window_meta_cache = window_meta
        self.encoded_cache_list = byte_list
        return byte_list[start_idx:]

    def unserialize(self, output, output_size):
        output = np.frombuffer(output, dtype=np.float32).reshape(output_size)
        return output

    def _compute_buf_size(self, input_data):
        size = 0
        for output in input_data:
            size += len(output)
            print("One encoded image size: ", len(output))
        print("Total buffer size", size)

    def _process_response(self, future):
        model_output = future.result()
        window_meta = model_output.window_meta
        window_meta = (window_meta.start_frame,
                       window_meta.end_frame, window_meta.id)
        output_size = pickle.loads(model_output.output_size)
        output_data = self.unserialize(model_output.output_data, output_size)

        return (window_meta, output_data, output_size)

    def _handle_predict_response(self, future):
        '''
        Handle response from model server.
        params:
        future: Future object.
        '''
        output = self._process_response(future)
        window_id = output[0][ID]
        # print("Received window ", window_id)
        # Call upper layer callback. If this is the next expected result.
        # Use window_id sequence to know if this is the next expected result.
        try:
            self.cv_response_lock.acquire()
            self.response_dict[window_id] = output
            while len(self.outstanding_windows) > 0:
                outstanding_window_meta = self.outstanding_windows[0]
                window_id = outstanding_window_meta.id
                if window_id in self.response_dict:
                    window_meta, output_data, output_size = self.response_dict[window_id]
                    self._callback((window_meta, output_data, output_size))
                    del self.outstanding_windows[0]
                    del self.response_dict[window_id]
                else:
                    break
            self.cv_response_lock.notify()
        finally:
            self.cv_response_lock.release()

        return

    def _throttle(self):
        if self._callback is None:
            return
        try:
            self.cv_response_lock.acquire()
            while len(self.outstanding_windows) > self.throttle_limit:
                self.cv_response_lock.wait()
            self.cv_response_lock.notify()
        finally:
            self.cv_response_lock.release()

        return

    def predict(self, window_meta, input_data):
        """
        Client function to call the rpc for predict
        """
        # Serialize input
        start_time = timeit.default_timer()
        input_data = self.serialize(window_meta, input_data)
        # self._compute_buf_size(input_data)
        input_size = pickle.dumps((len(input_data), ))
        window_meta = predict_pb2.WindowMeta(
            id=window_meta.id, start_frame=window_meta.start_frame, end_frame=window_meta.end_frame)
        model_input = predict_pb2.ModelInput(
            window_meta=window_meta, input_size=input_size)
        model_input.input_data.extend(input_data)
        preprocessing_time = (timeit.default_timer() - start_time) * 1000

        # check throttle.
        self._throttle()

        # predict remote
        start_time = timeit.default_timer()
        output_future = self.stub.predict.future(model_input)
        output = None
        if self._callback is None:
            output = self._process_response(output_future)
        else:
            with self.response_lock:
                self.outstanding_windows.append(window_meta)
            output_future.add_done_callback(self._handle_predict_response)
        # print("Sent window id:", window_meta.id)
        prediction_time = (timeit.default_timer() - start_time) * 1000

        # unserialize output
        start_time = timeit.default_timer()
        postprocessing_time = (timeit.default_timer() - start_time) * 1000

        if self.stats is not None:
            self.stats.update({"preprocessing_time": preprocessing_time,
                               "prediction_time": prediction_time,
                               "postprocessing_time": postprocessing_time})

        return output

    def reset(self):
        empty = predict_pb2.Empty()
        response = self.stub.reset(empty)
        self.encoded_cache_list = None
        self.window_meta_cache = None

    def __call__(self, window_meta, input_data):
        return self.predict(window_meta, input_data)
