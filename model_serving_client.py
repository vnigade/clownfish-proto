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
from multiprocessing.pool import ThreadPool

_MAX_MSG_LENGTH = 100 * 1024 * 1024


def tuple_to_int(tuple):
    num = sum(tuple[i] << (i * 8) for i in range(len(tuple)))
    return num


START_FRAME, END_FRAME, ID = 0, 1, 2


def _png_encoding(input):
    buf = io.BytesIO()
    # @todo speedup the encoding process for the higher compression level.
    _compress_level = 1
    # print("width, height = {}, {}".format(input.size, input.mode))
    input.save(buf, "PNG", compress_level=_compress_level)
    bytes_buf = buf.getvalue()
    # print("Compress level", _compress_level, len(bytes_buf))
    return bytes_buf


class ModelServingClient(object):
    """
    Client for connecting to ModelServingServer
    """

    def __init__(self, host="localhost", port_number=9999, stats=False, stats_name="model_client", cache_data=True, callback=None):
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
        self.cache_data = cache_data
        self._callback = callback

        # ThreadPool
        self._mp_pool = None
        self.use_mp = False

    def set_cache_data(self, value):
        self.cache_data = value

    def set_callback(self, callback):
        self._callback = callback

    def get_encoded_cache_list(self, window_meta, inputs):
        byte_list = []
        if self.window_meta_cache is None:
            return byte_list
        assert (window_meta[END_FRAME] - window_meta[START_FRAME]) == \
            (self.window_meta_cache[END_FRAME] -
             self.window_meta_cache[START_FRAME])
        assert window_meta[START_FRAME] > self.window_meta_cache[START_FRAME], print(
            "Assertion", window_meta, self.window_meta_cache)
        start_idx = window_meta[START_FRAME] - \
            self.window_meta_cache[START_FRAME]
        if start_idx > self.window_meta_cache[END_FRAME]:
            # no overlapping frames
            return byte_list
        byte_list = self.encoded_cache_list[start_idx:]
        return byte_list

    def encode_numpy2img(self, window_meta, inputs):
        byte_list = self.get_encoded_cache_list(window_meta, inputs)
        # size_before =  inputs.nbytes
        size_after = 0
        # print("Input size: {}/{}".format(size_after, size_before))
        if len(byte_list) < len(inputs):
            if self.use_mp:
                if self._mp_pool is None:
                    self._mp_pool = ThreadPool(8)
                encoded_list = self._mp_pool.map(
                    _png_encoding, inputs[len(byte_list):])
            else:
                encoded_list = []
                for i in range(len(byte_list), len(inputs)):
                    encoded_list.append(_png_encoding(inputs[i]))
            byte_list.extend(encoded_list)

        return byte_list

    def serialize(self, window_meta, inputs):
        byte_list = self.encode_numpy2img(window_meta, inputs)
        start_idx = 0
        if self.window_meta_cache is not None and \
                window_meta[START_FRAME] >= self.window_meta_cache[START_FRAME] and \
                self.window_meta_cache[END_FRAME] >= window_meta[START_FRAME]:
            # Overlapping window
            start_idx = self.window_meta_cache[END_FRAME] - \
                window_meta[START_FRAME] + 1

        # update cache
        if self.cache_data:
            self.window_meta_cache = window_meta
            self.encoded_cache_list = byte_list
        return byte_list[start_idx:]

    def unserialize(self, output, output_size):
        output = np.frombuffer(output, dtype=np.float32).reshape(output_size)
        return output

    def compute_buf_size(self, input_data):
        size = 0
        for output in input_data:
            size += len(output)
        print("Total buffer size", size)

    def _process_response(self, model_output):
        window_meta = model_output.window_meta
        window_meta = (window_meta.start_frame,
                       window_meta.end_frame, window_meta.id)
        output_size = pickle.loads(model_output.output_size)
        output_data = self.unserialize(model_output.output_data, output_size)

        return (window_meta, output_data, output_size)

    def _handle_predict_response(self, future):
        output = self._process_response(future.result())
        self._callback(output)

    def predict(self, window_meta, input_data):
        """
        Client function to call the rpc for predict
        """
        # Serialize input
        start_time = timeit.default_timer()
        # assert torch.is_tensor(input_data)
        input_data = self.serialize(window_meta, input_data)
        # print("Input data len", len(input_data), window_meta)
        # self.compute_buf_size(input_data)
        input_size = pickle.dumps((len(input_data), ))
        window_meta = predict_pb2.WindowMeta(
            start_frame=window_meta[START_FRAME], end_frame=window_meta[END_FRAME], id=window_meta[ID])
        model_input = predict_pb2.ModelInput(
            window_meta=window_meta, input_size=input_size)
        model_input.input_data.extend(input_data)
        preprocessing_time = (timeit.default_timer() - start_time) * 1000

        # predict remote
        start_time = timeit.default_timer()
        if self._callback:
            output_future = self.stub.predict.future(model_input)
            output_future.add_done_callback(self._handle_predict_response)
            output = None
        else:
            model_output = self.stub.predict(model_input)
            output = self._process_response(model_output)
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
