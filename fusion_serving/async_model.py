import multiprocessing as mp
import time
from threading import Timer
import queue
import threading


class Watchdog:
    def __init__(self, timeout):
        self.timeout = timeout
        self.timer = None

    def start(self):
        self.timer = Timer(self.timeout, self.default_handler)
        self.timer.start()

    def reset(self):
        self.stop()
        self.start()

    def stop(self):
        if self.timer:
            self.timer.cancel()

    def default_handler(self):
        raise self


class AsyncModel():
    class WorkerProcess(mp.Process):
        # class WorkerProcess(threading.Thread):
        def __init__(self, base_model, input_queue, output_queue, lock):
            super(AsyncModel.WorkerProcess, self).__init__()
            self.base_model = base_model
            self.input_queue = input_queue
            self.output_queue = output_queue
            # self.lock = lock
            self.lock = threading.Lock()
            self._terminate_event = mp.Event()
            self._reset_event = mp.Event()
            self._working = False
            self.timeout = 0.0001
            self.watch_dog = Watchdog(100)

            self.base_model.set_cache_data(False)
            self.base_model.set_callback(self._handle_window_output)
            self.outstanding_windows = {}
            self._throttle_limit = 2

        def _reset(self):
            # Empty the input queue
            try:
                while True:
                    self.input_queue.get(timeout=self.timeout)
            except queue.Empty:
                pass

            # Wait for the outstanding results
            while True:
                with self.lock:
                    if len(self.outstanding_windows) <= 0:
                        break
                time.sleep(1)
            self.outstanding_windows = {}
            self.base_model.reset()
            self._reset_event.clear()

        def _handle_window_output(self, output):
            with self.lock:
                # print("Handling remote output ", output[0][2], self.outstanding_windows.keys())
                assert output[0][2] in self.outstanding_windows, print(
                    "Received untracked window", self.outstanding_windows, output[0][2])
                start_time = self.outstanding_windows[output[0][2]][1]
                # print("Received window_id with total time:", output[0][2], (time.time() - start_time)*1000)
                del self.outstanding_windows[output[0][2]]
                self.output_queue.put(
                    (output[0], output[1], output[2], start_time))

        def run(self):
            print("Asychronous worker started!")
            while not self._terminate_event.is_set():
                # Check if reset is set
                if self._reset_event.is_set():
                    self._reset()

                try:
                    latest_input = None
                    while True:
                        input = self.input_queue.get(timeout=self.timeout)
                        if latest_input is not None:
                            print("Skipping window sending for:",
                                  latest_input[0])
                        latest_input = input
                except queue.Empty:
                    pass
                except Exception as e:
                    print("Exception reading from the queue..", e)
                # print("Foudn in the the output queue", latest_input)
                input = latest_input
                if input is not None:
                    # self.watch_dog.reset()
                    self._working = True
                    window_id = input[0]
                    input_data = input[1]
                    # start_time = time.time()
                    start_time = input[2]
                    # print("Pickedup input:", window_id.id, (time.time() - start_time)*1000)
                    while True:
                        with self.lock:
                            if len(self.outstanding_windows) < self._throttle_limit:
                                break
                        time.sleep(0.001)

                    with self.lock:
                        self.outstanding_windows[window_id.id] = (
                            window_id, start_time)
                    output = self.base_model(window_id, input_data)

                    # The below code is for synchronous GRPC call
                    # assert output[0][0] == window_id.start_frame and \
                    #    output[0][1] == window_id.end_frame
                    # print("Received window_id with total time:", window_id.id, (time.time() - start_time)*1000)
                    # with self.lock:
                    #     self.output_queue.put(output)
                    #     self._working = False
                    # self.watch_dog.stop()
                else:
                    pass  # should we introduce any sleep
            print("Asynchronous worker stopped!")
            self._terminate_event.clear()
            return

    def __init__(self, base_model, num_workers=2):
        self.base_model = base_model
        # self.input_queue = queue.Queue()
        # self.output_queue = queue.Queue()
        # self.lock = threading.Lock()
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.lock = mp.Lock()
        self.timeout = 0.001
        self.num_workers = num_workers
        self._workers = []
        self._start_workers()

    def _start_workers(self):
        for i in range(self.num_workers):
            worker = AsyncModel.WorkerProcess(self.base_model, self.input_queue,
                                              self.output_queue, self.lock)
            worker.start()
            self._workers.append(worker)

    def _wait(self):
        for worker in self._workers:
            worker._reset_event.set()
            while worker._reset_event.is_set():
                time.sleep(1)
            self.next_output()

    def reset(self):
        self._wait()

    def has_work(self):
        raise NotImplemented

    def predict(self, window_id, input):
        self.input_queue.put((window_id, input, time.time()))
        return

    def next_output(self):
        latest_window_id = -1
        latest_output = None
        try:
            while True:
                output = self.output_queue.get(timeout=self.timeout)
                start_time = output[3]
                (_, _, window_id) = output[0]  # frames
                # print("Retrieved output at: ", window_id, (time.time() - start_time)*1000)
                if window_id > latest_window_id:
                    if latest_window_id != -1:
                        print("Skipped output: ", latest_window_id)
                    latest_window_id = window_id
                    latest_output = output
        except queue.Empty:
            pass
        return latest_output

    def close(self):
        self._wait()
        for worker in self._workers:
            worker._terminate_event.set()
            while worker._terminate_event.is_set():
                time.sleep(1)
        # Close mp.Queues
        self.input_queue.cancel_join_thread()
        self.input_queue.close()
        self.output_queue.cancel_join_thread()
        self.output_queue.close()
        for worker in self._workers:
            worker.terminate()
            # worker.join()
