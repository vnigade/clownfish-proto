# class TransitionFilter:

class PeriodicFilter:
    def __init__(self, interval):
        self.interval = interval
        self.counter = interval

    def reset(self):
        self.counter = self.interval

    def should_send(self):
        self.counter -= 1
        if self.counter == 0:
            self.reset()
            return True
        return False


class TransitionFilter:
    def __init__(self, threshold):
        self.threshold = threshold
        self.prev_rho = None

    def reset(self):
        self.prev_rho = None

    def update(self, window_id, rho):
        self.prev_rho = (window_id, rho)

    def should_send(self, window_id, rho):
        if self.prev_rho is None:
            self.prev_rho = (window_id, rho)
            return True

        assert self.prev_rho[0] == (window_id - 1)

        should_send = False
        if (rho - self.prev_rho[1]) >= self.threshold:
            should_send = True

        self.prev_rho = (window_id, rho)
        return should_send


class Filter:
    def __init__(self, opt):
        self.period_filter = PeriodicFilter(opt.filter_interval)
        self.transition_filter = TransitionFilter(opt.transition_threshold)
        self.last_window_sent = -1

    def reset(self):
        self.last_window_sent = -1
        self.period_filter.reset()
        self.transition_filter.reset()

    def should_send(self, window_id, scheme, rho):
        if self.last_window_sent == window_id:
            self.transition_filter.update(window_id, rho)
            return False

        should_send = False
        if scheme == 'periodic':
            should_send = self.period_filter.should_send()
        elif scheme == 'transition':
            should_send = self.transition_filter.should_send(window_id, rho)

        if should_send:
            self.period_filter.reset()
            self.last_window_sent = window_id

        return should_send
