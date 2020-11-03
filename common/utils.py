
from ResNets_3D_PyTorch.utils import AverageMeter
from ResNets_3D_PyTorch.utils import Logger
import os


class Stats:
    def __init__(self, stats_list, stats_freq=10, stats_path="", stats_name="temp", warm_up=10):
        self.stats_list = stats_list
        self.count = 0
        self.stats_freq = stats_freq
        self.warm_up = warm_up
        stats_list = ['Type'] + stats_list
        self.logger = Logger(os.path.join(stats_path, stats_name + ".stats"),
                             stats_list)

        self.stats_objects = {}
        for key in stats_list:
            stats_object = AverageMeter()
            self.stats_objects[key] = stats_object

    def update(self, values):
        # Introduce warmup period
        if self.warm_up > 0:
            self.warm_up -= 1
            return
        non_avg_log = {'Type': 'Non-AVG'}
        for key in values:
            self.stats_objects[key].update(values[key])
            non_avg_log[key] = values[key]

        if self.stats_freq > 0 and (self.count % self.stats_freq) == 0:
            avg_log = {'Type': 'AVG'}
            for key in self.stats_objects:
                if key == 'Type':
                    continue
                avg_log[key] = self.stats_objects[key].avg

            self.logger.log(avg_log)
            self.logger.log(non_avg_log)
            self.count = 0
        self.count += 1
