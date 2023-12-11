import bisect


class Logger(object):

    def __init__(self):
        super(Logger, self).__init__()
        self.log_dict = dict()
        self.indices = list()

    def log(self, name, index, value):
        if name not in self.log_dict:
            self.log_dict[name] = dict()
        i = bisect.bisect_left(self.indices, index)
        if i >= len(self.indices) or self.indices[i] != index:
            self.indices.insert(i, index)
        self.log_dict[name][index] = value