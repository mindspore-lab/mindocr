# For each image, use one collector
class Collector:
    def __init__(self):
        self.collect_keys = []
        self.collect_value = dict()
    def init_keys(self, key_list):
        self.collect_keys.clear()
        self.collect_value.clear()
        self.collect_keys = key_list
        for k in key_list:
            self.collect_value[k] = None
    def update(self, key, value):
        self.collect_value[key] = value
        if key in self.collect_keys:
            self.collect_keys.remove(key)
    def complete(self):
        if len(self.collect_keys) == 0:
            return True
        else:
            return False
    def get_data(self):
        return list(self.collect_value.values())