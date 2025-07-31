"""
created by Allan Yu, 23-10-14
基本的Utils支持，其他更细分的Utils有专门的文件
"""

from datetime import datetime
import torch
from torch.nn import init


def extract(dict, key, default=None):
    if key in dict:
        value = dict[key]
        del dict[key]
        return value
    return default


# Define the initial function to init the layer's parameters for the network
def weight_init(m):
    if isinstance(m, torch.nn.Conv2d):
        # init.xavier_uniform_(m.weight)
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Linear):
        init.kaiming_normal_(m.weight.data)
        # init.normal_(m.weight, std=1e-3)
        if m.bias is not None:
            init.constant_(m.bias, 0)


def save_model(model, path, ignore_keys=None):
    model_dict = model.state_dict()
    if ignore_keys is not None:
        to_delete_key = []
        for igkey in ignore_keys:
            for key, value in model_dict.items():
                if igkey in key and key not in to_delete_key:
                    to_delete_key.append(key)

        for key in to_delete_key:
            #####print(key)
            del model_dict[key]

    torch.save(model_dict, path)


def load_model(model, path, ignore_keys=None, replace_keys=None, print_out=True):
    pretrained_dict = torch.load(path)
    if ignore_keys is not None:
        to_delete_key = []
        for igkey in ignore_keys:
            for key, value in pretrained_dict.items():
                if igkey in key and key not in to_delete_key:
                    to_delete_key.append(key)

        for key in to_delete_key:
            if print_out:
                print("ignored key:", key)
            del pretrained_dict[key]

    if replace_keys is not None:
        replaced_dict = {}
        to_delete_key = []
        for repkey, repvalue in replace_keys.items():
            for key, value in pretrained_dict.items():
                if repkey in key and key not in to_delete_key:
                    to_delete_key.append(key)
                    replaced_dict[key.replace(repkey, repvalue)] = value

        for key in to_delete_key:
            if print_out:
                print("replace key:", key)
            del pretrained_dict[key]

        pretrained_dict.update(replaced_dict)

    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)


TIME = datetime.now().strftime("%A_%d_%B_%Y_%Hh_%Mm_%Ss")
TIME_NOW = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
DATE_NOW = datetime.now().strftime("%y-%m-%d")


def time2hms(seconds, simplify=True):
    if simplify:
        seconds = int(seconds)

    # 计算总小时数
    hours = int(seconds // 3600)
    # 计算剩余的秒数
    seconds %= 3600
    # 计算总分钟数
    minutes = int(seconds // 60)
    # 计算剩余的秒数
    seconds %= 60

    return hours, minutes, seconds


def readable_hms(hours, minutes, seconds, simplify=True):
    text = ""
    text += f"{hours}h" if hours != 0 else ""
    text += f"{minutes}m" if minutes != 0 else ""
    text += f"{seconds}s" if simplify else "{:.2f}s".format(seconds)
    return text


def readable_time(seconds):
    hours, minutes, seconds = time2hms(seconds)
    return readable_hms(hours, minutes, seconds)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.weight = 0
        self.avg = 0

    def update(self, val, weight=1):
        self.val = val
        self.sum += val * weight
        self.weight += weight
        self.avg = self.sum / self.weight

    def __str__(self):
        return "{:0.4f}".format(self.avg)


class DynamicMeters(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.meters = []

    def update(self, vals, major, weight=1):
        if len(self.meters) == 0:
            for val in vals:
                self.meters.append(AverageMeter())
        assert len(self.meters) == len(vals)
        for idx, val in enumerate(vals):
            self.meters[idx].update(val, weight)

        self.major_id = len(self.meters) + major if major < 0 else major

    def major(self):
        return self.meters[self.major_id]

    def num_of_meters(self):
        return len(self.meters)

    def __str__(self):
        result = ""
        first = True
        for meter in self.meters:
            result += "" if first else "/"
            first = False
            result += "{:0.4f}".format(meter.avg)

        return result


def message(obj, method, **param):
    if not hasattr(obj, method):
        return False

    method = getattr(obj, method)
    if not callable(method):
        return False

    method(**param)
    return True


def dict2list(dict, selection=None):
    if selection is None:
        return tuple([value for value in dict.values()])

    if isinstance(selection, (list, tuple)):
        return tuple([dict[key] for key in selection])
    return dict[selection]
