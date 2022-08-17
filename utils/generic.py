# -*- encoding: utf-8 -*-
'''
@create_time: 2022/05/27 14:13:46
@author: lichunyu
'''
from enum import Enum
from dataclasses import dataclass, field


@dataclass
class AdvanceArguments:

    cuda_visible_devices: str = field(default='0')

    model: str = field(default="bert_classification")

    # data_path: str = field(default=None)
    # train_data_path: str = field(default=None)
    # dev_data_path: str = field(default=None)

    strategy: str = field(default=None)

    log_file_path: str = field(default=None)




class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class cached_property(property):
    """
    Descriptor that mimics @property but caches output in member variable.

    From tensorflow_datasets

    Built-in in functools from Python 3.8.
    """

    def __get__(self, obj, objtype=None):
        # See docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            cached = self.fget(obj)
            setattr(obj, attr, cached)
        return cached


def get_args(arg_name, t_args, extra_args:list, default_value=None):
    args_list = [t_args] + extra_args
    for args in args_list:
        if arg_name in args.__dict__:
            return args.__dict__[arg_name]
    return default_value


def setup_logger(log_file_path=None, rank=-1):
    import logging
    import sys
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file_path is not None:
        handlers.append(logging.FileHandler(log_file_path))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=handlers,
    )


class MirrorDict(object):

    def __getitem__(self, key):
        return key