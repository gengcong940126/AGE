import random
import os, sys
import unittest

import template_lib.utils as utils


class TestingBuildImagenet(unittest.TestCase):

  def test_subimagenet(self):
    """

    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    os.makedirs(outdir, exist_ok=True)

    import matplotlib.pylab as plt
    from mydata.build_subimagenet import data_path, registed_name_list, \
      registed_func_list, kwargs_list
    from detectron2.data import MetadataCatalog

    idx = 0
    dataset_dicts = registed_func_list[idx](name=registed_name_list[idx], data_path=data_path, **kwargs_list[idx])
    metadata = MetadataCatalog.get(registed_name_list[idx])

    pass