import os
import sys
import unittest
import argparse

from template_lib.examples import test_bash
from template_lib import utils


class TestingAutoencoder(unittest.TestCase):

  def test_train_introvae(self):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=./submodule
        python 	exp/introvae.py \
          --config exp/configs/autoencoder.yaml \
          --command train_introvae \
          --outdir results/IntroVAE/train_introvae

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'

    argv_str = f"""
                --config exp/configs/autoencoder.yaml
                --command {command}
                --outdir {outdir}
                --overwrite_opts False
                """
    from exp.introvae import run
    run(argv_str)

  def test_train_autoencoder(self):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=1
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=./exp:./
        python 	exp/scripts/train_net.py \
          --config exp/configs/autoencoder.yaml \
          --command train_autoencoder \
          --outdir results/IntroVAE/train_autoencoder

    :return:
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

    argv_str = f"""
                --config exp/configs/autoencoder.yaml
                --command {command}
                --outdir {outdir}
                """
    from exp.scripts.train_net import run
    run(argv_str)

  def test_train_autoencoder_conv(self):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=2
        export PORT=6008
        export TIME_STR=1
        export PYTHONPATH=./exp:./
        python 	exp/scripts/train_net.py \
          --config exp/configs/autoencoder.yaml \
          --command train_autoencoder_conv \
          --outdir results/Autoencoder/train_autoencoder_conv

    :return:
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

    argv_str = f"""
                --config exp/configs/autoencoder.yaml
                --command {command}
                --outdir {outdir}
                """
    from exp.scripts.train_net import run
    run(argv_str)

  def test_train_AEGAN_acai(self):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=./exp:./
        python 	exp/scripts/train_net.py \
          --config exp/configs/autoencoder.yaml \
          --command train_AEGAN_acai \
          --outdir results/Autoencoder/train_train_AEGAN_acai

    :return:
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

    argv_str = f"""
                --config exp/configs/autoencoder.yaml
                --command {command}
                --outdir {outdir}
                """
    from exp.scripts.train_net import run
    run(argv_str)



