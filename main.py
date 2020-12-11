
import os
import sys
import numpy as np
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = BASE_DIR

sys.path.append(os.path.join(ROOT_DIR, 'data'))
from dataset import Dataset_, Dataset2
from data_utils import Util

sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'models/modules'))
sys.path.append(os.path.join(ROOT_DIR, 'models/modules/functional'))

from train_helper import  train

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="", choices=['train', 'eval', 'test'], help="mode must be in ['train', 'eval', 'test']")
parser.add_argument('--save_folder', type=str, default="", help='Folder to save checkpoints in.')
parser.add_argument('--restore_from', type=str, default='', help='Checkpoint to restore from.')

def logger_0(string):
  LOG_FILE.write(string+"\n")

def logger_1(string):
  LOG_FILE.write(string+"\n")
  print(string)

FLAGS = parser.parse_args()
assert FLAGS.mode and FLAGS.save_folder, "Must specify mode and save foler"

if not os.path.exists(FLAGS.save_folder):
  os.makedirs(FLAGS.save_folder)
LOG_FILE = open(FLAGS.save_folder+"/log", "a")

if __name__ =='__main__':

  if FLAGS.mode == "train":
    logger_1("Start Training ...")
    train(FLAGS.save_folder, FLAGS.restore_from, [logger_0, logger_1])

  elif FLAGS.mode == 'eval':
    raise NotImplementedError

  else: # mode=test
    raise NotImplementedError 

  LOG_FILE.close()





