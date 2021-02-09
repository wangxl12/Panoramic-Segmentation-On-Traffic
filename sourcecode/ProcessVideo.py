import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, sys
import numpy as np
import cv2
import random
from PIL import Image
import shutil
from tqdm import tqdm
# import some common detectron2 utilities
ROOTPATH = '/home/ma-user/work/'
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
os.chdir(ROOTPATH + 'sourcecode/tools')
# list_dir = os.listdir(ROOTPATH + 'sourcecode/tools')
# print(list_dir)
from cutImg import CutImage, _PanopticPrediction