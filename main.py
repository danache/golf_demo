import sys
sys.path.append(sys.path[0])
del sys.path[0]
from src.skeleton.HG.test_datagen import DataGenerator as hgData

import os
from network.ian_hourglass import hourglassnet
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils.aux_fuction import *


def parse_args():
    """
        get parameters from command line.
        
        Parameters:
        ----------
        video_path : str, default = ""
            The origin video path (file or folders)
            
        output_path : str, default = ""
            Output dir.  including: frames , optical flows , output videos, limb length.
        
        detection : bool, default = False
            Can use mask-rcnn detections result.If False, use pretrained SSD network results.
        
        model : int , default = 0.
            0 means Hourglass model.
            1 means CPN model
            2 means
        
        height: int, default = 175
            Human height which will be used in predict limb length.
    """

    parser = argparse.ArgumentParser(description='hourglass test code')
    parser.add_argument('--v', dest='video_path', type=str,
                        default="",
                        )
    parser.add_argument('--o', dest='output_path', type=str,
                        default="/media/bnrc2/_backup/dataset/output/" ,
                        )
    parser.add_argument('--d', dest='detection', type=bool,
                        default=False,
                        )
    parser.add_argument('--m', dest='model_type', type=int,
                        default=0,
                        )
    parser.add_argument('--h', dest='height', type=int,
                        default=175,
                        )
    args = parser.parse_args()
    return args


def start():
    """
        get predictions
        :return: joint coordinates, videos, limb length
    """
    print('--Parsing Params')

    args = parse_args()
    video_path, output_path, use_mask_RCNN, human_height, model_type = generate_sys_Log(args)

    ########1. Video -> Framse #########
    return_lst = videos2frames(video_path, output_path)

    ####### 2. Generate detect  #######



    ######### load data #########
    if model_type == 0:
        test_data = hgData()


    if model_type == 0:
        network_params = process_network("./config/config.cfg")
        model = hourglassnet(stacks=network_params['nstack'])



    # test.test("/home/bnrc2/data/")


if __name__ == '__main__':
    start()
