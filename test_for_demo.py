import sys

sys.path.append(sys.path[0])
del sys.path[0]

import os

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
                        default="/media/bnrc2/_backup/dataset/output/",
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
    # return_lst = videos2frames(video_path, output_path)
    #
    # ####### 2. Generate detect  #######

    cmd = '/home/bnrc2/anaconda3/envs/tf14/bin/python  /home/bnrc2/mu/mxnet-ssd/demo.py '
    cmd += '--images_path {} --outdir {}'.format("/media/bnrc2/_backup/dataset/normalsizeImages/png/",output_path)
    print(cmd)
    print(os.system( cmd))

    ######### load data #########
    json_path = os.path.join(output_path, "golf.json")
    img_path = "/media/bnrc2/_backup/dataset/normalsizeImages/png"

    if model_type == 0:

        from src.skeleton.HG.predict_class import test_class
        from network.ian_hourglass import hourglassnet

        from src.skeleton.HG.test_datagen import DataGenerator as hgData
        network_params = process_network("./config/config.cfg")
        model = hourglassnet(stacks=network_params['nstack'])

        valid_data = hgData(json=json_path, img_path=img_path, resize=256, normalize=True)
        model_path  = "/media/bnrc2/_backup/models/hg_multiGauss/hg2_32"

        test = test_class(model=model, nstack=network_params['nstack'],

                          resume=model_path, dataset_name="byd",
                          gpu=[0], partnum=network_params['partnum'], dategen=valid_data,
                          save_dir=output_path
                          )

        test.generateModel()
        test.test_init()
        test.pred()
    elif model_type == 1:
        from src.skeleton.CPN.Cpn_datagen import DataGenerator
        from src.skeleton.CPN.predict_class import test_class
        import sys

        sys.path.insert(0, "./src/skeleton/CPN/")
        #valid_data = DataGenerator(json=json_path, img_path=img_path)
        print(json_path)
        print(img_path)
        valid_data = DataGenerator(json=json_path,
                                   img_path=img_path)
        test = test_class(resume="/media/bnrc2/_backup/models/cpn/res101/snapshot_350.ckpt",
                          dategen=valid_data,save_dir=output_path)

        pred, imgs, hm = test.pred()
        #print(pred)
        #
    # valid_data = DataGenerator(json=json_path, img_path=img_path, resize=256, normalize=True)
    #
    # test = test_class(model=model, nstack=network_params['nstack'],
    #
    #                   resume=model_path, dataset_name="byd",
    #                   gpu=[0], partnum=network_params['partnum'], dategen=valid_data, save_dir=save_path
    #                   )
    #
    # test.generateModel()
    # test.test_init()
    # test.pred()
    # # test.test("/home/bnrc2/data/")
    #

if __name__ == '__main__':
    start()

