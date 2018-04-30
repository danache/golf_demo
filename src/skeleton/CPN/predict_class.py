
import numpy as np

import cv2

import sys

import tensorflow as tf
import scipy.misc as scm

sys.path.insert(0,"lib/")
sys.path.insert(0,"../*")
from src.skeleton.CPN.CPN_config import cfg

from src.skeleton.CPN.lib.tfflat.base import Tester

from src.skeleton.CPN.network import Network
import os
def EnsureDir(dirs):
    if os.path.isdir(dirs):
        return
    else:
        os.mkdir(dirs)

class test_class():
    def __init__(self, resume, partnum=14, dategen=None, save_dir="", dataset_name=""
                 ):

        self.resume = resume

        self.dataset = dataset_name
        self.save_dir = save_dir

        self.cpu = '/cpu:0'

        self.partnum = partnum

        self.datagen = dategen
        self.colors = [
            [0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 245, 255], [255, 131, 250], [255, 255, 0],
            [0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 245, 255], [255, 131, 250], [255, 255, 0],
            [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

        self.line = [[5, 7],[5,6], [6, 8], [7, 9], [8, 10], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]]

        self.symmetry = [(0, 3), (1, 4), (2, 5), (6, 9), (7, 10), (8, 11)]

        self.tester = Tester(Network(), cfg)
        self.tester.load_weights(resume)

    def pred(self):
        generator = self.datagen.get_batch_generator()
        num_test = self.datagen.getN()
        hm = {}
        coordssss = {}
        pred_data = dict()

        imgs = []
        out_hm = []
        cls_skeleton = np.zeros((num_test, cfg.nr_skeleton, 3))
        crops = np.zeros((num_test, 4))

        idx = 0
        while True:
            try:
                gt_img, img, details, name = next(generator)

                feed = [img[np.newaxis, ...]]
                ori_img = img.transpose(1, 2, 0)
                flip_img = cv2.flip(ori_img, 1)
                feed.append(flip_img.transpose(2, 0, 1)[np.newaxis, ...])
                feed = np.vstack(feed)
                # feed = np.array(feed)[0]
                res = self.tester.predict_one([feed.transpose(0, 2, 3, 1).astype(np.float32)])[0]
                res = res.transpose(0, 3, 1, 2)

                out_hm.append(res)
                fmp = res[1].transpose((1, 2, 0))
                fmp = cv2.flip(fmp, 1)
                fmp = list(fmp.transpose((2, 0, 1)))
                for (q, w) in cfg.symmetry:
                    fmp[q], fmp[w] = fmp[w], fmp[q]
                fmp = np.array(fmp)
                res[0] += fmp
                res[0] /= 2

                r0 = res[0].copy()
                r0 /= 255.
                r0 += 0.5
                for w in range(cfg.nr_skeleton):
                    res[0, w] /= np.amax(res[0, w])
                border = 10
                dr = np.zeros((cfg.nr_skeleton, cfg.output_shape[0] + 2 * border, cfg.output_shape[1] + 2 * border))
                dr[:, border:-border, border:-border] = res[0][:cfg.nr_skeleton].copy()
                for w in range(cfg.nr_skeleton):
                    dr[w] = cv2.GaussianBlur(dr[w], (21, 21), 0)
                for w in range(cfg.nr_skeleton):
                    lb = dr[w].argmax()
                    y, x = np.unravel_index(lb, dr[w].shape)
                    dr[w, y, x] = 0
                    lb = dr[w].argmax()
                    py, px = np.unravel_index(lb, dr[w].shape)
                    y -= border
                    x -= border
                    py -= border + y
                    px -= border + x
                    ln = (px ** 2 + py ** 2) ** 0.5
                    delta = 0.25
                    if ln > 1e-3:
                        x += delta * px / ln
                        y += delta * py / ln
                    x = max(0, min(x, cfg.output_shape[1] - 1))
                    y = max(0, min(y, cfg.output_shape[0] - 1))
                    cls_skeleton[idx, w, :2] = (x * 4 + 2, y * 4 + 2)
                    cls_skeleton[idx, w, 2] = r0[w, int(round(y) + 1e-10), int(round(x) + 1e-10)]
                # map back to original images
                # print(cls_skeleton)

                crops[idx, :] = details

                for w in range(cfg.nr_skeleton):
                    cls_skeleton[idx, w, 0] = cls_skeleton[idx, w, 0] / cfg.data_shape[1] * (
                        crops[idx][2] - crops[idx][0]) + crops[idx][0]
                    cls_skeleton[idx, w, 1] = cls_skeleton[idx, w, 1] / cfg.data_shape[0] * (
                        crops[idx][3] - crops[idx][1]) + crops[idx][1]


                img_cp = gt_img.copy()
                joint = cls_skeleton[idx].astype(np.int)
                print(joint)
                for i in range(3, joint.shape[0]):
                    cv2.circle(img_cp, (joint[i][0], joint[i][1]), 5, (255, 255, 0), -1)
                for j in range(len(self.line)):
                    #print((joint[self.line[j][0]][0], joint[self.line[j][0]][1]))
                    cv2.line(img_cp, (joint[self.line[j][0]][0], joint[self.line[j][0]][1]),
                             (joint[self.line[j][1]][0], joint[self.line[j][1]][1]), (0, 255, 0), 3)

                img_path = os.path.join(self.save_dir, name)
                folder = "/" + os.path.join(*(img_path.split("/")[:-1]))
                EnsureDir(folder)
                print(idx)
                print(os.path.join(self.save_dir, name),)
                scm.imsave(os.path.join(self.save_dir, name), img_cp)
                idx += 1

                #imgs.append(gt_img)
            # return cls_skeleton, imgs, out_hm
            except Exception as e:
                print(e)
                return cls_skeleton, imgs, out_hm
                break