import numpy as np
import os
import scipy.misc as scm
import json
import scipy
import cv2
import random
import tqdm
class DataGenerator():
    def __init__(self, json=None, img_path="", resize=256, normalize=True,
                 pixel_mean=np.array([[[102.9801, 115.9465, 122.7717]]])):
        self.img_dir = img_path
        self.json_path = json
        self.resize = [resize, resize]
        self.res = [64, 64]
        self.pixel_means = pixel_mean
        self.normalize = normalize
        self.load_data()

    def load_data(self):
        with open(self.json_path, "r") as f:
            json_file = json.load(f)
        human_lst = []
        for files in json_file:
            human_anno = files["human_annotations"]

            for human in human_anno.keys():
                human_lst.append(dict(box=human_anno[human],
                                      name=files['image_id']))

        self.test_data = human_lst

    def open_img(self, name, color='RGB'):
        """ Open an image
        Args:
            name	: Name of the sample
            color	: Color Mode (RGB/BGR/GRAY)
        """
        img = scipy.misc.imread(os.path.join(self.img_dir,name))
        return img

    def get_transform(self, center, scale, res, rot=0):
        # Generate transformation matrix
        h = scale
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / (h[1])
        t[1, 1] = float(res[0]) / (h[0])
        t[0, 2] = res[1] * (-float(center[0]) / h[1] + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h[0] + .5)
        t[2, 2] = 1
        return t

    def getN(self):
        return len(self.dataset)

    def transform(self, pt, center, scale, res, invert=0, rot=0):
        # Transform pixel location to different reference
        t = self.get_transform(center, scale, res, rot=rot)
        if invert:
            t = np.linalg.inv(t)
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2].astype(int)

    def transformPreds(self, coords, center, scale, res, reverse=0):
        #     local origDims = coords:size()
        #     coords = coords:view(-1,2)
        lst = []

        for i in range(coords.shape[0]):
            lst.append(self.transform(coords[i], center, scale, res, reverse, 0, ))

        newCoords = np.stack(lst, axis=0)

        return newCoords

    def crop(self, img, center, scale, res):

        ul = np.array(self.transform([0, 0], center, scale, res, invert=1))

        br = np.array(self.transform(res, center, scale, res, invert=1))

        old_x = max(0, ul[0]), min(len(img[0]), br[0])
        old_y = max(0, ul[1]), min(len(img), br[1])
        new_img = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

        return scipy.misc.imresize(new_img, res)

    def getFeature(self, box):
        x1, y1, x2, y2 = box
        center = np.array(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
        scale = y2 - y1, x2 - x1
        return center, scale

    def get_candidate(self,hm, thresh_ratio=0.8, num_joint=14, mask_kernel=(13, 13)):
        assert len(hm.shape) == 4

        per_cand_lst = []
        for num_hm in range(hm.shape[3]):

            hm_tmp = hm[0, :, :, num_hm].copy()

            joint_candi_index_lst = []
            flag = True
            max_val = np.max(hm_tmp)

            thresh = max_val * thresh_ratio
            while (flag):

                index = np.unravel_index(hm_tmp.argmax(), [64, 64])
                if hm_tmp[index] < thresh:
                    flag = False
                    break

                joint_candi_index_lst.append(index[::-1])

                mask = np.zeros([64, 64])
                mask[index] = 1
                mask = cv2.GaussianBlur(mask, mask_kernel, 0)

                # print(np.unique(mask))

                mask[mask > 0.00341797] = 1

                for i in range(hm_tmp.shape[0]):
                    for j in range(hm_tmp.shape[1]):
                        if mask[i, j] == 1:
                            hm_tmp[i, j] = 0
            # print(" {} : {}".format(num_hm,joint_candi_index_lst))
            per_cand_lst.append(joint_candi_index_lst)

            # print(joint_candi_index_lst)

        return per_cand_lst

    def get_data_by_index(self,hm, index):
        if index < 0 or index >= len(hm):
            return False
        return hm[index]

    def calculate_distance(self,a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def revise_by_pre_next(self,pre_cand_coord, this_cand_coord, next_cand_coord, joint_idx,
                           pre_dpflow=None, this_dpflow=None):

        pre_joint_coord = None
        next_joint_coord = None

        if pre_cand_coord:
            if len(pre_cand_coord[joint_idx]) == 1:
                pre_joint_coord = pre_cand_coord[joint_idx]

        if pre_joint_coord:
            best_match = None
            distance = 99999
            for joints in this_cand_coord[joint_idx]:
                dis = self.calculate_distance(joints, pre_joint_coord[0])
                if dis < distance:
                    distance = dis
                    best_match = joints
            return best_match

        if next_cand_coord:
            if len(next_cand_coord[joint_idx]) == 1:
                next_joint_coord = next_cand_coord[joint_idx]

        if next_joint_coord:
            best_match = None
            distance = 99999
            for joints in this_cand_coord[joint_idx]:
                dis = self.calculate_distance(joints, next_joint_coord[0])
                if dis < distance:
                    distance = dis
                    best_match = joints
            return best_match
        return random.choice(this_cand_coord)

    #     if pre_dpflow:



    def fix_heatmap_using_context_dpflow(self,hm, dpflow=None, context_num=1):
        return_lst = []
        for i in range(len(hm)):
            this_coord = []
            pre_cand_coord = self.get_data_by_index(hm, i - 1)
            this_cand_coord = self.get_data_by_index(hm, i)
            next_cand_coord = self.get_data_by_index(hm, i + 1)

            if dpflow:
                pre_dpflow = self.get_data_by_index(dpflow, i - 1)
                this_dpflow = self.get_data_by_index(dpflow, i)
                next_dpflow = self.get_data_by_index(dpflow, i + 1)

            for j in range(14):
                if len(this_cand_coord[j]) == 1:
                    this_coord.append(this_cand_coord[j])
                elif dpflow:
                    res = self.revise_by_pre_next(pre_cand_coord, this_cand_coord,
                                             next_cand_coord, j, pre_dpflow, this_dpflow)
                    this_coord.append(res)
                else:
                    res = self.revise_by_pre_next(pre_cand_coord, this_cand_coord,
                                             next_cand_coord, j)
                    this_coord.append(res)
            return_lst.append(np.vstack(this_coord))
        return np.array(return_lst)

    def recoverFromHm(self, hm):
        '''
        从heatmap中获取原始输出
        :param hm:
        :param center:
        :param scale:
        :return:
        '''
        res = []
        for nbatch in range(hm.shape[0]):
            tmp_lst = []
            assert hm.shape[0] == 1

            for i in range(hm.shape[-1]):
                index = np.unravel_index(hm[nbatch, :, :, i].argmax(), self.res)
                tmp_lst.append(index[::-1])
            res.append(np.stack(tmp_lst) )
        return np.stack(res)


    def get_batch_generator(self):
        idx = 0
        while idx < len(self.test_data):
            data_slice = self.test_data[idx]
            img = self.open_img(data_slice['name'])

            box = data_slice['box']
            center, scale = self.getFeature(box)

            crop_img = self.crop(img, center, scale, self.resize)
            crop_img = (crop_img.astype(np.float64) - self.pixel_means)

            if self.normalize:
                train_img = crop_img.astype(np.float32) / 255
            else:
                train_img = crop_img.astype(np.float32)
            train_img = np.expand_dims(train_img,0)
            yield train_img, img, center, scale,data_slice['name']
            idx += 1
            if idx % 50 == 0:
                print("%.2f "%(idx / len(self.test_data) * 100) +"% have done!!")