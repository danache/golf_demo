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
        self.imgExtXBorder = 0#0.1
        self.imgExtYBorder = 0#0.15
        self.data_shape = (384, 288)

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
        return len(self.test_data)

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
            height, width = self.data_shape
            data_slice = self.test_data[idx]
            img = self.open_img(data_slice['name'])
            ori_img = img.copy()
            # box = data_slice['box'].reshape(4, ).astype(np.float32)
            # print(box)
            add = max(img.shape[0], img.shape[1])
            bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT,
                                      value=self.pixel_means.reshape(-1))

            bbox = np.array(data_slice['box']).reshape(4, ).astype(np.float32)
            bbox[:2] += add

            crop_width = bbox[2] * (1 + self.imgExtXBorder * 2)
            crop_height = bbox[3] * (1 + self.imgExtYBorder * 2)
            objcenter = np.array([bbox[0] + bbox[2] / 2., bbox[1] + bbox[3] / 2.])

            if crop_height / height > crop_width / width:
                crop_size = crop_height
                min_shape = height
            else:
                crop_size = crop_width
                min_shape = width
            crop_size = min(crop_size, objcenter[0] / width * min_shape * 2. - 1.)
            crop_size = min(crop_size, (bimg.shape[1] - objcenter[0]) / width * min_shape * 2. - 1)
            crop_size = min(crop_size, objcenter[1] / height * min_shape * 2. - 1.)
            crop_size = min(crop_size, (bimg.shape[0] - objcenter[1]) / height * min_shape * 2. - 1)

            min_x = int(objcenter[0] - crop_size / 2. / min_shape * width)
            max_x = int(objcenter[0] + crop_size / 2. / min_shape * width)
            min_y = int(objcenter[1] - crop_size / 2. / min_shape * height)
            max_y = int(objcenter[1] + crop_size / 2. / min_shape * height)

            # x_ratio = float(width) / (max_x - min_x)
            # y_ratio = float(height) / (max_y - min_y)

            img = cv2.resize(bimg[min_y:max_y, min_x:max_x, :], (width, height))


            details = np.asarray([min_x - add, min_y - add, max_x - add, max_y - add])
            img = img - self.pixel_means

            img = img / 255. * 2.

            img = img.transpose(2, 0, 1)
            #print("detail shape : {}".format(details))
            yield ori_img, img, details, data_slice['name']

            idx += 1
            if idx % 50 == 0:
                print("%.2f "%(idx / len(self.test_data) * 100) +"% have done!!")