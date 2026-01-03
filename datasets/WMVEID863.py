import os
from .bases import BaseImageDataset
import re
import pdb
import random
random.seed(10)

class WMVEID863(BaseImageDataset):

    dataset_dir = 'WMVEID863'

    def __init__(self, root, print_info= True):
        self.root = root
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.gallery_dir = os.path.join(self.dataset_dir, 'test')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        
        # query contains the first version sample
        # query1 contains cyclegan generated sample

        self.check_dir_exist()



        self.train , self.num_train_pids , self.num_train_cams = self.get_data(self.train_dir, relabel=True)
        self.num_train_imgs = len(self.train)
        self.num_train_vids = 1
        self.gallery, self.num_gallery_pids, self.num_gallery_cams = self.get_data(self.gallery_dir, relabel=False)
        self.num_gallery_imgs = len(self.gallery)
        self.num_gallery_vids = 1
        self.query, self.num_query_pids ,self.num_query_cams= self.get_data(self.query_dir, relabel=False, ratio = 1)
        self.num_query_imgs = len(self.query)
        self.num_query_vids = 1

        if print_info:
            self.print_statistics_info()

    def get_data(self, folder, relabel=False, ratio = 1):
        vids = os.listdir(folder)
        # pdb.set_trace()
        
        if ratio != 1:
            print('randomly sample ',ratio, 'ids for ttt')
            vids = random.sample(vids, int(len(vids)*ratio))
        labels = [int(vid) for vid in vids]

        if relabel:
            label_map = dict()
            for i, lab in enumerate(labels):
                label_map[lab] = i
        cam_set = set()
        img_info = []
        for vid in vids:
            id_vimgs = os.listdir(os.path.join(folder, vid, "vis"))
            id_nimgs = os.listdir(os.path.join(folder, vid, "ni"))
            # print(vid)
            id_timgs = os.listdir(os.path.join(folder, vid, "th"))
            for i, img in enumerate(id_vimgs):
                img_list = []
                vpath = os.path.join(folder, vid, "vis", id_vimgs[i])
                npath = os.path.join(folder, vid, "ni", id_nimgs[i])
                tpath = os.path.join(folder, vid, "th", id_timgs[i])
                label = label_map[int(vid)] if relabel else int(vid)
                img_list.append(vpath)
                img_list.append(npath)
                img_list.append(tpath)

                night = re.search('n+\d',img).group(0)[1]
                cam = re.search('v+\d',img).group(0)[1]
                cam = int(cam)
                night = int(night)
                cam_set.add(cam)
                img_info.append((img_list, label, cam, 1))

                # img_info_R.append((vpath, label, cam))
                # img_info_N.append((npath, label, cam))
                # img_info_T.append((tpath, label, cam))

        return img_info, len(vids), len(cam_set)

    def check_dir_exist(self):
        if not os.path.exists(self.root):
            raise Exception('Error path: {}'.format(self.root))
        if not os.path.exists(self.train_dir):
            raise Exception('Error path:{}'.format(self.train_dir))
        if not os.path.exists(self.gallery_dir):
            raise Exception('Error path:{}'.format(self.gallery_dir))
        if not os.path.exists(self.query_dir):
            raise Exception('Error path:{}'.format(self.query_dir))

    def print_statistics_info(self):
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras ")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(self.num_train_pids, len(self.train)*3, self.num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(self.num_query_pids, len(self.query)*3, self.num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(self.num_gallery_pids, len(self.gallery)*3, self.num_gallery_cams))
        print("  ----------------------------------------")


if __name__ == "__main__":
    root = 'F:/1_mzq\datasets\data enlargement\Multi-Spectral Vehicle Dataset\Sample'
    dataset = WMVEID863(root, True)
    # train_data = dataset.train_data
    # for i  in train_data:
    #     print(i)