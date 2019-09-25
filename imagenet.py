from __future__ import print_function
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.imagenet
from .voc_eval import voc_eval


import os, sys
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import subprocess
import pdb
import pickle
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class imagenet(imdb):
    def __init__(self, image_set, devkit_path, data_path):
        imdb.__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() # data/imagenet/ILSVRC/
        self._data_path = os.path.join(self._devkit_path, 'ILSVRC')

        with open(os.path.join(self._data_path, 'ImageSets', 'CLS-LOC', 'meta-train-class.txt'), 'r') as f:
            self.mtrain_classes = [x.strip() for x in f.readlines()]     
        with open(os.path.join(self._data_path, 'ImageSets', 'CLS-LOC', 'meta-test-class.txt'), 'r') as f:
            self.mtest_classes = [x.strip() for x in f.readlines()]     
        with open(os.path.join(self._data_path, 'ImageSets', 'CLS-LOC', 'meta-train-wnid.txt'), 'r') as f:
            self.mtrain_wnid = [x.strip() for x in f.readlines()]     
        with open(os.path.join(self._data_path, 'ImageSets', 'CLS-LOC', 'meta-test-wnid.txt'), 'r') as f:
            self.mtest_wnid = [x.strip() for x in f.readlines()]     
                
        if 'meta-train' in image_set:
            self._classes = tuple(self.mtrain_classes)
            self._wnid = tuple(self.mtrain_wnid)
        elif 'meta-test' in image_set:
            self._classes = tuple(self.mtest_classes)
            self._wnid = tuple(self.mtest_wnid)
        else:
            print('Not Support')
            exit()

        self._classes = ('__background__',) + self._classes
        self._wnid = (0,) + self._wnid

        self._classes_to_wnid = dict(zip(self._classes, self._wnid))
        self._wnid_to_ind = dict(zip(self._wnid, xrange(self.num_classes)))
        self._class_to_ind = dict(zip(self._classes, xrange(self.num_classes)))

        self._image_ext = ['.JPEG']
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), 'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

        # redo the step
        self.cls2img = {}
        for i in xrange(1, self.num_classes):
            self.cls2img[i] = []
    

       

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'Data', self._image_set, index + self._image_ext[0])
        assert os.path.exists(image_path), 'path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'CLS-LOC',self._image_set+'.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        print(len(image_index))
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if False:
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

       gt_roidb = [self._load_imagenet_annotation(index, rid)
                    for rid, index in enumerate(self.image_index)]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb


    def _load_imagenet_annotation(self, index, rid):
        """
        Load image and bounding boxes info from txt files of imagenet.
        """
        filename = os.path.join(self._data_path, 'Annotations', 'train', index + '.xml')

        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')

        valid_objs = []
        for ix, obj in enumerate(objs):
            x1 = float(get_data_from_tag(obj, 'xmin'))
            y1 = float(get_data_from_tag(obj, 'ymin'))
            x2 = float(get_data_from_tag(obj, 'xmax'))
            y2 = float(get_data_from_tag(obj, 'ymax'))
            
            cls_name = str(get_data_from_tag(obj, "name")).lower().strip()
            if clsname in self._wnid:
                valid_objs.append(obj)
                cls = self._wnid_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
                if rid not in self.cls2img[cls]:
                    self.cls2img[cls].append(rid)
            
            

        num_objs = len(valid_objs)
        assert num_objs > 0

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(valid_objs):
            x1 = float(get_data_from_tag(obj, 'xmin'))
            y1 = float(get_data_from_tag(obj, 'ymin'))
            x2 = float(get_data_from_tag(obj, 'xmax'))
            y2 = float(get_data_from_tag(obj, 'ymax'))
            cls_name = str(get_data_from_tag(obj, "name")).lower().strip()
            assert clsname in self._wnid
            cls = self._wnid_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}



    def _get_default_path(self):
        """
        Return the default path where ImageNet is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'imagenet')

    def _get_comp_id(self):
        comp_id = ('meta' + '_' + self._salt if self.config['use_salt']
                   else 'meta')
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._devkit_path, 'results', 'CLS-LOC')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes, catIds, imgIds, classes):
        for cls_ind, cls in enumerate(classes):
            if cls == '__background__':
                continue
            print('Writing {} Imagenet results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(imgIds):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] , dets[k, 1] ,
                                       dets[k, 2] , dets[k, 3] ))

    def _do_python_eval(self, output_dir='output', catIds=None, imgIds=None, classes=None):
        annopath = os.path.join(
            self._devkit_path,
            'ILSVRC',
            'Annotations', 'train'
            '{:s}.xml')
        if imgIds != None:
            imagesetfile = imgIds
        else:
            imagesetfile = os.path.join(
                self._devkit_path,
                'ILSVRC',
                'ImageSets',
                'CLS-LOC',
                self._image_set + '.txt')

        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        for i, cls in enumerate(classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        return np.mean(aps)
       
 
    def evaluate_detections(self, all_boxes, output_dir, episodeIds, catIds, imgIds, classes, useCats=True):
        # classes = ['person','bike','car','cat','horse']
        classes = ['__background__'] + classes
        self._write_voc_results_file(all_boxes, catIds, imgIds, classes)
        mAP = self._do_python_eval(output_dir, catIds, imgIds, classes)

        if self.config['cleanup']:
            for cls in classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)
        return mAP

    def competition_mode(self, on):
        if True:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.imagenet('val', '')
    res = d.roidb
    from IPython import embed; embed()
