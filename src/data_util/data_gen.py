# coding=utf-8
__author__ = 'moonkey'

import os
import numpy as np
from PIL import Image
from collections import Counter
import pickle as cPickle
import json
import random, math
from data_util.bucketdata import BucketData
from imgaug import augmenters as iaa

vocabulary_dict=json.load(open('/home/dilligencer/Attention-OCR/competition/vocabulary.json'))
vocabulary=vocabulary_dict['vocabulary']

class DataGen(object):
    GO = 1
    EOS = 2

    def __init__(self,
                 data_root, annotation_fn,
                 evaluate = False,
                 valid_target_len = float('inf'),
                 img_width_range = (39,936),
                 word_len = 50):
        """
        :param data_root:
        :param annotation_fn:
        :param lexicon_fn:
        :param img_width_range: only needed for training set
        :return:
        """

        img_height = 48
        self.data_root = data_root
        if os.path.exists(annotation_fn):
            self.annotation_path = annotation_fn  # abs path
        else:
            self.annotation_path = os.path.join(data_root, annotation_fn)  # relative path

        if evaluate:
            self.bucket_specs = [(int(math.floor(64 / 4)), int(word_len + 2)), (int(math.floor(128 / 4)), int(word_len + 2)),
                                 (int(math.floor(256 / 4)), int(word_len + 2)), (int(math.floor(512 / 4)), int(word_len + 2)),
                                 (int(math.floor(img_width_range[1] / 4)), int(word_len + 2))]
        else:
            self.bucket_specs = [(int(64 / 4), 9 + 2), (int(128 / 4), 15 + 2),
                             (int(256 / 4), 17 + 2), (int(512 / 4), 25 + 2),
                             (int(math.ceil(img_width_range[1] / 4)), word_len + 2)]

        self.bucket_min_width, self.bucket_max_width = img_width_range
        self.image_height = img_height
        self.valid_target_len = valid_target_len

        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

    def clear(self):
        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

    def get_size(self):
        with open(self.annotation_path, 'r') as ann_file:
            return len(ann_file.readlines())

    def gen(self, batch_size):
        valid_target_len = self.valid_target_len
        with open(self.annotation_path, 'r') as ann_file:
            lines = ann_file.readlines()
            random.shuffle(lines)
            for l in lines:
                img_path = l.strip().split()[0]
                lex = l.strip().split(img_path + ' ')[-1]
                try:
                    img_bw, word = self.read_data(img_path, lex)
                    if random.random() < 0.5:
                        augMethodsNumber = int(math.floor(random.random() * 2) + 1)
                        seq = iaa.SomeOf(augMethodsNumber, [
                            iaa.CropAndPad(
                                px=((0, 30), (0, 10), (0, 30), (0, 10)),
                                pad_mode=["constant", "edge"],
                                pad_cval=(0, 128),
                            ),
                            iaa.Add((-100, 100)),
                            iaa.Fliplr(0.5),
                            iaa.Flipud(0.5),
                            iaa.Superpixels(p_replace=random.random()*0.1, n_segments=4),
                            # iaa.Grayscale(alpha=(0.0, 0.1)),
                            iaa.GaussianBlur(sigma=(0.0, 3.0)),
                            iaa.AverageBlur(k=(1, 4)),
                            iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
                            iaa.AdditiveGaussianNoise(scale=0.1 * 255),
                            iaa.Dropout(p=(0, 0.2)),
                            iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))
                        ], random_order=True)
                        img_bw = seq.augment_images(img_bw)
                    if valid_target_len < float('inf'):
                        word = word[:valid_target_len + 1]
                    width = img_bw.shape[-2]

                    # TODO:resize if > 320
                    b_idx = min(width, self.bucket_max_width)
                    bs = self.bucket_data[b_idx].append(img_bw, word, os.path.join(self.data_root,img_path))
                    if bs >= batch_size:
                        b = self.bucket_data[b_idx].flush_out(
                                self.bucket_specs,
                                valid_target_length=valid_target_len,
                                go_shift=1)
                        if b is not None:
                            yield b
                        else:
                            assert False, 'no valid bucket of width %d'%width
                except IOError:
                    pass # ignore error images
                    #with open('error_img.txt', 'a') as ef:
                    #    ef.write(img_path + '\n')
        self.clear()

    def read_data(self, img_path, lex):
        assert 0 < len(lex) < self.bucket_specs[-1][1]
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        with open(os.path.join(self.data_root, img_path), 'rb') as img_file:
            img = Image.open(img_file)
            w, h = img.size
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < float(self.bucket_min_width) / self.image_height:
                img = img.resize(
                    (self.bucket_min_width, self.image_height),
                    Image.ANTIALIAS)
            elif aspect_ratio > float(
                    self.bucket_max_width) / self.image_height:
                img = img.resize(
                    (self.bucket_max_width, self.image_height),
                    Image.ANTIALIAS)
            elif h != self.image_height:
                img = img.resize(
                    (int(aspect_ratio * self.image_height), self.image_height),
                    Image.ANTIALIAS)

            img_bw = img.convert('RGB')
            img_bw = np.asarray(img_bw, dtype=np.uint8)
            img_bw = img_bw[np.newaxis, :]

        # 'a':97, '0':48
        word = [self.GO]
        for c in lex:
            index=vocabulary.index(c)
            word.append(index+3)
        word.append(self.EOS)
        word = np.array(word, dtype=np.int32)
        # word = np.array( [self.GO] +
        # [ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3
        # for c in lex] + [self.EOS], dtype=np.int32)

        return img_bw, word


def test_gen():
    print('testing gen_valid')
    # s_gen = EvalGen('../../data/evaluation_data/svt', 'test.txt')
    # s_gen = EvalGen('../../data/evaluation_data/iiit5k', 'test.txt')
    # s_gen = EvalGen('../../data/evaluation_data/icdar03', 'test.txt')
    s_gen = DataGen('../../data/evaluation_data/icdar13', 'test.txt')
    count = 0
    for batch in s_gen.gen(1):
        count += 1
        print(str(batch['bucket_id']) + ' ' + str(batch['data'].shape[2:]))
        assert batch['data'].shape[2] == 48
    print(count)


if __name__ == '__main__':
    test_gen()
