import numpy as np
import os
from data.base_dataset import BaseDataset
from .imlib import imlib
from multiprocessing.dummy import Pool
from tqdm import tqdm
from util.util import augment
import random
import cv2
import data.data_processing.synthetic_burst_generation as syn_burst_utils
from util.data_format_utils import torch_to_numpy, numpy_to_torch, npimage_to_torch


class ZTHDataset(BaseDataset):
	""" Canon RGB images from the "Zurich RAW to RGB mapping" dataset. You can download the full
	dataset (22 GB) from http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset. Alternatively, you can only download the
	Canon RGB images (5.5 GB) from https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip
	"""
	def __init__(self, opt, split='train', dataset_name='ZTH'):
		super(ZTHDataset, self).__init__(opt, split, dataset_name)

		if self.root == '':
			rootlist = ['/Data/dataset/Zurich-RAW-to-DSLR/']
			for root in rootlist:
				if os.path.isdir(root):
					self.root = root
					break
		
		self.imio = imlib(opt.mode, lib=opt.imlib)
		self.names, self.gt_dirs = self._get_image_dir(self.root, split)

		self.burst_size = 14
		self.crop_sz = opt.patch_size

		self.downsample_factor = 4
		self.burst_transformation_params = {'max_translation': 24.0,
											'max_rotation': 1.0,
											'max_shear': 0.0,
											'max_scale': 0.0,
											'border_crop': 24}

		self.image_processing_params = {'random_ccm': True, 'random_gains': True, 'smoothstep': True,
										'gamma': True,
										'add_noise': True}
		self.interpolation_type = 'bilinear'

		if split == 'train':
			self._getitem = self._getitem_train
		elif split == 'test': 
			self._getitem = self._getitem_test
		else:
			raise ValueError

		self.len_data = len(self.names)
		self.gt_images = [0] * len(self.names)

		read_images(self)


	def __getitem__(self, index):
		return self._getitem(index)

	def __len__(self):
		return self.len_data

	def _getitem_train(self, idx):
		""" returns:
            burst: Generated LR RAW burst, a torch tensor of shape
                   [burst_size, 4, self.crop_sz / (2*self.downsample_factor), self.crop_sz / (2*self.downsample_factor)]
                   The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer mosaick.
                   The extra factor 2 in the denominator (2*self.downsample_factor) corresponds to the mosaicking
                   operation.

            frame_gt: The HR RGB ground truth in the linear sensor space, a torch tensor of shape
                      [3, self.crop_sz, self.crop_sz]
		"""
		gt_img = self.gt_images[idx]
		gt_img = self._crop_patch(gt_img, \
			p = self.crop_sz + self.burst_transformation_params.get('border_crop') * 2)
		
		gt_img = self._augment(gt_img)

		gt_img = npimage_to_torch(gt_img, input_bgr=False)
		# gt_img = numpy_to_torch(gt_img) / 255
		# Generate RAW burst
		burst, frame_gt, burst_rgb, flow_vectors, meta_info = syn_burst_utils.rgb2rawburst(gt_img,
																						   self.burst_size,
																						   self.downsample_factor,
																						   burst_transformation_params=self.burst_transformation_params,
																						   image_processing_params=self.image_processing_params,
																						   interpolation_type=self.interpolation_type
																						   )

		if self.burst_transformation_params.get('border_crop') is not None:
			border_crop = self.burst_transformation_params.get('border_crop')
			frame_gt = frame_gt[:, border_crop:-border_crop, border_crop:-border_crop]

		# torch.Size([14, 4, 48, 48]) torch.Size([3, 384, 384])
		return {'burst': burst,
				'frame_gt': frame_gt,
				'meta_info': meta_info,
				'fname': self.names[idx]}

	def _getitem_test(self, idx):
		gt_img = self.gt_images[idx]
		gt_img = self._crop_patch(gt_img, \
			p = self.crop_sz + self.burst_transformation_params.get('border_crop') * 2)
		
		gt_img = npimage_to_torch(gt_img, input_bgr=False)
		# gt_img = numpy_to_torch(gt_img) / 255
		# Generate RAW burst
		burst, frame_gt, burst_rgb, flow_vectors, meta_info = syn_burst_utils.rgb2rawburst(gt_img,
																						   self.burst_size,
																						   self.downsample_factor,
																						   burst_transformation_params=self.burst_transformation_params,
																						   image_processing_params=self.image_processing_params,
																						   interpolation_type=self.interpolation_type
																						   )

		if self.burst_transformation_params.get('border_crop') is not None:
			border_crop = self.burst_transformation_params.get('border_crop')
			frame_gt = frame_gt[:, border_crop:-border_crop, border_crop:-border_crop]

		return {'burst': burst,
				'frame_gt': frame_gt,
				'meta_info': meta_info,
				'fname': self.names[idx]}

	def _crop_patch(self, gt, p):
		ih, iw = gt.shape[:2]
		pw = random.randrange(0, iw - p + 1)
		ph = random.randrange(0, ih - p + 1)
		return gt[ph:ph+p, pw:pw+p, :]

	def _augment(self, img):  # CxHxW
		hflip = random.random() < 0.5
		vflip = random.random() < 0.5
		rot90 = random.random() < 0.5
		if hflip:   img = img[:, ::-1, :].copy()
		if vflip:   img = img[::-1, :, :].copy()
		if rot90:   img = img.transpose(1, 0, 2)
		return np.ascontiguousarray(img)

	def _get_image_dir(self, dataroot, split=None):
		image_dirs = [] 
		image_names = []

		data_len = 46839 # 46839
		if split == 'train':
			image_dirs = ['{}/{:d}.jpg'.format(dataroot + 'train/canon', i) for i in range(data_len)]
			image_names = ['{:d}'.format(i) for i in range(data_len)]
		elif split == 'test':
			image_dirs = ['{}/{:d}.jpg'.format(dataroot + 'test/canon', i) for i in range(1204)]
			image_names = ['{:d}'.format(i) for i in range(1204)]
		# elif split == 'val':
		# 	image_dirs = ['{}/{:4d}.jpg'.format(dataroot + 'synburst_val_2022', i) for i in range(100)]
		# 	image_names = ['{:4d}'.format(i) for i in range(100)]
		else:
			raise ValueError

		image_dirs = sorted(image_dirs) #, reverse=True)[6:56]
		image_names = sorted(image_names) # , reverse=True)[6:56]

		return image_names, image_dirs


def iter_obj(num, objs):
	for i in range(num):
		yield (i, objs)

def imreader(arg):
	# Due to the memory (32 GB) limitation, here we only preload the raw images. 
	# If you have enough memory, you can also modify the code to preload the sRGB images to speed up the training process.
	i, obj = arg
	for _ in range(3):
		try:
			obj.gt_images[i] = cv2.imread(obj.gt_dirs[i])
			failed = False
			break
		except:
			failed = True
	if failed: print('%s fails!' % obj.names[i])

def read_images(obj):
	# may use `from multiprocessing import Pool` instead, but less efficient and
	# NOTE: `multiprocessing.Pool` will duplicate given object for each process.
	print('Starting to load images via multiple imreaders')
	pool = Pool() # use all threads by default
	for _ in tqdm(pool.imap(imreader, iter_obj(len(obj.names), obj)), total=len(obj.names)):
		pass
	pool.close()
	pool.join()
