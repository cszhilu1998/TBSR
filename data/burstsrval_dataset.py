import os
import torch
import torch.nn.functional as F
import random
from .burstsr_dataset import SamsungRAWImage, flatten_raw_image, pack_raw_image
from .imlib import imlib
from data.base_dataset import BaseDataset


class BurstSRVALDataset(BaseDataset):
	""" Real-world burst super-resolution dataset. """
	def __init__(self, opt, split='test', dataset_name='BurstSRVAL'):
		super(BurstSRVALDataset, self).__init__(opt, split, dataset_name)
		"""
		args:
			root : path of the root directory
			burst_size : Burst size. Maximum allowed burst size is 14.
			crop_sz: Size of the extracted crop. Maximum allowed crop size is 80
			center_crop: Whether to extract a random crop, or a centered crop.
			random_flip: Whether to apply random horizontal and vertical flip
			split: Can be 'train' or 'val'
		"""

		if self.root == '':
			rootlist = ['/Data/dataset/Burst_SR/']
			for root in rootlist:
				if os.path.isdir(root):
					self.root = root
					break

		self.burst_size = 14
		self.split = split
		self.substract_black_level = True
		self.white_balance = False

		self.imio = imlib(opt.mode, lib=opt.imlib)
		self.root = self.root + 'val'
		self.center_crop = False
		self.random_flip = False
		self.crop_sz = 80
		# self._getitem = self._getitem_train

		self.burst_list = self._get_burst_list()

	def _get_burst_list(self):
		burst_list = sorted(os.listdir('{}'.format(self.root)))

		return burst_list

	def get_burst_info(self, burst_id):
		burst_info = {'burst_size': 14, 'burst_name': self.burst_list[burst_id]}
		return burst_info

	def _get_raw_image(self, burst_id, im_id):
		raw_image = SamsungRAWImage.load('{}/{}/samsung_{:02d}'.format(self.root, self.burst_list[burst_id], im_id))
		return raw_image

	def get_burst(self, burst_id, im_ids, info=None):
		frames = [self._get_raw_image(burst_id, i) for i in im_ids]

		if info is None:
			info = self.get_burst_info(burst_id)

		return frames, info

	def _sample_images(self):
		burst_size = 14
		ids = [i for i in range(burst_size)]

		# ids = random.sample(range(1, burst_size), k=self.burst_size - 1)
		# ids = [0, ] + ids
		return ids

	def __len__(self):
		return len(self.burst_list)

	def __getitem__(self, index):
		# Sample the images in the burst, in case a burst_size < 14 is used.
		im_ids = self._sample_images()

		# Read the burst images along with HR ground truth
		frames, meta_info = self.get_burst(index, im_ids)

		# Extract crop if needed
		# if frames[0].shape()[-1] != self.crop_sz:
		# 	if getattr(self, 'center_crop', False):
		# 		r1 = (frames[0].shape()[-2] - self.crop_sz) // 2
		# 		c1 = (frames[0].shape()[-1] - self.crop_sz) // 2
		# 	else:
		# 		r1 = random.randint(0, frames[0].shape()[-2] - self.crop_sz)
		# 		c1 = random.randint(0, frames[0].shape()[-1] - self.crop_sz)
		# 	r2 = r1 + self.crop_sz
		# 	c2 = c1 + self.crop_sz

		# 	frames = [im.get_crop(r1, r2, c1, c2) for im in frames]

		# Load the RAW image data
		burst_image_data = [im.get_image_data(normalize=True, substract_black_level=self.substract_black_level,
											  white_balance=self.white_balance) for im in frames]

		# if self.random_flip:
		# 	burst_image_data = [flatten_raw_image(im) for im in burst_image_data]

		# 	pad = [0, 0, 0, 0]
		# 	if random.random() > 0.5:
		# 		burst_image_data = [im.flip([1, ])[:, 1:-1].contiguous() for im in burst_image_data]
		# 		pad[1] = 1

		# 	if random.random() > 0.5:
		# 		burst_image_data = [im.flip([0, ])[1:-1, :].contiguous() for im in burst_image_data]
		# 		pad[3] = 1

		# 	burst_image_data = [pack_raw_image(im) for im in burst_image_data]
		# 	burst_image_data = [F.pad(im.unsqueeze(0), pad, mode='replicate').squeeze(0) for im in burst_image_data]

		burst_image_meta_info = frames[0].get_all_meta_data()

		burst_image_meta_info['black_level_subtracted'] = self.substract_black_level
		burst_image_meta_info['while_balance_applied'] = self.white_balance
		burst_image_meta_info['norm_factor'] = frames[0].norm_factor

		burst = torch.stack(burst_image_data, dim=0)

		burst_exposure = frames[0].get_exposure_time()

		burst_f_number = frames[0].get_f_number()

		burst_iso = frames[0].get_iso()

		burst_image_meta_info['exposure'] = burst_exposure
		burst_image_meta_info['f_number'] = burst_f_number
		burst_image_meta_info['iso'] = burst_iso

		burst = burst.float()

		meta_info_burst = burst_image_meta_info

		for k, v in meta_info_burst.items():
			if isinstance(v, (list, tuple)):
				meta_info_burst[k] = torch.tensor(v)

		return {'burst': burst,
				'frame_gt': burst[0],
				'meta_info_burst': meta_info_burst,
				'meta_info_gt': meta_info_burst,
				'fname': meta_info['burst_name']}
		# return burst, meta_info_burst