import os
import torch
import cv2
import numpy as np
import pickle as pkl
import torch.nn.functional as F
import random
from data.base_dataset import BaseDataset
from .imlib import imlib


class BurstSRDataset(BaseDataset):
	def __init__(self, opt, split='train', dataset_name='BurstSR'):
		super(BurstSRDataset, self).__init__(opt, split, dataset_name)
		"""
		args:
			root : path of the root directory
			burst_size : Burst size. Maximum allowed burst size is 14.
			crop_sz: Size of the extracted crop. Maximum allowed crop size is 80
			center_crop: Whether to extract a random crop, or a centered crop.
			random_flip: Whether to apply random horizontal and vertical flip
			split: Can be 'train' or 'val'
		"""
		# assert burst_size <= 14, 'burst_sz must be less than or equal to 14'
		# assert crop_sz <= 80, 'crop_sz must be less than or equal to 80'
		# assert split in ['train', 'val']

		if self.root == '':
			rootlist = ['/mnt/disk10T/dataset/Burst_SR/']
			for root in rootlist:
				if os.path.isdir(root):
					self.root = root
					break

		self.burst_size = 14
		self.split = split
		self.substract_black_level = True
		self.white_balance = False
		self.imio = imlib(opt.mode, lib=opt.imlib)

		if split == 'train':
			self.root = self.root + 'train'
			self.center_crop = False
			self.random_flip = True
			self.crop_sz = opt.patch_size // 8
		elif split == 'test': 
			self.root = self.root + 'val'
			self.center_crop = False
			self.random_flip = False
			self.crop_sz = 80
		else:
			raise ValueError
		
		self._getitem = self._getitem_train
		self.burst_list = self._get_burst_list()
		self.len_data = len(self.burst_list)

	def __getitem__(self, index):
		return self._getitem(index)

	def __len__(self):
		return self.len_data

	def _get_burst_list(self):
		burst_list = sorted(os.listdir('{}'.format(self.root)))
		# if self.split == 'test': 
		# 	burst_list = burst_list[::10]
		return burst_list

	def get_burst_info(self, burst_id):
		burst_info = {'burst_size': 14, 'burst_name': self.burst_list[burst_id]}
		return burst_info

	def _get_raw_image(self, burst_id, im_id):
		raw_image = SamsungRAWImage.load('{}/{}/samsung_{:02d}'.format(self.root, self.burst_list[burst_id], im_id))
		return raw_image

	def _get_gt_image(self, burst_id):
		canon_im = CanonImage.load('{}/{}/canon'.format(self.root, self.burst_list[burst_id]))
		return canon_im

	def get_burst(self, burst_id, im_ids, info=None):
		frames = [self._get_raw_image(burst_id, i) for i in im_ids]

		gt = self._get_gt_image(burst_id)
		if info is None:
			info = self.get_burst_info(burst_id)

		return frames, gt, info

	def _sample_images(self):
		burst_size = 14

		ids = random.sample(range(1, burst_size), k=self.burst_size - 1)
		ids = [0, ] + ids
		return ids

	def _getitem_train(self, index):
		# Sample the images in the burst, in case a burst_size < 14 is used.
		im_ids = self._sample_images()

		# Read the burst images along with HR ground truth
		frames, gt, meta_info = self.get_burst(index, im_ids)

		# Extract crop if needed
		if frames[0].shape()[-1] != self.crop_sz:
			# if getattr(self, 'center_crop', False):
			# 	r1 = (frames[0].shape()[-2] - self.crop_sz) // 2
			# 	c1 = (frames[0].shape()[-1] - self.crop_sz) // 2
			# else:
			r1 = random.randint(0, frames[0].shape()[-2] - self.crop_sz)
			c1 = random.randint(0, frames[0].shape()[-1] - self.crop_sz)
			r2 = r1 + self.crop_sz
			c2 = c1 + self.crop_sz

			scale_factor = gt.shape()[-1] // frames[0].shape()[-1]
			frames = [im.get_crop(r1, r2, c1, c2) for im in frames]

			gt = gt.get_crop(scale_factor * r1, scale_factor * r2, scale_factor * c1, scale_factor * c2)

		# Load the RAW image data
		burst_image_data = [im.get_image_data(normalize=True, substract_black_level=self.substract_black_level,
											  white_balance=self.white_balance) for im in frames]

		# Convert to tensor
		gt_image_data = gt.get_image_data(normalize=True, white_balance=self.white_balance,
										  substract_black_level=self.substract_black_level)

		if self.random_flip:
			burst_image_data = [flatten_raw_image(im) for im in burst_image_data]

			pad = [0, 0, 0, 0]
			if random.random() > 0.5:
				burst_image_data = [im.flip([1, ])[:, 1:-1].contiguous() for im in burst_image_data]
				gt_image_data = gt_image_data.flip([2, ])[:, :, 2:-2].contiguous()
				pad[1] = 1

			if random.random() > 0.5:
				burst_image_data = [im.flip([0, ])[1:-1, :].contiguous() for im in burst_image_data]
				gt_image_data = gt_image_data.flip([1, ])[:, 2:-2, :].contiguous()
				pad[3] = 1

			burst_image_data = [pack_raw_image(im) for im in burst_image_data]
			burst_image_data = [F.pad(im.unsqueeze(0), pad, mode='replicate').squeeze(0) for im in burst_image_data]

			gt_image_data = F.pad(gt_image_data.unsqueeze(0), [4 * p for p in pad], mode='replicate').squeeze(0)

		burst_image_meta_info = frames[0].get_all_meta_data()

		burst_image_meta_info['black_level_subtracted'] = self.substract_black_level
		burst_image_meta_info['while_balance_applied'] = self.white_balance
		burst_image_meta_info['norm_factor'] = frames[0].norm_factor

		gt_image_meta_info = gt.get_all_meta_data()

		burst = torch.stack(burst_image_data, dim=0)

		burst_exposure = frames[0].get_exposure_time()
		canon_exposure = gt.get_exposure_time()

		burst_f_number = frames[0].get_f_number()
		canon_f_number = gt.get_f_number()

		burst_iso = frames[0].get_iso()
		canon_iso = gt.get_iso()

		# Normalize the GT image to account for differences in exposure, ISO etc
		light_factor_burst = burst_exposure * burst_iso / (burst_f_number ** 2)
		light_factor_canon = canon_exposure * canon_iso / (canon_f_number ** 2)

		exp_scale_factor = (light_factor_burst / light_factor_canon)
		gt_image_data = gt_image_data * exp_scale_factor

		gt_image_meta_info['black_level_subtracted'] = self.substract_black_level
		gt_image_meta_info['while_balance_applied'] = self.white_balance
		gt_image_meta_info['norm_factor'] = gt.norm_factor / exp_scale_factor

		burst_image_meta_info['exposure'] = burst_exposure
		burst_image_meta_info['f_number'] = burst_f_number
		burst_image_meta_info['iso'] = burst_iso

		gt_image_meta_info['exposure'] = canon_exposure
		gt_image_meta_info['f_number'] = canon_f_number
		gt_image_meta_info['iso'] = canon_iso

		burst = burst.float()
		frame_gt = gt_image_data.float()

		meta_info_burst = burst_image_meta_info
		meta_info_gt = gt_image_meta_info

		del meta_info_gt['crop_info']

		for k, v in meta_info_gt.items():
			if isinstance(v, (list, tuple)):
				meta_info_gt[k] = torch.tensor(v)

		for k, v in meta_info_burst.items():
			if isinstance(v, (list, tuple)):
				meta_info_burst[k] = torch.tensor(v)

		return {'burst': burst,
				'frame_gt': frame_gt,
				'meta_info_burst': meta_info_burst,
				'meta_info_gt': meta_info_gt,
				'fname': meta_info['burst_name']}
		# return burst, frame_gt, meta_info_burst, meta_info_gt


class SamsungRAWImage:
	@staticmethod
	def load(path):
		im_raw = cv2.imread('{}/im_raw.png'.format(path), cv2.IMREAD_UNCHANGED)
		im_raw = np.transpose(im_raw, (2, 0, 1)).astype(np.int16)
		im_raw = torch.from_numpy(im_raw)
		meta_data = pkl.load(open('{}/meta_info.pkl'.format(path), "rb", -1))

		return SamsungRAWImage(im_raw, meta_data['black_level'], meta_data['cam_wb'],
							   meta_data['daylight_wb'], meta_data['color_matrix'], meta_data['exif_data'],
							   meta_data.get('crop_info', None), meta_data.get('im_preview', None))

	def __init__(self, im_raw, black_level, cam_wb, daylight_wb, color_matrix, exif_data, crop_info=None,
				 im_preview=None):
		self.im_raw = im_raw
		self.black_level = black_level
		self.cam_wb = cam_wb
		self.daylight_wb = daylight_wb
		self.color_matrix = color_matrix
		self.exif_data = exif_data
		self.crop_info = crop_info
		self.im_preview = im_preview

		self.norm_factor = 1023.0

	def get_all_meta_data(self):
		return {'black_level': self.black_level, 'cam_wb': self.cam_wb, 'daylight_wb': self.daylight_wb,
				'color_matrix': self.color_matrix.tolist()}

	def get_exposure_time(self):
		return self.exif_data['Image ExposureTime'].values[0].decimal()

	def get_noise_profile(self):
		noise = self.exif_data['Image Tag 0xC761'].values
		noise = [n[0] for n in noise]
		noise = np.array(noise).reshape(3, 2)
		return noise

	def get_f_number(self):
		return self.exif_data['Image FNumber'].values[0].decimal()

	def get_iso(self):
		return self.exif_data['Image ISOSpeedRatings'].values[0]

	def get_image_data(self, substract_black_level=False, white_balance=False, normalize=False):
		im_raw = self.im_raw.float()

		if substract_black_level:
			im_raw = im_raw - torch.tensor(self.black_level).view(4, 1, 1)

		if white_balance:
			im_raw = im_raw * torch.tensor(self.cam_wb).view(4, 1, 1)

		if normalize:
			im_raw = im_raw / self.norm_factor

		return im_raw

	def shape(self):
		shape = (4, self.im_raw.shape[1], self.im_raw.shape[2])
		return shape

	def crop_image(self, r1, r2, c1, c2):
		self.im_raw = self.im_raw[:, r1:r2, c1:c2]

	def get_crop(self, r1, r2, c1, c2):
		im_raw = self.im_raw[:, r1:r2, c1:c2]

		if self.im_preview is not None:
			im_preview = self.im_preview[2*r1:2*r2, 2*c1:2*c2]
		else:
			im_preview = None

		return SamsungRAWImage(im_raw, self.black_level, self.cam_wb, self.daylight_wb, self.color_matrix,
							   self.exif_data, im_preview=im_preview)

	def postprocess(self, return_np=True, norm_factor=None):
		# Convert to rgb
		im = torch.from_numpy(self.im_raw.astype(np.float32))

		im = (im - torch.tensor(self.black_level).view(4, 1, 1)) * torch.tensor(self.cam_wb).view(4, 1, 1)

		if norm_factor is None:
			im = im / im.max()
		else:
			im = im / norm_factor

		im = torch.stack((im[0], (im[1] + im[2])/2, im[3]), dim=0)

		im_out = im.clamp(0.0, 1.0)

		if return_np:
			im_out = im_out.permute(1, 2, 0).numpy() * 255.0
			im_out = im_out.astype(np.uint8)
		return im_out


class CanonImage:
	@staticmethod
	def load(path):
		im_raw = cv2.imread('{}/im_raw.png'.format(path), cv2.IMREAD_UNCHANGED)
		im_raw = np.transpose(im_raw, (2, 0, 1)).astype(np.int16)
		im_raw = torch.from_numpy(im_raw)
		meta_data = pkl.load(open('{}/meta_info.pkl'.format(path), "rb", -1))

		return CanonImage(im_raw.float(), meta_data['black_level'], meta_data['cam_wb'],
						  meta_data['daylight_wb'], meta_data['rgb_xyz_matrix'], meta_data['exif_data'],
						  meta_data.get('crop_info', None))

	def __init__(self, im_raw, black_level, cam_wb, daylight_wb, rgb_xyz_matrix, exif_data, crop_info=None):
		super(CanonImage, self).__init__()
		self.im_raw = im_raw

		if len(black_level) == 4:
			black_level = [black_level[0], black_level[1], black_level[3]]
		self.black_level = black_level

		if len(cam_wb) == 4:
			cam_wb = [cam_wb[0], cam_wb[1], cam_wb[3]]
		self.cam_wb = cam_wb

		if len(daylight_wb) == 4:
			daylight_wb = [daylight_wb[0], daylight_wb[1], daylight_wb[3]]
		self.daylight_wb = daylight_wb

		self.rgb_xyz_matrix = rgb_xyz_matrix
		self.xyz_srgb_matrix = torch.tensor([3.2404542, -1.5371385, -0.4985314,
											 -0.9692660,  1.8760108,  0.0415560,
											 0.0556434, -0.2040259,  1.0572252]).view(3, 3)
		self.exif_data = exif_data
		self.crop_info = crop_info

		self.norm_factor = 16383

	def shape(self):
		shape = (3, self.im_raw.shape[1], self.im_raw.shape[2])
		return shape

	def get_all_meta_data(self):
		return {'black_level': self.black_level, 'cam_wb': self.cam_wb, 'daylight_wb': self.daylight_wb,
				'rgb_xyz_matrix': self.rgb_xyz_matrix.tolist(), 'crop_info': self.crop_info,
				'norm_factor': self.norm_factor}

	def get_exposure_time(self):
		return self.exif_data['EXIF ExposureTime'].values[0].decimal()

	def get_f_number(self):
		return self.exif_data['EXIF FNumber'].values[0].decimal()

	def get_iso(self):
		return self.exif_data['EXIF ISOSpeedRatings'].values[0]

	def get_image_data(self, substract_black_level=False, white_balance=False, normalize=False):
		im_raw = self.im_raw.float()

		if substract_black_level:
			im_raw = im_raw - torch.tensor(self.black_level).view(3, 1, 1)

		if white_balance:
			im_raw = im_raw * torch.tensor(self.cam_wb).view(3, 1, 1) / 1024.0

		if normalize:
			im_raw = im_raw / self.norm_factor

		return im_raw

	def set_image_data(self, im_data):
		self.im_raw = im_data

	def crop_image(self, r1, r2, c1, c2):
		self.im_raw = self.im_raw[:, r1:r2, c1:c2]

	def get_crop(self, r1, r2, c1, c2):
		im_raw = self.im_raw[:, r1:r2, c1:c2]
		return CanonImage(im_raw, self.black_level, self.cam_wb, self.daylight_wb, self.rgb_xyz_matrix,
						  self.exif_data, self.crop_info)

	def set_crop_info(self, crop_info):
		self.crop_info = crop_info

	def resize(self, size=None, scale_factor=None):

		self.im_raw = F.interpolate(self.im_raw.unsqueeze(0), size=size, scale_factor=scale_factor,
									mode='bilinear').squeeze(0)

	def postprocess(self, return_np=True):
		# Convert to rgb
		im = self.im_raw

		im = (im - torch.tensor(self.black_level).view(3, 1, 1)).float() * torch.tensor(self.cam_wb).view(3, 1, 1)

		im_out = im / im.max()
		im_out = im_out.clamp(0.0, 1.0)

		if return_np:
			im_out = im_out.permute(1, 2, 0).numpy() * 255.0
			im_out = im_out.astype(np.uint8)
		return im_out


def load_txt(path):
	with open(path, 'r') as fh:
		out = [d.rstrip() for d in fh.readlines()]

	return out


def pack_raw_image(im_raw):
	if isinstance(im_raw, np.ndarray):
		im_out = np.zeros_like(im_raw, shape=(4, im_raw.shape[0] // 2, im_raw.shape[1] // 2))
	elif isinstance(im_raw, torch.Tensor):
		im_out = torch.zeros((4, im_raw.shape[0] // 2, im_raw.shape[1] // 2), dtype=im_raw.dtype)
	else:
		raise Exception

	im_out[0, :, :] = im_raw[0::2, 0::2]
	im_out[1, :, :] = im_raw[0::2, 1::2]
	im_out[2, :, :] = im_raw[1::2, 0::2]
	im_out[3, :, :] = im_raw[1::2, 1::2]
	return im_out


def flatten_raw_image(im_raw_4ch):
	if isinstance(im_raw_4ch, np.ndarray):
		im_out = np.zeros_like(im_raw_4ch, shape=(im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2))
	elif isinstance(im_raw_4ch, torch.Tensor):
		im_out = torch.zeros((im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2), dtype=im_raw_4ch.dtype)
	else:
		raise Exception

	im_out[0::2, 0::2] = im_raw_4ch[0, :, :]
	im_out[0::2, 1::2] = im_raw_4ch[1, :, :]
	im_out[1::2, 0::2] = im_raw_4ch[2, :, :]
	im_out[1::2, 1::2] = im_raw_4ch[3, :, :]

	return im_out

def pack_raw_image_batch(im_raw):
	im_out = torch.zeros((im_raw.shape[0], im_raw.shape[1], 4, im_raw.shape[2] // 2, im_raw.shape[3] // 2), dtype=im_raw.dtype).to(im_raw.device)

	im_out[:, :, 0, :, :] = im_raw[:, :, 0::2, 0::2]
	im_out[:, :, 1, :, :] = im_raw[:, :, 0::2, 1::2]
	im_out[:, :, 2, :, :] = im_raw[:, :, 1::2, 0::2]
	im_out[:, :, 3, :, :] = im_raw[:, :, 1::2, 1::2]
	return im_out


def flatten_raw_image_batch(im_raw_4ch):
	# print(im_raw_4ch.device)
	im_out = torch.zeros((im_raw_4ch.shape[0], im_raw_4ch.shape[1], 1, im_raw_4ch.shape[3] * 2, im_raw_4ch.shape[4] * 2), dtype=im_raw_4ch.dtype).to(im_raw_4ch.device)
	# print(im_out.device)
	im_out[:, :, 0, 0::2, 0::2] = im_raw_4ch[:, :, 0, :, :]
	im_out[:, :, 0, 0::2, 1::2] = im_raw_4ch[:, :, 1, :, :]
	im_out[:, :, 0, 1::2, 0::2] = im_raw_4ch[:, :, 2, :, :]
	im_out[:, :, 0, 1::2, 1::2] = im_raw_4ch[:, :, 3, :, :]

	return im_out
