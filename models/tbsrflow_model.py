import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from . import losses as L
import numpy as np
import random
from pwc import pwc_net
from .tbsr_model import EBSR


class TBSRFLOWModel(BaseModel):
	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		return parser

	def __init__(self, opt):
		super(TBSRFLOWModel, self).__init__(opt)

		self.opt = opt
		self.loss_names = ['EBSR_L1', 'EBSR_SW', 'EBSR_VGG', 'Total']
		self.visual_names = ['data_lr', 'data_gt', 'data_out', 'gt_warp']
		self.model_names = ['EBSR']
		self.optimizer_names = ['EBSR_optimizer_%s' % opt.optimizer]

		ebsr = EBSR(opt)
		self.netEBSR = N.init_net(ebsr, opt.init_type, opt.init_gain, opt.gpu_ids)
		# self.load_network_path(self.netEBSR, './ckpt/track1_model/EBSR_model_1.pth')

		pwcnet = pwc_net.PWCNET()
		self.netPWCNET = N.init_net(pwcnet, opt.init_type, opt.init_gain, opt.gpu_ids)
		self.set_requires_grad(self.netPWCNET, requires_grad=False)

		if self.isTrain: 
			self.optimizer_EBSR = optim.AdamW(self.netEBSR.parameters(), 
										 lr=opt.lr, 
										 betas=(0.9, 0.999), 
										 weight_decay=1e-4)

			self.optimizers = [self.optimizer_EBSR]

			self.criterionL1 = N.init_net(L.L1Loss(), gpu_ids=opt.gpu_ids)
			self.criterionVGG = N.init_net(L.VGGLoss(), gpu_ids=opt.gpu_ids)
			self.criterionSW = N.init_net(L.SWDLoss(), gpu_ids=opt.gpu_ids)

	def set_input(self, input):
		self.burst = input['burst'].to(self.device)
		# torch.Size([N, 14, 4, 48, 48])
		self.data_gt = input['frame_gt'].to(self.device)
		self.data_lr = self.burst[:, 0, :, :, :]
		self.image_paths = input['fname']

	def forward(self):
		input = self.flatten_raw_image_batch(self.burst)
		self.data_out = self.netEBSR(input)
		
		self.gt_warp, self.gt_mask = self.get_backwarp(self.data_out, self.data_gt, self.netPWCNET)

		if not self.isTrain and self.opt.self_ensemble:
			self.data_out = self.forward_spa_x8(input, self.netEBSR)

	def forward_spa_x8(self, x, model):
		# input_pattern = 'RGGB'
		def _transform(v, op):
			# aug_pattern, target_pattern = input_pattern, input_pattern
			v2np = v.data.cpu().numpy()
			h, w = v2np.shape[-2:]
			if op == 'v':
				tfnp = v2np[:, :, :, :, ::-1].copy()
				# aug_pattern = 'GRBG' # aug_pattern[1] + aug_pattern[0] + aug_pattern[3] + aug_pattern[2]
				out = tfnp[..., 0:h, 1:w-1]
				out = np.pad(out, [[0,0], [0,0], [0,0], [0,0], [0, 2]], 'constant', constant_values = (0,0))
			elif op == 'h':
				tfnp = v2np[:, :, :, ::-1, :].copy()
				# aug_pattern = 'GBRG' # aug_pattern[2] + aug_pattern[3] + aug_pattern[0] + aug_pattern[1]
				out = tfnp[..., 1:h-1, 0:w]
				out = np.pad(out, [[0,0], [0,0], [0,0], [0,2], [0, 0]], 'constant', constant_values = (0,0))
			elif op == 't':
				tfnp = v2np.transpose((0, 1, 2, 4, 3)).copy()
				out = tfnp
				# aug_pattern = 'RGGB' # aug_pattern[0] + aug_pattern[2] + aug_pattern[1] + aug_pattern[3]
			ret = torch.Tensor(out).to(self.device)
			return ret

		def _transform_inverse(v, op):
			v2np = v.data.cpu().numpy()
			h, w = v2np.shape[-2:]
			if op == 'v':
				out = np.pad(v2np, [[0,0], [0,0], [0,0], [7, 0]], 'constant', constant_values = (0,0))
				out = out[..., 0:h, 0:w]
				tfnp = out[:, :, :, ::-1].copy()
			elif op == 'h':
				out = np.pad(v2np, [[0,0], [0,0], [7,0], [0, 0]], 'constant', constant_values = (0,0))
				out = out[..., 0:h, 0:w]
				tfnp = out[:, :, ::-1, :].copy()

			elif op == 't':
				tfnp = v2np.transpose((0, 1, 3, 2)).copy()

			ret = torch.Tensor(tfnp).to(self.device)
			return ret

		lr_list = [x]
		for tf in 'v', 'h', 't':
			lr_list.extend([_transform(t, tf) for t in lr_list])

		sr_list = [model(aug) for aug in lr_list]
		for i in range(len(sr_list)):
			if i > 3:
				sr_list[i] = _transform_inverse(sr_list[i], 't')
			if i % 4 > 1:
				sr_list[i] = _transform_inverse(sr_list[i], 'h')
			if (i % 4) % 2 == 1:
				sr_list[i] = _transform_inverse(sr_list[i], 'v')

		output_cat = torch.cat(sr_list, dim=0)
		output = output_cat.mean(dim=0, keepdim=True)

		return output

	def backward(self):
		self.loss_EBSR_L1 = self.criterionL1(self.data_out*self.gt_mask, self.gt_warp).mean()
		self.loss_EBSR_SW = self.criterionSW(self.data_out*self.gt_mask, self.gt_warp).mean() * 0.4
		self.loss_EBSR_VGG = self.criterionVGG(self.data_out*self.gt_mask, self.gt_warp).mean() * 0.02

		self.loss_Total = self.loss_EBSR_L1 + self.loss_EBSR_SW + self.loss_EBSR_VGG
		self.loss_Total.backward()

	def optimize_parameters(self):
		self.forward()
		self.optimizer_EBSR.zero_grad()
		self.backward()
		self.optimizer_EBSR.step()

	def flatten_raw_image_batch(self, im_raw_4ch):
		im_out = torch.zeros((im_raw_4ch.shape[0], im_raw_4ch.shape[1], 1, im_raw_4ch.shape[3] * 2, im_raw_4ch.shape[4] * 2), dtype=im_raw_4ch.dtype).to(im_raw_4ch.device)
		im_out[:, :, 0, 0::2, 0::2] = im_raw_4ch[:, :, 0, :, :]
		im_out[:, :, 0, 0::2, 1::2] = im_raw_4ch[:, :, 1, :, :]
		im_out[:, :, 0, 1::2, 0::2] = im_raw_4ch[:, :, 2, :, :]
		im_out[:, :, 0, 1::2, 1::2] = im_raw_4ch[:, :, 3, :, :]

		return im_out
