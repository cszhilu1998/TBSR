import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from . import losses as L
import numpy as np
import math
import functools
import torchvision.ops as ops


try:
	from models.non_local.non_local_cross_dot_product import NONLocalBlock2D as NonLocalCross
	from models.non_local.non_local_dot_product import NONLocalBlock2D as NonLocal
except ImportError:
	raise ImportError('Failed to import Non_Local module.')



class TBSRModel(BaseModel):
	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		return parser

	def __init__(self, opt):
		super(TBSRModel, self).__init__(opt)

		self.opt = opt
		self.loss_names = ['EBSR_L1', 'EBSR_VGG', 'EBSR_SSIM', 'Total']
		self.visual_names = ['data_lr', 'data_gt', 'data_out']

		self.model_names = ['EBSR']
		self.optimizer_names = ['EBSR_optimizer_%s' % opt.optimizer]

		ebsr = EBSR(opt)
		self.netEBSR= N.init_net(ebsr, opt.init_type, opt.init_gain, opt.gpu_ids)

		if self.isTrain: 

			self.optimizer_EBSR = optim.AdamW(self.netEBSR.parameters(), 
										 lr=opt.lr, 
										 betas=(0.9, 0.999), 
										 weight_decay=1e-4)

			self.optimizers = [self.optimizer_EBSR]

			self.criterionL1 = N.init_net(L.L1Loss(), gpu_ids=opt.gpu_ids)
			self.criterionVGG = N.init_net(L.VGGLoss(), gpu_ids=opt.gpu_ids)
			self.criterionSSIM = N.init_net(L.SSIMLoss(), gpu_ids=opt.gpu_ids)

	def set_input(self, input):
		self.burst = input['burst'].to(self.device)
		# torch.Size([N, 14, 4, 48, 48])
		self.data_gt = input['frame_gt'].to(self.device)
		self.data_lr = self.burst[:, 0, :, :, :]
		self.image_paths = input['fname']
		# self.meta_info_burst = input['meta_info_burst']

	def forward(self):
		input = self.flatten_raw_image_batch(self.burst)
		self.data_out = self.netEBSR(input)

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
		self.loss_EBSR_L1 = self.criterionL1(self.data_out, self.data_gt).mean()
		self.loss_EBSR_VGG = 0
		self.loss_EBSR_SSIM = 0 

		self.loss_Total = self.loss_EBSR_L1 + self.loss_EBSR_VGG + self.loss_EBSR_SSIM
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

class EBSR(nn.Module):
	def __init__(self, args):
		super(EBSR, self).__init__()
		num_groups = 8
		num_blocks = 8
		nf = 128
		nframes = 14
		front_RBs = 5
		deform_groups = 8
		self.center = 0

		wn = lambda x: torch.nn.utils.weight_norm(x)
		WARB = functools.partial(WideActResBlock, nf=nf)

		#### extract features (for each frame)
		self.conv_first = wn(nn.Conv2d(1, nf, 3, 1, 1, bias=True))
		self.feature_extraction = make_layer(WARB, front_RBs)
		self.fea_L2_conv1 = wn(nn.Conv2d(nf, nf*2, 3, 2, 1, bias=True))
		self.fea_L3_conv1 = wn(nn.Conv2d(nf*2, nf*4, 3, 2, 1, bias=True))

		############### Feature Enhanced PCD Align #####################
		# Top layers
		self.toplayer = wn(nn.Conv2d(nf*4, nf, kernel_size=1, stride=1, padding=0))
		# Smooth layers
		self.smooth1 = wn(nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1))
		self.smooth2 = wn(nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1))
		# Lateral layers
		self.latlayer1 = wn(nn.Conv2d(nf*2, nf, kernel_size=1, stride=1, padding=0))
		self.latlayer2 = wn(nn.Conv2d(nf*1, nf, kernel_size=1, stride=1, padding=0))

		self.pcd_align = PCD_Align(nf=nf, groups=deform_groups, wn=wn)
		#################################################################

		self.fusion = CrossNonLocal_Fusion(nf=nf, nframes=nframes, center=self.center, wn=wn)

		#### reconstruction
		self.recon_trunk = N.TransformerGroups(nf, nf, ng=num_groups, nb=num_blocks)
		
		#### upsampling
		self.upconv1 = wn(nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True))
		self.upconv2 = wn(nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True))
		self.pixel_shuffle = nn.PixelShuffle(2)
		self.HRconv = wn(nn.Conv2d(64, 64, 3, 1, 1, bias=True))
		self.conv_last = wn(nn.Conv2d(64, 3, 3, 1, 1, bias=True))

		#### skip #############
		self.skip_pixel_shuffle = nn.PixelShuffle(2)
		self.skipup1 = wn(nn.Conv2d(1, nf * 4, 3, 1, 1, bias=True))
		self.skipup2 = wn(nn.Conv2d(nf, 3 * 4, 3, 1, 1, bias=True))

		#### activation function
		self.lrelu = nn.ReLU(inplace=True)
		self.lrelu2 = nn.ReLU(inplace=True)

	def _upsample_add(self, x, y):
		return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) + y

	def forward(self, x):
		B, N, C, H, W = x.size()  # N video frames
		x_center = x[:, self.center, :, :, :].contiguous()

		#### skip module ########
		skip1 = self.lrelu2(self.skip_pixel_shuffle(self.skipup1(x_center)))
		skip2 = self.skip_pixel_shuffle(self.skipup2(skip1))

		#### extract LR features
		# L1
		L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
		L1_fea = self.feature_extraction(L1_fea)
		# L2
		L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
		# L3
		L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))

		# FPN enhance features
		L3_fea = self.lrelu(self.toplayer(L3_fea))
		L2_fea = self.smooth1(self._upsample_add(L3_fea, self.latlayer1(L2_fea)))
		L1_fea = self.smooth2(self._upsample_add(L2_fea, self.latlayer2(L1_fea)))

		L1_fea = L1_fea.view(B, N, -1, H, W)
		L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
		L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

		#### PCD align
		# ref feature list
		ref_fea_l = [
			L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
			L3_fea[:, self.center, :, :, :].clone()
		]
		aligned_fea = []
		for i in range(N):
			nbr_fea_l = [
				L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
				L3_fea[:, i, :, :, :].clone()
			]
			aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))
		aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W] --> [B, T, C, H, W]

		fea = self.lrelu(self.fusion(aligned_fea))

		out = self.recon_trunk(fea)
		out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
		out = skip1 + out
		out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
		out = self.lrelu(self.HRconv(out))
		out = self.conv_last(out)

		out = skip2 + out
		return out

def make_layer(block, n_layers):
	layers = []
	for _ in range(n_layers):
		layers.append(block())
	return nn.Sequential(*layers)

def make_layer_idx(block, n_layers):
	layers = []
	for i in range(n_layers):
		layers.append(block(idx=i))
	return nn.Sequential(*layers)

def make_model(args, parent=False):
	return EBSR(args)


class PCD_Align(nn.Module):
	''' Alignment module using Pyramid, Cascading and Deformable convolution
	with 3 pyramid levels. [From EDVR]
	'''

	def __init__(self, nf=64, groups=8, wn=None):
		super(PCD_Align, self).__init__()
		# L3: level 3, 1/4 spatial size
		self.L3_offset_conv1 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for diff
		self.L3_offset_conv2 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
		# self.L3_shift = ShiftAlign(nf)
		self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
							  # extra_offset_mask=True)
		# L2: level 2, 1/2 spatial size
		self.L2_offset_conv1 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for diff
		self.L2_offset_conv2 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for offset
		self.L2_offset_conv3 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
		# self.L2_shift = ShiftAlign(nf)
		self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
							  # extra_offset_mask=True)
		self.L2_fea_conv = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for fea
		# L1: level 1, original spatial size
		self.L1_offset_conv1 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for diff
		self.L1_offset_conv2 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for offset
		self.L1_offset_conv3 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
		# self.L1_shift = ShiftAlign(nf)
		self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
							  # extra_offset_mask=True)
		self.L1_fea_conv = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for fea
		# Cascading DCN
		self.cas_offset_conv1 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for diff
		self.cas_offset_conv2 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

		self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)

		self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

	def forward(self, nbr_fea_l, ref_fea_l):
		'''align other neighboring frames to the reference frame in the feature level
		nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
		'''
		# L3
		L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
		L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
		L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
		# L3_nbr_fea = self.L3_shift(L3_offset, nbr_fea_l[2])
		L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset))
		# L2
		L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
		L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
		L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
		L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset*2], dim=1)))
		L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
		# L2_nbr_fea = self.L2_shift(L2_offset, nbr_fea_l[1])
		L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset)
		L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
		L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
		# L1
		L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
		L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
		L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
		L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
		L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
		# L1_nbr_fea = self.L1_shift(L1_offset, nbr_fea_l[0])
		L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset)
		L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
		L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
		# Cascading
		offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
		offset = self.lrelu(self.cas_offset_conv1(offset))
		offset = self.lrelu(self.cas_offset_conv2(offset))
		L1_fea = self.lrelu(self.cas_dcnpack(L1_fea, offset))

		return L1_fea


class CrossNonLocal_Fusion(nn.Module):
	''' Cross Non Local fusion module
	'''
	def __init__(self, nf=64, nframes=5, center=2, wn=None):
		super(CrossNonLocal_Fusion, self).__init__()
		self.center = center

		self.non_local_T = nn.ModuleList()
		self.non_local_F = nn.ModuleList()

		for i in range(nframes):
			self.non_local_T.append(NonLocalCross(nf, inter_channels=nf//2, sub_sample=True, bn_layer=False))
			self.non_local_F.append(NonLocal(nf, inter_channels=nf//2, sub_sample=True, bn_layer=False))

		# fusion conv: using 1x1 to save parameters and computation
		self.fea_fusion = wn(nn.Conv2d(nframes * nf * 2, nf, 1, 1, bias=True))

		self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

	def forward(self, aligned_fea):
		B, N, C, H, W = aligned_fea.size()  # N video frames
		ref = aligned_fea[:, self.center, :, :, :].clone()

		cor_l = []
		non_l = []
		for i in range(N):
			nbr = aligned_fea[:, i, :, :, :]
			non_l.append(self.non_local_F[i](nbr))
			cor_l.append(self.non_local_T[i](nbr, ref))

		aligned_fea_T = torch.cat(cor_l, dim=1)
		aligned_fea_F = torch.cat(non_l, dim=1)
		aligned_fea = torch.cat([aligned_fea_T, aligned_fea_F], dim=1)

		#### fusion
		fea = self.fea_fusion(aligned_fea)

		return fea

class WideActResBlock(nn.Module):
	def __init__(self, nf=64):
		super(WideActResBlock, self).__init__()
		self.res_scale = 1
		body = []
		expand = 6
		linear = 0.8
		kernel_size = 3
		wn = lambda x: torch.nn.utils.weight_norm(x)
		act=nn.ReLU(True)

		body.append(wn(nn.Conv2d(nf, nf*expand, 1, padding=1//2)))
		body.append(act)
		body.append(wn(nn.Conv2d(nf*expand, int(nf*linear), 1, padding=1//2)))
		body.append(wn(nn.Conv2d(int(nf*linear), nf, kernel_size, padding=kernel_size//2)))

		self.body = nn.Sequential(*body)

	def forward(self, x):
		res = self.body(x) * self.res_scale
		res += x
		return res


class LRSCWideActResBlock(nn.Module):
	def __init__(self, nf=64, idx=0):
		super(LRSCWideActResBlock, self).__init__()
		self.res_scale = 1

		expand = 6
		linear = 0.8
		kernel_size = 3
		wn = lambda x: torch.nn.utils.weight_norm(x)
		act=nn.ReLU(True)
		head = [wn(nn.Conv2d(nf*(idx+1), nf, 1, bias=True))] if idx > 0 else []

		body = []
		body.append(wn(nn.Conv2d(nf, nf*expand, 1, padding=1//2)))
		body.append(act)
		body.append(wn(nn.Conv2d(nf*expand, int(nf*linear), 1, padding=1//2)))
		body.append(wn(nn.Conv2d(int(nf*linear), nf, kernel_size, padding=kernel_size//2)))

		self.head = nn.Sequential(*head)
		self.body = nn.Sequential(*body)

	def forward(self, x):
		res = self.head(x)
		res = self.body(res)
		res  = torch.cat([res, x], dim=1)
		return res

## Long-Range Skip-connect Residual Group (RG)
class LRSCWideActResGroup(nn.Module):
	def __init__(self, nf, n_resblocks, idx=0):
		super(LRSCWideActResGroup, self).__init__()
		kernel_size = 3

		# conv = PyConv2d
		wn = lambda x: torch.nn.utils.weight_norm(x)

		modules_head = [wn(nn.Conv2d(nf*(idx+1), nf, 1, 1, 0, bias=True))] if idx > 0 else []
		modules_body = [LRSCWideActResBlock(nf=nf, idx=i) for i in range(n_resblocks)]
		modules_tail = [wn(nn.Conv2d(nf*(n_resblocks+1), nf, 1))]
		self.head = nn.Sequential(*modules_head)
		self.body = nn.Sequential(*modules_body)
		self.tail = nn.Sequential(*modules_tail)

	def forward(self, x):
		res = self.head(x)
		res = self.body(res)
		res = self.tail(res)
		res  = torch.cat([res, x], dim=1)
		return res


class DCN(nn.Module):
	'''Use other features to generate offsets and masks'''

	def __init__(self,
				 in_channels,
				 out_channels,
				 kernel_size,
				 stride,
				 padding,
				 dilation=1,
				 deformable_groups=1):
		super(DCN, self).__init__()

		channels_ = deformable_groups * 2 * kernel_size * kernel_size
		self.conv_offset_mask = nn.Conv2d(
			in_channels,
			channels_,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
			bias=True)
		self.deform_conv = ops.DeformConv2d(in_channels, out_channels, 3, 1, 1, groups=deformable_groups)
		self.init_offset()

	def init_offset(self):
		self.conv_offset_mask.weight.data.zero_()
		self.conv_offset_mask.bias.data.zero_()

	def forward(self, input, fea):
		'''input: input features for deformable conv
		fea: other features used for generating offsets and mask'''
		out = self.conv_offset_mask(fea)
		offset = torch.clamp(out, -200, 200)

		offset_mean = torch.mean(torch.abs(offset))
		if offset_mean > 150:
			print('Offset mean is {}, larger than 100.'.format(offset_mean))
			offset[offset>=150] = 1e-3

		return self.deform_conv(input, offset)

