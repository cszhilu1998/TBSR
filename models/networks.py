import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from collections import OrderedDict
import torch.nn.functional as F
from einops import rearrange
import torchvision.ops as ops
import numbers
import numpy as np


def get_scheduler(optimizer, opt):
	if opt.lr_policy == 'linear':
		def lambda_rule(epoch):
			return 1 - max(0, epoch-opt.niter) / max(1, float(opt.niter_decay))
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer,
										step_size=opt.lr_decay_iters,
										gamma=0.5)
	elif opt.lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
												   mode='min',
												   factor=0.2,
												   threshold=0.01,
												   patience=5)
	elif opt.lr_policy == 'cosine':
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
												   T_max=opt.niter,
												   eta_min=0)
	else:
		return NotImplementedError('lr [%s] is not implemented', opt.lr_policy)
	return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
	def init_func(m):  # define the initialization function
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 \
				or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=init_gain)
			elif init_type == 'uniform':
				init.uniform_(m.weight.data, b=init_gain)
			else:
				raise NotImplementedError('[%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, init_gain)
			init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='default', init_gain=0.02, gpu_ids=[]):
	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
	if init_type != 'default' and init_type is not None:
		init_weights(net, init_type, init_gain=init_gain)
	return net


'''
# ===================================
# Advanced nn.Sequential
# reform nn.Sequentials and nn.Modules
# to a single nn.Sequential
# ===================================
'''

def seq(*args):
	if len(args) == 1:
		args = args[0]
	if isinstance(args, nn.Module):
		return args
	modules = OrderedDict()
	if isinstance(args, OrderedDict):
		for k, v in args.items():
			modules[k] = seq(v)
		return nn.Sequential(modules)
	assert isinstance(args, (list, tuple))
	return nn.Sequential(*[seq(i) for i in args])

'''
# ===================================
# Useful blocks
# --------------------------------
# conv (+ normaliation + relu)
# concat
# sum
# resblock (ResBlock)
# resdenseblock (ResidualDenseBlock_5C)
# resinresdenseblock (RRDB)
# ===================================
'''

wn = lambda x: torch.nn.utils.weight_norm(x)


# -------------------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# -------------------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
		 output_padding=0, dilation=1, groups=1, bias=True,
		 padding_mode='zeros', mode='CBR'):
	L = []
	for t in mode:
		if t == 'C':
			L.append(wn(nn.Conv2d(in_channels=in_channels,
							   out_channels=out_channels,
							   kernel_size=kernel_size,
							   stride=stride,
							   padding=padding,
							   dilation=dilation,
							   groups=groups,
							   bias=bias,
							   padding_mode=padding_mode)))
		elif t == 'X':
			assert in_channels == out_channels
			L.append(nn.Conv2d(in_channels=in_channels,
							   out_channels=out_channels,
							   kernel_size=kernel_size,
							   stride=stride,
							   padding=padding,
							   dilation=dilation,
							   groups=in_channels,
							   bias=bias,
							   padding_mode=padding_mode))
		elif t == 'T':
			L.append(nn.ConvTranspose2d(in_channels=in_channels,
										out_channels=out_channels,
										kernel_size=kernel_size,
										stride=stride,
										padding=padding,
										output_padding=output_padding,
										groups=groups,
										bias=bias,
										dilation=dilation,
										padding_mode=padding_mode))
		elif t == 'B':
			L.append(nn.BatchNorm2d(out_channels))
		elif t == 'I':
			L.append(nn.InstanceNorm2d(out_channels, affine=True))
		elif t == 'i':
			L.append(nn.InstanceNorm2d(out_channels))
		elif t == 'R':
			L.append(nn.ReLU(inplace=True))
		elif t == 'r':
			L.append(nn.ReLU(inplace=False))
		elif t == 'S':
			L.append(nn.Sigmoid())
		elif t == 'P':
			L.append(nn.PReLU())
		elif t == 'L':
			L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=True))
		elif t == 'l':
			L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=False))
		elif t == '2':
			L.append(nn.PixelShuffle(upscale_factor=2))
		elif t == '3':
			L.append(nn.PixelShuffle(upscale_factor=3))
		elif t == '4':
			L.append(nn.PixelShuffle(upscale_factor=4))
		elif t == 'U':
			L.append(nn.Upsample(scale_factor=2, mode='nearest'))
		elif t == 'u':
			L.append(nn.Upsample(scale_factor=3, mode='nearest'))
		elif t == 'M':
			L.append(nn.MaxPool2d(kernel_size=kernel_size,
								  stride=stride,
								  padding=0))
		elif t == 'A':
			L.append(nn.AvgPool2d(kernel_size=kernel_size,
								  stride=stride,
								  padding=0))
		else:
			raise NotImplementedError('Undefined type: '.format(t))
	return seq(*L)


##########################################################################
## Layer Norm

def to_3d(x):
	return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
	return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
	def __init__(self, normalized_shape):
		super(BiasFree_LayerNorm, self).__init__()
		if isinstance(normalized_shape, numbers.Integral):
			normalized_shape = (normalized_shape,)
		normalized_shape = torch.Size(normalized_shape)

		assert len(normalized_shape) == 1

		self.weight = nn.Parameter(torch.ones(normalized_shape))
		self.normalized_shape = normalized_shape

	def forward(self, x):
		sigma = x.var(-1, keepdim=True, unbiased=False)
		return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
	def __init__(self, normalized_shape):
		super(WithBias_LayerNorm, self).__init__()
		if isinstance(normalized_shape, numbers.Integral):
			normalized_shape = (normalized_shape,)
		normalized_shape = torch.Size(normalized_shape)

		assert len(normalized_shape) == 1

		self.weight = nn.Parameter(torch.ones(normalized_shape))
		self.bias = nn.Parameter(torch.zeros(normalized_shape))
		self.normalized_shape = normalized_shape

	def forward(self, x):
		mu = x.mean(-1, keepdim=True)
		sigma = x.var(-1, keepdim=True, unbiased=False)
		return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
	def __init__(self, dim, LayerNorm_type):
		super(LayerNorm, self).__init__()
		if LayerNorm_type =='BiasFree':
			self.body = BiasFree_LayerNorm(dim)
		else:
			self.body = WithBias_LayerNorm(dim)

	def forward(self, x):
		h, w = x.shape[-2:]
		return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
	def __init__(self, dim, ffn_expansion_factor, bias):
		super(FeedForward, self).__init__()

		hidden_features = int(dim*ffn_expansion_factor)

		self.project_in = wn(nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias))

		self.dwconv = wn(nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias))

		self.project_out = wn(nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias))

	def forward(self, x):
		x = self.project_in(x)
		x1, x2 = self.dwconv(x).chunk(2, dim=1)
		x = F.gelu(x1) * x2
		x = self.project_out(x)
		return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
	def __init__(self, dim, num_heads, bias):
		super(Attention, self).__init__()
		self.num_heads = dim // 32
		self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

		self.qkv = wn(nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias))
		self.qkv_dwconv = wn(nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias))
		self.project_out = wn(nn.Conv2d(dim, dim, kernel_size=1, bias=bias))
		
	def forward(self, x):
		b,c,h,w = x.shape

		qkv = self.qkv_dwconv(self.qkv(x))
		q,k,v = qkv.chunk(3, dim=1)   
		
		q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
		k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
		v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

		q = torch.nn.functional.normalize(q, dim=-1)
		k = torch.nn.functional.normalize(k, dim=-1)

		attn = (q @ k.transpose(-2, -1)) * self.temperature
		attn = attn.softmax(dim=-1)

		out = (attn @ v)
		
		out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

		out = self.project_out(out)
		return out


##########################################################################
class TransformerBlock(nn.Module):
	def __init__(self, dim=128, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
		super(TransformerBlock, self).__init__()

		self.norm1 = LayerNorm(dim, LayerNorm_type)
		self.attn = Attention(dim, num_heads, bias)
		self.norm2 = LayerNorm(dim, LayerNorm_type)
		self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

	def forward(self, x):
		x = x + self.attn(self.norm1(x))
		x = x + self.ffn(self.norm2(x))

		return x


class TransformerBlocks(nn.Module):
	def __init__(self, in_channels, out_channels=128, nb=8):
		super().__init__()

		head = []
		head.append(wn(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)))
		self.head = nn.Sequential(*head)

		main = [TransformerBlock(dim=out_channels) for _ in range(nb)]
		self.main = nn.Sequential(*main)

	def forward(self, feat):
		h = F.gelu(self.head(feat))
		return h + self.main(h)


class TransformerGroups(nn.Module):
	def __init__(self, in_channels, out_channels=128, ng=8, nb=8):
		super().__init__()

		head = []
		head.append(wn(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)))
		self.head = nn.Sequential(*head)

		main = [TransformerBlocks(out_channels, out_channels, nb) for _ in range(ng)]
		self.main = nn.Sequential(*main)

	def forward(self, feat):
		h = self.head(feat)
		return h + self.main(h)

