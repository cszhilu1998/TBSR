import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
from util.util import calc_psnr as calc_psnr
import time
import numpy as np
from collections import OrderedDict as odict
from copy import deepcopy
import cv2


if __name__ == '__main__':
    opt = TestOptions().parse()

    if not isinstance(opt.load_iter, list):
        load_iters = [opt.load_iter]
    else:
        load_iters = deepcopy(opt.load_iter)

    if not isinstance(opt.dataset_name, list):
        dataset_names = [opt.dataset_name]
    else:
        dataset_names = deepcopy(opt.dataset_name)
    datasets = odict()
    for dataset_name in dataset_names:
        dataset = create_dataset(dataset_name, 'test', opt)
        datasets[dataset_name] = tqdm(dataset)

    for load_iter in load_iters:
        opt.load_iter = load_iter
        model = create_model(opt)
        model.setup(opt)
        model.eval()
        # log_dir = '%s/%s/logs/log_epoch_%d.txt' % (
        #         opt.checkpoints_dir, opt.name, load_iter)
        # os.makedirs(os.path.split(log_dir)[0], exist_ok=True)
        # f = open(log_dir, 'a')

        for dataset_name in dataset_names:
            opt.dataset_name = dataset_name
            tqdm_val = datasets[dataset_name]
            dataset_test = tqdm_val.iterable
            dataset_size_test = len(dataset_test)

            print('='*80)
            print(dataset_name + ' dataset')
            tqdm_val.reset()

            psnr = [0.0] * dataset_size_test

            time_val = 0
            for i, data in enumerate(tqdm_val):
                torch.cuda.empty_cache()
                model.set_input(data)
                torch.cuda.synchronize()
                time_val_start = time.time()
                model.test()
                torch.cuda.synchronize()
                time_val += time.time() - time_val_start
                res = model.get_current_visuals()

                # if opt.calc_metrics:
                #     psnr[i] = calc_psnr(res['data_gt'], res['data_out'])
                
                if opt.save_imgs:
                    file_name = data['fname'][0].split('-')
                    folder_dir = './ckpt/%s/output' % (opt.name)  
                    os.makedirs(folder_dir, exist_ok=True)

                    net_pred_np = (res['data_out'].squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(np.uint16)
                    cv2.imwrite('{}/{}.png'.format(folder_dir, file_name[0]), net_pred_np)

            avg_psnr = '%.2f'%np.mean(psnr)

            print('Time: %.3f s AVG Time: %.3f ms PSNR: %s\n' % (time_val, time_val/dataset_size_test*1000, avg_psnr))

    for dataset in datasets:
        datasets[dataset].close()
