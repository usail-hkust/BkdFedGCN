import logging
import os
import numpy as np
import torch
import random
import datetime
import subprocess
import csv
from mmcv.runner import get_dist_info
import torch.nn as nn
np.seterr(divide='ignore', invalid='ignore')

import torch.distributed as dist
import torch.multiprocessing as mp
from mmcv.runner import get_dist_info
def get_root_logger(log_level=logging.INFO, log_dir='./'):
    ISOTIMEFORMAT = '%Y.%m.%d-%H.%M.%S'
    thetime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    logname = os.path.join(log_dir, thetime + '.log')
    logger = logging.getLogger()

    if not logger.hasHandlers():
        fmt ='%(asctime)s - %(levelname)s - %(message)s'
        format_str = logging.Formatter(fmt)
        logging.basicConfig(filename=logname, filemode='a', format=fmt, level=log_level)
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        logger.addHandler(sh)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    else:
        logger.setLevel(log_level)
    return logger

def logger_info(logger, dist, info):
    # to only write on rank0
    if not dist:
        logger.info(info)
    else:
        local_rank = torch.distributed.get_rank()
        if local_rank == 0:
            logger.info(info)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_dist(launcher, backend='nccl', **kwargs):
    # if mp.get_start_method(allow_none=True) is None:
    #     mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))

def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def _init_dist_mpi(backend, **kwargs):
    raise NotImplementedError

def _init_dist_slurm(backend, port=29500, **kwargs):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        'scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

def set_default_configs(cfgs, display=False):
    "set default configs for new coming features"
    default_dict = dict(
        resume_epoch=0,
        load_from=None,
        log_level='INFO',
        white_box_attack=None,
        source_model_path=None,
        target_model_path=None,
        lr_dict=None,
        method='',
        load_modules=(),
        remark='',
        cpu_data=False,
        eval_freq=5,
        other_params=dict(),
        load_model=None,
        deffer_opt=None,
        warmup_epochs=0,
        epochs=cfgs.epochs,
        backbone=cfgs.backbone,
        batch_size=cfgs.batch_size,
        dataset=cfgs.dataset,
    )
