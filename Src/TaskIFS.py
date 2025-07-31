"""
created by Allan Yu, 14-10-03
"""

from copy import deepcopy
from LocalOnly.PathConfig import path_config


ifs_config = {
    "device": 'cuda:0',
    "epochs": 20,
    "chkpt_milestones": [10, 15, 19],
    "sch_epoch": [10, 15, 19],
    "tr_batch_size": 8,
    "batchs_in_step": 4,
    "num_workers": 16,
    "lr": 1e-4,
    "lr_weight_decay": 0,
    "sch_gamma": 0.1,
    "apply_weight_init": True,
    "seed": 17,
    "topic": '0721_64_12_avg'
}


def create_ifs_dataset(tm, is_train=False):
    from Dataset.PathedIFS import get_dataloader
    from Model.IFSNet import IFSNet

    return get_dataloader(path=path_config["path_of_ifs"], is_train=is_train, roi=1024)


def create_model_ifs(tm):
    from Dataset.PathedIFS import ifs_pack
    from Model.IFSNet import IFSNet

    return IFSNet(tm, pack=ifs_pack, rov=64, fusion='wheel', **tm.args)


def task():
    config = deepcopy(ifs_config)
    config.update({"abstract": "0721_48_16_wheel_1536img", "best_performance": 0.01})
    return {
        "name": "ifs",
        "create_dataloader": create_ifs_dataset,
        "create_model": create_model_ifs,
        "args": config,
    }
