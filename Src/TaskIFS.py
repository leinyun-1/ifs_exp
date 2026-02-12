"""
created by Allan Yu, 14-10-03
"""

from copy import deepcopy
from LocalOnly.PathConfig import path_config


ifs_config = {
    "device": 'cuda:1',
    "epochs": 20,  #20
    "reload": False,
    "chkpt": None,
    "chkpt_milestones": [3, 5, 9, 14, 19], #[10,15,19]
    "sch_epoch": [3, 5, 9, 14, 19], #[10,15,19]
    "tr_batch_size": 8, #3
    "batchs_in_step": 6, #8
    "num_workers": 8,
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

    return get_dataloader(path=path_config["path_of_ifs"], is_train=is_train, roi=1024, yaw_list=[0,2,4,6,8,10,12,14], dataset_type='A+')


def create_model_ifs(tm):
    from Dataset.PathedIFS import ifs_pack
    from Model.IFSNet import IFSNet

    return IFSNet(tm, pack=ifs_pack, rov=48, fusion='wheel', **tm.args)


def task():
    config = deepcopy(ifs_config)
    config.update({"abstract": "0211_42_16_wheel_1024img", "best_performance": 0.01})
    return {
        "name": "ifs",
        "create_dataloader": create_ifs_dataset,
        "create_model": create_model_ifs,
        "args": config,
    }
