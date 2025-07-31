"""
created by Allan Yu, 14-10-03
"""

from copy import deepcopy
from LocalOnly.PathConfig import path_config


ifs_config = {
    "device": 'cuda:0',
    "epochs": 200,
    "chkpt_milestones": [50, 100, 199],
    "sch_epoch": [50, 100, 199],
    "tr_batch_size": 1,
    "batchs_in_step": 16,
    "num_workers": 16,
    "lr": 1e-3,
    "lr_weight_decay": 0,
    "sch_gamma": 0.1,
    "apply_weight_init": True,
    "seed": 17,
    "topic": '0714_1_ifs, 0714_ifs 失败后后继。encoder更改为原分辨率'
}




def create_pifu_dataset(tm, is_train=False):
    from Dataset.PifuIFS import get_dataloader
    from Model.IFSNet import IFSNet

    return get_dataloader(path=path_config["path_of_ifs"], is_train=is_train, roi=768)



def create_model_pifu(tm):
    from Dataset.PathedIFS import ifs_pack
    from Model.Pifu import IFSNet

    return IFSNet(tm, pack=ifs_pack, **tm.args)


def task():
    config = deepcopy(ifs_config)
    config.update({"abstract": "ifs refactoring version", "best_performance": 0.01})
    return {
        "name": "ifs",
        "create_dataloader": create_pifu_dataset,
        "create_model": create_model_pifu,
        "args": config,
    }
