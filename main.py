#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from genericpath import isfile
import json
import os
if __name__ == '__main__':
    os.sys.path.append('./src')
from src.model.model import Pdiff4SSG_Pretraining
from src.model.ddptrain import Pdiff4SSG_Pretraining_ddp 
from utils import util
import torch
import argparse

class Config:
    def __init__(self, data_dict: dict):
        for key, value in data_dict.items():
            # 如果值是字典，则递归地创建一个新的Config实例
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            # 否则，直接将值设置为属性
            else:
                setattr(self, key, value)
    
    def __repr__(self, indent=0):
        lines = []
        indent_str = "  " * indent
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                lines.append(f"{indent_str}{key}:")
                lines.append(value.__repr__(indent + 1))
            else:
                lines.append(f"{indent_str}{key}: {repr(value)}")
        return "\n".join(lines)

if __name__ == '__main__':

    import json
    import os
    config_file_path = '/home/hyc/hyc_work/sceneGraph/SGG_DIR/config/Diff_pretrain.json'
    
    # b. 从JSON文件读取内容并加载到Python字典
    try:
        with open(config_file_path, 'r') as f:
            # 使用 json.load() 从文件对象中读取
            config_dict = json.load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件 '{config_file_path}' 未找到。")
        exit()
    except json.JSONDecodeError:
        print(f"错误: 配置文件 '{config_file_path}' 不是有效的JSON格式。")
        exit()
    config = Config(config_dict)
    is_ddp = True
    if is_ddp:
        pretrainer = Pdiff4SSG_Pretraining_ddp(config)
    else:
        pretrainer = Pdiff4SSG_Pretraining(config, val_cls_mode=True)
    
    pretrainer.train()
    
    # epoch_list =[90, 100, 110, 120, 130, 140, 150]# [10, 20, 30, 40, 50, 60, 70, 80]#, 90, 100, 110, 120, 130, 
    # for epoch in epoch_list:
    #     pretrainer.validation_for_cls(epoch=epoch)