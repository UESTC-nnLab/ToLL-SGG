#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from genericpath import isfile
import json
import os
if __name__ == '__main__':
    os.sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.model.model import Pdiff4SSG_Pretraining
from src.model.ddptrain import Pdiff4SSG_Pretraining_ddp 
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default=os.path.join(os.path.dirname(__file__), 'configs', 'mmgnet.json')
    )
    parser.add_argument('--no_ddp', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--eval_ckpt', type=str, default='')
    parser.add_argument('--eval_epoch', type=int, default=0)
    args = parser.parse_args()
    config_file_path = args.config
    
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
    is_ddp = not args.no_ddp

    if args.eval_only:
        if is_ddp:
            pretrainer = Pdiff4SSG_Pretraining_ddp(config, val_cls_mode=True)
        else:
            pretrainer = Pdiff4SSG_Pretraining(config, val_cls_mode=True)

        ckpt_path = args.eval_ckpt if args.eval_ckpt else getattr(config, 'RESUME_PATH', '')
        if not ckpt_path:
            print("错误: 离线验证需要提供 checkpoint 路径。请使用 --eval_ckpt 或在 config 里设置 RESUME_PATH")
            exit(1)
        if not os.path.exists(ckpt_path):
            print(f"错误: checkpoint 不存在: {ckpt_path}")
            exit(1)

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        try:
            pretrainer.model.load_state_dict(state_dict)
        except Exception:
            if hasattr(pretrainer, 'raw_model'):
                pretrainer.raw_model.load_state_dict(state_dict)
            else:
                raise
        setattr(pretrainer, '_ckpt_loaded', True)
        setattr(pretrainer, '_ckpt_path', ckpt_path)

        if is_ddp and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                pretrainer.validation_for_cls(epoch=args.eval_epoch)
            torch.distributed.barrier()
        else:
            pretrainer.validation_for_cls(epoch=args.eval_epoch)
        exit(0)

    if is_ddp:
        pretrainer = Pdiff4SSG_Pretraining_ddp(config)
    else:
        pretrainer = Pdiff4SSG_Pretraining(config)
    
    pretrainer.train()
