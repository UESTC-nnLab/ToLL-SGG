#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import types

import torch


if __name__ == '__main__':
    project_root = os.path.dirname(__file__)
    src_diff_root = os.path.join(project_root, 'src_diff')
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [src_diff_root]
    sys.modules["src"] = src_pkg

from src.model.model import Pdiff4SSG_Pretraining
from src.model.ddptrain import Pdiff4SSG_Pretraining_ddp


class Config:
    def __init__(self, data_dict: dict):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default=os.path.join(os.path.dirname(__file__), 'configs', 'tollpp_scannet.json')
    )
    parser.add_argument('--no_ddp', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--eval_ckpt', type=str, default='')
    parser.add_argument('--eval_epoch', type=int, default=0)
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件 '{args.config}' 未找到。")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误: 配置文件 '{args.config}' 不是有效的JSON格式。")
        sys.exit(1)

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
            sys.exit(1)
        if not os.path.exists(ckpt_path):
            print(f"错误: checkpoint 不存在: {ckpt_path}")
            sys.exit(1)

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        try:
            if hasattr(pretrainer, 'raw_model'):
                pretrainer.raw_model.load_state_dict(state_dict, strict=False)
            else:
                pretrainer.model.load_state_dict(state_dict, strict=False)
        except Exception:
            normalized_state_dict = {
                (k[7:] if k.startswith('module.') else k): v
                for k, v in state_dict.items()
            }
            if hasattr(pretrainer, 'raw_model'):
                pretrainer.raw_model.load_state_dict(normalized_state_dict, strict=False)
            else:
                pretrainer.model.load_state_dict(normalized_state_dict, strict=False)

        if is_ddp and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                pretrainer.validation_for_cls(epoch=args.eval_epoch)
            torch.distributed.barrier()
        else:
            pretrainer.validation_for_cls(epoch=args.eval_epoch)
        sys.exit(0)

    if is_ddp:
        pretrainer = Pdiff4SSG_Pretraining_ddp(config)
    else:
        pretrainer = Pdiff4SSG_Pretraining(config)

    pretrainer.train()
