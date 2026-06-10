from torch.optim.lr_scheduler import LambdaLR
import math
def get_warmup_cosine_scheduler(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    num_cycles=0.5,
    min_lr=0.0,
    last_epoch=-1
):
    """
    创建一个学习率调度器，先线性预热，然后进行余弦退火
    
    参数:
        optimizer: 优化器
        num_warmup_steps: 预热阶段的步数
        num_training_steps: 总训练步数
        num_cycles: 余弦函数的波数 (0.5表示半个余弦波)
        min_lr: 最低学习率 (相对于初始lr的比例)
        last_epoch: 上一轮的索引(用于继续训练)
    """
    def lr_lambda(current_step):
        # 预热阶段: 线性增长到基础学习率
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 预热后: 余弦退火
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        
        # 将学习率从1降到min_lr
        return min_lr + (1.0 - min_lr) * cosine_decay

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_freeze_warmup_scheduler(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    num_cycles=0.5,
    min_lr_ratio=0.0, 
    last_epoch=-1
):
    """
    专门针对您的多参数组优化器定制的调度器。
    
    逻辑:
    1. 索引为 0 的参数组 (mask_encoder): 在 num_warmup_steps 期间 LR=0 (冻结)，之后余弦退火。
    2. 索引 > 0 的参数组 (mask_token, ca_net 等): 在 num_warmup_steps 期间线性预热，之后余弦退火。
    """
    
    # --- 通用的余弦退火计算逻辑 (预热结束后使用) ---
    def _get_cosine_schedule(current_step):
        # 计算预热后的进度 [0.0, 1.0]
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # 计算余弦衰减因子
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        # 缩放到 [min_lr_ratio, 1.0] 之间
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    # --- 策略 A: 冻结策略 (用于 mask_encoder) ---
    def lr_lambda_freeze(current_step):
        if current_step < num_warmup_steps:
            return 0.0  # <--- 关键：预热期乘数为 0，实现冻结
        return _get_cosine_schedule(current_step)

    # --- 策略 B: 正常预热策略 (用于其他层) ---
    def lr_lambda_normal(current_step):
        if current_step < num_warmup_steps:
            # 线性预热: 0 -> 1
            return float(current_step) / float(max(1, num_warmup_steps))
        return _get_cosine_schedule(current_step)

    # --- 动态构建 lambda 列表 ---
    # 根据您提供的 optimizer，这里有 4 个 group。
    # 我们遍历所有 group，如果是第 0 个就用冻结策略，否则用正常策略。
    lambda_list = []
    for i in range(len(optimizer.param_groups)):
        if i == 0:
            # Group 0: mask_encoder -> 冻结
            lambda_list.append(lr_lambda_freeze)
        else:
            # Group 1, 2, 3...: 其他部分 -> 正常预热
            lambda_list.append(lr_lambda_normal)

    return LambdaLR(optimizer, lr_lambda=lambda_list, last_epoch=last_epoch)