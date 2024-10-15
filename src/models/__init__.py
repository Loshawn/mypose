from .ViTPose import ViTPose
from .losses import build_loss_and_optimizer
import torch


def build_model(args):
    if args.model == "ViTPose":
        return ViTPose(args.model_cfg)


def save_model(model, path):
    if isinstance(model, torch.nn.DataParallel) or isinstance(
            model, torch.nn.parallel.DistributedDataParallel):
        torch.save(model.module.state_dict(), path)  # 多卡模式下保存 module
    else:
        torch.save(model.state_dict(), path)


def load_model(model, path, device):
    state_dict = torch.load(
        path,
        map_location=device,
        weights_only=True)

    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    return model


def save_optimizer_and_logger(optimizer, logger, path):
    checkpoint = {
        'optimizer': optimizer.state_dict(),  # 保存优化器状态
        'info': logger.info  # 保存 logger 的 info 信息
    }

    torch.save(checkpoint, path)


def load_optimizer_and_logger(optimizer, path, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True)

    optimizer.load_state_dict(checkpoint['optimizer'])
    info = checkpoint['info']

    return optimizer, info
