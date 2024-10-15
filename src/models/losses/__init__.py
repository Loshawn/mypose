from . import mse_loss
import torch.optim as optim


def build_loss_and_optimizer(args, model):
    loss = mse_loss.JointsMSELoss(use_target_weight=args.model_cfg['keypoint_head']
                                  ['loss_keypoint']['use_target_weight'])
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.optimizer['lr'],
                            betas=args.optimizer['betas'],
                            weight_decay=args.optimizer['weight_decay'])

    return loss, optimizer
