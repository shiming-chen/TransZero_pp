import torch
import torch.optim as optim
import torch.nn as nn
from model_tzpp import TransZeroPP
from dataset import CUBDataLoader
from helper_func import eval_zs_gzsl
import numpy as np
import wandb

# init wandb from config file
wandb.init(project='TransZeroPP', config='wandb_config/cub_gzsl.yaml')
config = wandb.config
print('Config file from wandb:', config)

# load dataset
dataloader = CUBDataLoader('.', config.device, is_balance=False)

# set random seed
seed = config.random_seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# TransZero model
model = TransZeroPP(config, dataloader.att, dataloader.w2v_att,
                  dataloader.seenclasses, dataloader.unseenclasses).to(config.device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

# main loop
niters = dataloader.ntrain * config.epochs//config.batch_size
report_interval = niters//config.epochs
best_performance = [0, 0, 0, 0]
best_performance_zsl = 0
for i in range(0, niters):
    model.train()
    optimizer.zero_grad()

    batch_label, batch_feature, batch_att = dataloader.next_batch(
        config.batch_size)
    out_package = model(batch_feature)

    in_package1 = out_package['package_s2v']
    in_package2 = out_package['package_v2s']
    in_package1['batch_label'] = batch_label
    in_package2['batch_label'] = batch_label
    
    out_package1=model.compute_loss(in_package1)
    out_package2=model.compute_loss(in_package2)

    loss = out_package1['loss'] + config.lambda_v2s * out_package2['loss']
    loss_CE = out_package1['loss_CE'] + out_package2['loss_CE']
    loss_cal = out_package1['loss_cal'] + out_package2['loss_cal']
    loss_reg = out_package1['loss_reg'] + out_package2['loss_reg']

    loss_att, loss_cls = model.compute_contrastive_loss(
            in_package1, in_package2)
    loss += config.lambda_cst_reg_att * loss_att
    loss += config.lambda_cst_reg_cls * loss_cls
    
    loss.backward()
    optimizer.step()

    # report result
    if i % report_interval == 0:
        print('-'*30)
        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
            dataloader, model, config.device, batch_size=config.batch_size)

        if H > best_performance[2]:
            best_performance = [acc_novel, acc_seen, H, acc_zs]
        if acc_zs > best_performance_zsl:
            best_performance_zsl = acc_zs

        print('iter/epoch=%d/%d | loss=%.3f, loss_CE=%.3f, loss_cal=%.3f, '
              'loss_reg=%.3f, loss_cst_att=%.3f, loss_cst_cls=%.3f | ' % (
                  i, int(i//report_interval),
                  loss.item(), loss_CE.item(), loss_cal.item(),
                  loss_reg.item(), loss_att.item(), loss_cls.item()))
        print('Current GZSL: acc_unseen=%.3f, acc_seen=%.3f, H=%.3f | Current CZSL: acc_zs=%.3f' % (
                  acc_novel, acc_seen, H, acc_zs))
        print('BEST GZSL: acc_unseen=%.3f, acc_seen=%.3f, H=%.3f, acc_zs=%.3f'
              ' | BEST CZSL: acc_zs=%.3f' % (
                  best_performance[0], best_performance[1],
                  best_performance[2], best_performance[3],
                  best_performance_zsl))

        wandb.log({
            'iter': i,
            'loss': loss.item(),
            'loss_CE': loss_CE.item(),
            'loss_cal': loss_cal.item(),
            'loss_reg': loss_reg.item(),
            'loss_att': loss_att.item(),
            'loss_cls': loss_cls.item(), 
            'acc_unseen': acc_novel,
            'acc_seen': acc_seen,
            'H': H,
            'acc_zs': acc_zs,
            'best_acc_unseen': best_performance[0],
            'best_acc_seen': best_performance[1],
            'best_H': best_performance[2],
            'best_acc_zs': best_performance_zsl
        })
