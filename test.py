import torch
from eval_func import eval_zs_gzsl
from model import TransZero
from dataset import UNIDataloader
import argparse
import json
import wandb


def run_test(config):
    # dataset
    dataloader = UNIDataloader(config)
    # model
    model = TransZero(config)
    # load parameters
    model_dict = model.state_dict()
    saved_dict = torch.load(config.saved_model)
    saved_dict = {k: v for k, v in saved_dict.items() if k in model_dict}
    model_dict.update(saved_dict)
    model.load_state_dict(model_dict)
    model.to(config.device)
    # evaluation
    if config.zsl_task == 'CZSL':
        acc_zs = eval_zs_gzsl(config, dataloader, model)
        print('Acc_ZSL={:.3f}'.format(acc_zs))
    elif config.zsl_task == 'GZSL':
        acc_seen, acc_novel, H = eval_zs_gzsl(config, dataloader, model)
        print('Acc_Unseen={:.3f}, Acc_Seen={:.3f}, H={:.3f}'.format(
            acc_novel, acc_seen, H))


if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='config/test_CUB.json')
    # config = parser.parse_args()
    # with open(config.config, 'r') as f:
    #     config.__dict__ = json.load(f)
    # run_test(config)

    # wandb.init(project='TransZero', config='config/config_cub_czsl.yaml')
    # wandb.init(project='TransZero', config='config/config_cub_gzsl.yaml')
    # wandb.init(project='TransZero', config='config/config_sun_czsl.yaml')
    wandb.init(project='TransZero', config='config/config_sun_gzsl.yaml')
    # wandb.init(project='TransZero', config='config/config_awa2_gzsl.yaml')
    # wandb.init(project='TransZero', config='config/config_awa2_czsl.yaml')
    config = wandb.config
    print('Config file from wandb:', config)

    run_test(config)
