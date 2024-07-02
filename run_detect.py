import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pdb import set_trace as deg
from simclr import SimCLR
from simclr.modules import LogisticRegression
from simclr.modules.transformations import TransformsSimCLR

from utils import yaml_config_hook
from dataset import *
# from simclr.modules import SingleViT2 as VViT
from simclr.modules.STViT_rearrange import SingleViT2 as VViT
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from datetime import datetime
from model import *
from sklearn import metrics

import pandas as pd
import tqdm
import pickle

torch.multiprocessing.set_sharing_strategy('file_system')

def detection( args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("#"*80)
    print(f"Start training model on {args.train_file}")
    print(f"Testing on {args.test_file}")
    print("#"*80)

    test_dataset = dataset_DFD(args,mode = 'test') 

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        # batch_size=args.batch_size if args.test_aug!=1 else 1,
        batch_size=args.batch_size if args.test_aug!=1 else 1,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers if args.test_aug!=1 else 1,
    )

    model = VViT(args.device, args)
    

    # optimizer / loss
    # optimizer, scheduler = load_optimizer(args, model)
    # criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, args.pos_weight]).to(args.device))

    # DDP / DP
    #     if args.dataparallel:
    #         model = convert_model(model)
    #         model = DataParallel(model)
    #     else:
    #         if args.nodes > 1:
    #             model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #             model = DDP(model, device_ids=[rank], output_device=rank)

    # use last
    model_fp = os.path.join(args.model_path, "last_c23_STFE_FE_MA.pth")
    model.load_state_dict(torch.load(model_fp, map_location='cpu'), strict=False)
    
    # use checkpoint_{}
    # model_fp = os.path.join(
    #     args.model_path, "checkpoint_{}.pth".format(args.epoch_num)
    # )
    # checkpoint = torch.load(model_fp, map_location=args.device.type)
    # new_ckpt = {}
    # for k,v in checkpoint['model_state_dict'].items():
    #     new_ckpt[k.replace('module.', '')] = v
    # model.load_state_dict(new_ckpt)
    model = model.to(args.device)

    model.eval()
    # checkpoint = torch.load(model_fp, map_location=args.device.type)
    # new_ckpt = {}
    # for k,v in checkpoint['model_state_dict'].items():
    #     new_ckpt[k.replace('module.', '')] = v
    # model.load_state_dict(new_ckpt)

    #         model.load_state_dict()
    args.current_epoch = args.epoch_num

    args.global_step = 0
    preds = []
    labs = []
    results = []
    probability = []
    for x,y in tqdm.tqdm(test_loader,total=len(test_loader)):
        y=y.to(args.device)
        pred = model(x, forward_mode='test')

        post_function=nn.Softmax(dim=1)
        prob = post_function(pred)
        predicted = torch.argmax(pred,1)
        probability.extend(prob[:, 1].cpu().detach().numpy())
        preds.extend(predicted.cpu().detach().numpy())
        labs.extend(y.cpu().detach().numpy())

    probability = np.array(probability)
    preds = np.array(preds)
    # print(preds)
    labs = np.array(labs)
    # print(labs)
    
# oppisite result
#     for kk in range(len(preds)):
#         if preds[kk]==0:
#             preds[kk]=1
#         else:
#             preds[kk]=0
    
    accs = accuracy_score(labs, preds)
    f1 = f1_score(labs, preds, average='macro')
    re = recall_score(labs, preds, average='macro')
    pr = precision_score(labs, preds, average='macro')
    fpr, tpr, thresholds = metrics.roc_curve(labs, probability, pos_label=1)
    auc = metrics.auc(fpr, tpr)    

    print(accs, f1, re, pr, auc)
    return accs, f1, re, pr, auc

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="DFD")
    config = yaml_config_hook("./config/config_single.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument("--lm", type=str, default="")
    parser.add_argument("--exp", type=int, default=None)
    args = parser.parse_args([])
    print(args)

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"
    args.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    data_compression = args.dataset_dir_test.split('/')[-2]

    results = []
    for i in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
        args.percent = i
        print('mask percent:', args.percent)
        accs, f1, re, pr, auc = detection( args)
        results.append([i, accs, f1, re, pr, auc])
        

    df = pd.DataFrame(results)
    df.columns=['masked ratio', 'Accuracy', 'F1-Score', 'Recall', 'Precision', 'AUC']
    
    # df.to_csv('./result_csv/result_r{}h{}L{}_FFonFF{}_new.csv'.format(args.use_model_percent, args.heads, args.L1_depth, data_compression), index=False)
    df.to_csv('./result_csv/FF++_right/result_r{}h{}L{}Frr{}lamda{}_{}_STFE_FE_MA.csv'.format(args.use_model_percent, args.heads, args.L1_depth, args.FRR, args.loss2_weight, data_compression), index=False)
    with open('./result_csv/pickle/result_r{}h{}L{}Frr{}lamda{}_{}_STFE_FE_MA.pickle'.format(args.use_model_percent, args.heads, args.L1_depth, args.FRR, args.loss2_weight, data_compression), 'wb') as f:
        pickle.dump(df, f)