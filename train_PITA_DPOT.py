import sys
from Stridge import *
import os
sys.path.append(['.','./../'])
os.environ['OMP_NUM_THREADS'] = '16'

import json
import time
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utils.grid import *

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.tensorboard import SummaryWriter
from utils.optimizer import Adam, Lamb
from utils.utilities import count_parameters, get_grid, load_model_from_checkpoint, load_components_from_pretrained
from utils.criterion import SimpleLpLoss
from utils.griddataset import MixedTemporalDataset
from utils.AutomaticWeightedLoss import *
from utils.make_master_file import DATASET_DICT
from models.fno import FNO2d
from models.dpot import DPOTNet
import pickle




################################################################
# configs

################################################################


parser = argparse.ArgumentParser(description='Training or pretraining for the same data type')

### currently no influence
# model_size: Ti
parser.add_argument('--model', type=str, default='DPOT')
parser.add_argument('--model_size', type=str, default='Ti')
parser.add_argument('--des', type=str, default='ns2d_fno_1e-3')
parser.add_argument('--dataset',type=str, default='ns2d')

parser.add_argument('--train_paths',nargs='+', type=str, default=['dr_pdb'])
parser.add_argument('--test_paths',nargs='+',type=str, default=['dr_pdb'])
parser.add_argument('--resume_path',type=str, default='models/DPOT/model_S.pth')
parser.add_argument('--ntrain_list', nargs='+', type=int, default=[900])
parser.add_argument('--data_weights',nargs='+',type=int, default=[1])
parser.add_argument('--use_writer', action='store_true',default=False)

parser.add_argument('--res', type=int, default=128)
parser.add_argument('--noise_scale',type=float, default=0.0)
# parser.add_argument('--n_channels',type=int,default=-1)

### shared params
parser.add_argument('--width', type=int, default=512)
parser.add_argument('--n_layers',type=int, default=4)
parser.add_argument('--act',type=str, default='gelu')

### GNOT params
parser.add_argument('--max_nodes',type=int, default=-1)

### FNO params
parser.add_argument('--modes', type=int, default=32)
parser.add_argument('--use_ln',type=int, default=0)
parser.add_argument('--normalize',type=int, default=0)

### AFNO
parser.add_argument('--patch_size',type=int, default=8)
parser.add_argument('--n_blocks',type=int, default=8)
parser.add_argument('--mlp_ratio',type=int, default=1)
parser.add_argument('--out_layer_dim', type=int, default=32)

parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--opt',type=str, default='adam', choices=['adam','lamb'])
parser.add_argument('--beta1',type=float,default=0.9)
parser.add_argument('--beta2',type=float,default=0.9)
parser.add_argument('--lr_method',type=str, default='step')
parser.add_argument('--grad_clip',type=float, default=10000.0)
parser.add_argument('--step_size', type=int, default=100)
parser.add_argument('--step_gamma', type=float, default=0.5)
parser.add_argument('--warmup_epochs',type=int, default=50)
parser.add_argument('--sub', type=int, default=1)
parser.add_argument('--T_in', type=int, default=10)
parser.add_argument('--T_ar', type=int, default=1)
parser.add_argument('--T_bundle', type=int, default=1)
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--comment',type=str, default="")
parser.add_argument('--log_path',type=str,default='')
### finetuning parameters
parser.add_argument('--n_channels',type=int, default=4)
parser.add_argument('--n_class',type=int,default=12)
parser.add_argument('--load_components',nargs='+', type=str, default=['blocks','pos','time_agg'])
parser.add_argument('--batch_down',type=int,default=5)
parser.add_argument('--grid_xy',type=int,default=4)
parser.add_argument('--time_cut',type=int,default=3)
parser.add_argument('--seed',type=int,default=3407)
parser.add_argument('--use_full_test' ,action='store_true', default=False)
# parser.add_argument('--use_full_test', type=str, default='False')

args = parser.parse_args()


device = torch.device("cuda:{}".format(args.gpu))
awl = AutomaticWeightedLoss(3)
print(f"Current working directory: {os.getcwd()}")




################################################################
# load data and dataloader
################################################################
train_paths = args.train_paths
test_paths = args.test_paths
args.data_weights = [1] * len(args.train_paths) if len(args.data_weights) == 1 else args.data_weights
print('args',args)

# full_dataset = MixedTemporalDataset(args.train_paths, args.ntrain_list, res=args.res, t_in = args.T_in, t_ar = args.T_ar, normalize=False,train=True, data_weights=args.data_weights)

train_dataset = MixedTemporalDataset(args.train_paths, args.ntrain_list, res=args.res, t_in = args.T_in, t_ar = args.T_ar, normalize=False,train=True, data_weights=args.data_weights)
if args.use_full_test == True:
    test_datasets = [MixedTemporalDataset(test_path, res=args.res, n_channels = train_dataset.n_channels,t_in = args.T_in, t_ar=-1, normalize=False, train=False) for i, test_path in enumerate(test_paths)]
    print(f"The full length of testing data is used!")
elif args.use_full_test == False:
    test_datasets = [MixedTemporalDataset(test_path, res=args.res, n_channels = train_dataset.n_channels,t_in = args.T_in, t_ar=10, normalize=False, train=False, use_full_data = args.use_full_test) for i, test_path in enumerate(test_paths)]
    print(f"The length of 10 testing data is used!")
test_datasets_full = [MixedTemporalDataset(test_path, res=args.res, n_channels = train_dataset.n_channels,t_in = args.T_in, t_ar=-1, normalize=False, train=False, use_full_data = True) for i, test_path in enumerate(test_paths)]
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=True, 
                                            num_workers=8)
test_loaders = [torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8) for test_dataset in test_datasets]

test_loaders_full = [torch.utils.data.DataLoader(test_datasets_full_, batch_size=args.batch_size, shuffle=False,num_workers=8) for test_datasets_full_ in test_datasets_full]

ntrain, ntests = len(train_dataset), [len(test_dataset) for test_dataset in test_datasets]
print('Train num {} test num {}'.format(train_dataset.n_sizes, ntests))
if args.model == "FNO":
    model = FNO2d(args.modes, args.modes, args.width, img_size = args.res, patch_size=args.patch_size, in_timesteps = args.T_in, out_timesteps=1,normalize=args.normalize,n_layers = args.n_layers,use_ln = args.use_ln, n_channels=train_dataset.n_channels, n_cls=len(args.train_paths)).to(device)
elif args.model == 'DPOT':
    if args.resume_path!= '':
        if "model_S" in args.resume_path:
            model = DPOTNet(img_size=args.res, 
                        patch_size=8, 
                        mixing_type='afno', 
                        in_channels=train_dataset.n_channels, 
                        in_timesteps=10, 
                        out_timesteps=1, 
                        out_channels=train_dataset.n_channels, 
                        normalize=False, 
                        embed_dim=1024, 
                        modes=32, 
                        depth=6, 
                        n_blocks=8, 
                        mlp_ratio=1, 
                        out_layer_dim=32, 
                        n_cls=12).to(device)        # S模型
        elif "model_Ti" in args.resume_path:
            model = DPOTNet(img_size=args.res, 
                patch_size=8, 
                mixing_type='afno', 
                in_channels=train_dataset.n_channels, 
                in_timesteps=10, 
                out_timesteps=1, 
                out_channels=train_dataset.n_channels, 
                normalize=False, 
                embed_dim=512, 
                modes=32, 
                depth=4, 
                n_blocks=4, 
                mlp_ratio=1, 
                out_layer_dim=32, 
                n_cls=12).to(device)
        elif "model_M" in args.resume_path:
            model = DPOTNet(img_size=args.res, 
                        patch_size=8, 
                        mixing_type='afno', 
                        in_channels=train_dataset.n_channels, 
                        in_timesteps=10, 
                        out_timesteps=1, 
                        out_channels=train_dataset.n_channels, 
                        normalize=False, 
                        embed_dim=1024, 
                        modes=32, 
                        depth=12, 
                        n_blocks=8, 
                        mlp_ratio=4, 
                        out_layer_dim=32, 
                        n_cls=12).to(device)
        else:
            model = DPOTNet(img_size=args.res, 
                patch_size=8, 
                mixing_type='afno', 
                in_channels=train_dataset.n_channels, 
                in_timesteps=10, 
                out_timesteps=1, 
                out_channels=train_dataset.n_channels, 
                normalize=False, 
                embed_dim=1536, 
                modes=32, 
                depth=24, 
                n_blocks=16, 
                mlp_ratio=4, 
                out_layer_dim=32, 
                n_cls=12).to(device)
    else:
        if args.model_size == 'Ti':
            model = DPOTNet(img_size=args.res, 
                patch_size=8, 
                mixing_type='afno', 
                in_channels=train_dataset.n_channels, 
                in_timesteps=10, 
                out_timesteps=1, 
                out_channels=train_dataset.n_channels, 
                normalize=False, 
                embed_dim=512, 
                modes=32, 
                depth=4, 
                n_blocks=4, 
                mlp_ratio=1, 
                out_layer_dim=32, 
                n_cls=12).to(device)
        elif args.model_size == 'S':
            model = DPOTNet(img_size=args.res, 
                patch_size=8, 
                mixing_type='afno', 
                in_channels=train_dataset.n_channels, 
                in_timesteps=10, 
                out_timesteps=1, 
                out_channels=train_dataset.n_channels, 
                normalize=False, 
                embed_dim=1024, 
                modes=32, 
                depth=6, 
                n_blocks=8, 
                mlp_ratio=1, 
                out_layer_dim=32, 
                n_cls=12).to(device)        # S模型
            
        elif args.model_size == 'M':
            model = DPOTNet(img_size=args.res, 
                        patch_size=8, 
                        mixing_type='afno', 
                        in_channels=train_dataset.n_channels, 
                        in_timesteps=10, 
                        out_timesteps=1, 
                        out_channels=train_dataset.n_channels, 
                        normalize=False, 
                        embed_dim=1024, 
                        modes=32, 
                        depth=12, 
                        n_blocks=8, 
                        mlp_ratio=4, 
                        out_layer_dim=32, 
                        n_cls=12).to(device)
        else:
            model = DPOTNet(img_size=args.res, 
                patch_size=8, 
                mixing_type='afno', 
                in_channels=train_dataset.n_channels, 
                in_timesteps=10, 
                out_timesteps=1, 
                out_channels=train_dataset.n_channels, 
                normalize=False, 
                embed_dim=1536, 
                modes=32, 
                depth=24, 
                n_blocks=16, 
                mlp_ratio=4, 
                out_layer_dim=32, 
                n_cls=12).to(device)            
else:
    raise NotImplementedError

if args.resume_path:
    print('Loading models and fine tune from {}'.format(args.resume_path))
    load_components_from_pretrained(model, torch.load(args.resume_path,map_location='cuda:{}'.format(args.gpu))['model'], components=args.load_components)

#### set optimizer
if args.opt == 'lamb':
    optimizer = Lamb(model.parameters(), lr=args.lr, betas = (args.beta1, args.beta2), adam=True, debias=False,weight_decay=1e-4)
else:
    optimizer = Adam( [{"params":model.parameters(), "lr": args.lr, "betas":(args.beta1, args.beta2), "weight_decay":1e-6},
                       {"params":awl.parameters(),"lr": args.lr, "betas":(args.beta1, args.beta2), "weight_decay":1e-6 }
                       ])


if args.lr_method == 'cycle':
    print('Using cycle learning rate schedule')
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, div_factor=1e4, pct_start=(args.warmup_epochs / args.epochs), final_div_factor=1e4, steps_per_epoch=len(train_loader), epochs=args.epochs)
elif args.lr_method == 'step':
    print('Using step learning rate schedule')
    scheduler = StepLR(optimizer, step_size=args.step_size * len(train_loader), gamma=args.step_gamma)
elif args.lr_method == 'warmup':
    print('Using warmup learning rate schedule')
    scheduler = LambdaLR(optimizer, lambda steps: min((steps + 1) / (args.warmup_epochs * len(train_loader)), np.power(args.warmup_epochs * len(train_loader) / float(steps + 1), 0.5)))
elif args.lr_method == 'linear':
    print('Using warmup learning rate schedule')
    scheduler = LambdaLR(optimizer, lambda steps: (1 - steps / (args.epochs * len(train_loader))))
elif args.lr_method == 'restart':
    print('Using cos anneal restart')
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader) * args.lr_step_size, eta_min=0.)
elif args.lr_method == 'cyclic':
    scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=args.lr_step_size * len(train_loader),mode='triangular2', cycle_momentum=False)
else:
    raise NotImplementedError


torch.manual_seed(args.seed)
np.random.seed(args.seed)

comment = args.comment + '_{}_{}'.format(len(train_paths), ntrain)
if args.resume_path == '':
    model_size = args.model_size
else:
    model_size = args.resume_path[-5] 

log_path = f'./logs_Auto_T_ar=1_use_full_{args.use_full_test}/'+model_size+'/' + time.strftime('%m%d_%H_%M_%S') + args.train_paths[0]  + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
# log_path = f'./huagjyhbfytrfd{args.use_full_test}/'+args.resume_path[-5]+'/' + time.strftime('%m%d_%H_%M_%S') + args.train_paths[0]  + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
print(f"args.use_writer = {args.use_writer}")
model_path = log_path + '/model.pth'
if args.use_writer:
    print(f"use log_path = {log_path}")
    writer = SummaryWriter(log_dir=log_path)
    fp = open(log_path + f'/logs_{args.train_paths[0]}.txt', 'w+',buffering=1)
    json.dump(vars(args), open(log_path + '/params.json', 'w'),indent=4)
    sys.stdout = fp


else:
    writer = None
print(model)
count_parameters(model)
print(f"len(train_loader) = {len(train_loader)}")
print('args',args)
for key, value in vars(args).items():
    print(f'{key}: {value}')
upper_bound_x, upper_bound_y, upper_bound_t, lower_bound_x, lower_bound_y, lower_bound_t = get_grid_bound_2D(
    dataset = args.train_paths[0],
    T_ar = args.T_ar
) 
print(f"upper_bound_x = {upper_bound_x}\n upper_bound_y = {upper_bound_y}\n upper_bound_t = {upper_bound_t}\n lower_bound_x = {lower_bound_x}\n lower_bound_y = {lower_bound_y}\n lower_bound_t = {lower_bound_t} ")
full_data,_,_,_ = next(iter(train_loader))
full_data = full_data.to(device)

U_all_full, U_t_full = U_all_compute( 
                                    full_data,
                                    device = device,
                                    upper_bound_x = upper_bound_x,
                                    upper_bound_y = upper_bound_y,
                                    upper_bound_t = upper_bound_t,
                                    
                                    lower_bound_t = lower_bound_t,          
                                    lower_bound_x = lower_bound_x,
                                    lower_bound_y = lower_bound_y)
U = full_data.unsqueeze(0)
lib_poly = lib_poly_compute(P = 3,
                            U_all=U_all_full,
                            U = U
)

lam,d_tol,maxit,STR_iters,normalize,l0_penalty ,print_best_tol= 1e-5, 1, 100,  10, 2, 1e-4,  False

lambda_w = torch.randn([ lib_poly.shape[0],1]).to(device) # 比如16：1

start_time = time.time()
w_true,_ = Train_STRidge(lib_poly=lib_poly,
                    U_t=U_t_full,
                    device=device,
                    lam=lam,
                    maxit=maxit,
                    normalize=normalize,
                    lambda_w=lambda_w,
                    l0_penalty=l0_penalty,
                    print_best_tol=False,
                    d_tol = d_tol)
            
myloss = SimpleLpLoss(size_average=False)
clsloss = torch.nn.CrossEntropyLoss(reduction='sum')
iter_num = 0

for ep in range(args.epochs):
    model.train()
    t1 = t_1 = default_timer()
    t_load, t_train = 0., 0.
    train_l2_step = 0
    train_l2_full = 0
    data_loss_l2_step = 0
    data_loss_l2_full = 0
    loss_previous = np.inf
    torch.autograd.set_detect_anomaly(True) 
    
    for xx, yy, msk, cls in train_loader:
        t_load += default_timer() - t_1
        t_1 = default_timer()
        xx = xx.to(device)  ## B, n, n, T_in, C
        yy = yy.to(device)  ## B, n, n, T_ar, C
        msk = msk.to(device)
        cls = cls.to(device)

        model.to(device)

        loss, physics_loss, coefficient_loss= 0. , 0. , 0.
        torch.cuda.empty_cache()

        ## auto-regressive training loop, support 1. noise injection, 2. long rollout backward, 3. temporal bundling prediction
        for t in range(0, yy.shape[-2], args.T_bundle): 
            
            torch.cuda.empty_cache()
            y = yy[..., t:t + args.T_bundle, :]
            ### auto-regressive training
            xx = xx + args.noise_scale *torch.sum(xx**2, dim=(1,2,3), keepdim=True)**0.5 * torch.randn_like(xx)
            im, _ = model(xx)
            im_copy = im.clone()
            y_copy = y.clone()      
            loss += myloss(im_copy, y_copy, mask=msk)

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), dim=-2)
                
            xx = torch.cat((xx[..., args.T_bundle:, :], im), dim=-2)
            batch_size, x_size, y_size, t_size, channels = xx.shape
            xx_downsampled = downsample_modified(xx,
                            batch = args.batch_down    ,     
                            grid_xy = args.grid_xy  ,    
                            time_hold = args.time_cut ,      
                            device = device,
                            seed = args.seed
                            )
            U_all , U_t = U_all_compute(xx_downsampled,
                                        device = device,
                                        upper_bound_x = upper_bound_x,
                                        upper_bound_y = upper_bound_y,
                                        upper_bound_t = upper_bound_t,
                                        
                                        lower_bound_t = lower_bound_t,          
                                        lower_bound_x = lower_bound_x,
                                        lower_bound_y = lower_bound_y)
            
            U = xx_downsampled.unsqueeze(0)
            lib_poly = lib_poly_compute(P=3, 
                                        U_all = U_all,
                                        U = U)

            lambda_w = torch.randn([ lib_poly.shape[0],1]).to(device) 
            
            start_time = time.time()
            
                
            w_best, phi_loss = Train_STRidge(lib_poly=lib_poly,
                                U_t=U_t,
                                device=device,
                                lam=lam,
                                maxit=maxit,
                                normalize=normalize,
                                lambda_w=lambda_w,
                                l0_penalty=l0_penalty,
                                print_best_tol=False,
                                d_tol = d_tol)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            # physics_loss = physics_loss + phi_loss

            phi_loss_copy = phi_loss.clone()    #.detach()
            physics_loss = physics_loss + phi_loss_copy
            w_best_copy = w_best.clone()
            w_true_copy = w_true.clone()
            coefficient_loss = coefficient_loss + torch.mean((w_true_copy - w_best_copy)**2) 
            
            
        
        data_loss_l2_step += loss.item()
        data_loss_l2_full += myloss(pred, yy, mask=msk).item()
        train_l2_step += (loss.item() + physics_loss + coefficient_loss)
        l2_full = myloss(pred, yy, mask=msk).item() + physics_loss + coefficient_loss
        # print(f"myloss = {myloss} \t type(myloss) = {type(myloss)}")
        train_l2_full += l2_full    #.item()

        optimizer.zero_grad()
        
        loss_ = loss
        coefficient_loss_ = coefficient_loss
        physics_loss_ = physics_loss
        coefficient_loss_.requires_grad_(True)
        # coefficient_loss_.backward(retain_graph=True)
        physics_loss_.requires_grad_(True)
        # physics_loss_.backward(retain_graph=True)
        loss_.requires_grad_(True)
        # loss_.backward()
        a=  time.time()        
        # print( f"loss = {loss}\t coefficient_loss = {coefficient_loss}\t physics_loss = {physics_loss}")
        total_loss = awl(loss_, physics_loss_, coefficient_loss_)

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        
        train_l2_step_avg, train_l2_full_avg = train_l2_step / ntrain / (yy.shape[-2] / args.T_bundle), train_l2_full / ntrain
        data_loss_l2_step_avg, data_loss_l2_full_avg = data_loss_l2_step / ntrain / (yy.shape[-2] / args.T_bundle), data_loss_l2_full / ntrain
      

        iter_num +=1
        if args.use_writer:
            writer.add_scalar("train_loss_step", loss.item()/(xx.shape[0] * yy.shape[-2] / args.T_bundle), iter_num)
            writer.add_scalar("train_loss_full", l2_full / xx.shape[0], iter_num)

            ## reset model
            if loss.item() > 10 * loss_previous : # or (ep > 50 and l2_full / xx.shape[0] > 0.9):
                print('loss explodes, loading model from previous epoch')
                checkpoint = torch.load(model_path,map_location='cuda:{}'.format(args.gpu))
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint["optimizer"])
                loss_previous = loss.item()

        t_train += default_timer() -  t_1
        t_1 = default_timer()
        

    test_l2_fulls_full_data, test_l2_steps_full_data = [], []
    with torch.no_grad():
        model.eval()
        for id, test_loader in enumerate(test_loaders_full):
            test_l2_full_full_data, test_l2_step_full_data = 0, 0
            for xx, yy, msk, _ in test_loader:
                # print(f"in test_loaders_full, yy.shape = {yy.shape}\txx.shape = {xx.shape}, len(test_loaders_full) = {len(test_loaders_full) }")
                length = yy.shape[-2]
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)
                msk = msk.to(device)


                for t in range(0, yy.shape[-2], args.T_bundle):
                    y = yy[..., t:t + args.T_bundle, :]
                    im, _ = model(xx)
                    loss += myloss(im, y, mask=msk)

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -2)

                    xx = torch.cat((xx[..., args.T_bundle:,:], im), dim=-2)

                test_l2_step_full_data += loss.item()
                test_l2_full_full_data += myloss(pred, yy, mask=msk)
            file_name = os.path.join(args.train_paths[0], "save")
            if (ep + 1) % 500 == 0:
                os.makedirs(file_name, exist_ok=True)
                os.makedirs(f'{log_path}/{file_name}', exist_ok=True)
                
                # 保存 tensor
                save_path = f'{log_path}/{file_name}/test_{ep}.pt'
                torch.save({'ground_truth': yy, 'prediction': pred}, save_path)
                print(f"Saved tensors to {save_path}")
                
            test_l2_step_avg_full_data, test_l2_full_avg_full_data = test_l2_step_full_data / ntests[id] / (yy.shape[-2] / args.T_bundle), test_l2_full_full_data / ntests[id]
            test_l2_steps_full_data.append(test_l2_step_avg_full_data)
            test_l2_fulls_full_data.append(test_l2_full_avg_full_data)
            if args.use_writer:
                writer.add_scalar("test_loss_step_{}".format(test_paths[id]), test_l2_step_avg_full_data, ep)
                writer.add_scalar("test_loss_full_{}".format(test_paths[id]), test_l2_full_avg_full_data, ep)


    test_l2_fulls, test_l2_steps = [], []
    with torch.no_grad():
        model.eval()
        for id, test_loader in enumerate(test_loaders):
            test_l2_full, test_l2_step = 0, 0
            for xx, yy, msk, _ in test_loader:
                # print(f"in test_loaders: yy.shape = {yy.shape}\txx.shape = {xx.shape}, len(test_loaders) = {len(test_loaders)}")
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)
                msk = msk.to(device)
                
                for t in range(0, yy.shape[-2], args.T_bundle):
                    y = yy[..., t:t + args.T_bundle, :]
                    im, _ = model(xx)
                    loss += myloss(im, y, mask=msk)

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -2)

                    xx = torch.cat((xx[..., args.T_bundle:,:], im), dim=-2)

                test_l2_step += loss.item()
                test_l2_full += myloss(pred, yy, mask=msk)

            test_l2_step_avg, test_l2_full_avg = test_l2_step / ntests[id] / (yy.shape[-2] / args.T_bundle), test_l2_full / ntests[id]
            # print(f"yy.shape[-2] = {yy.shape[-2]}")
            test_l2_steps.append(test_l2_step_avg)
            test_l2_fulls.append(test_l2_full_avg)
            if args.use_writer:
                writer.add_scalar("test_loss_step_{}".format(test_paths[id]), test_l2_step_avg, ep)
                writer.add_scalar("test_loss_full_{}".format(test_paths[id]), test_l2_full_avg, ep)

    if args.use_writer:
        torch.save({'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, model_path)

    t_test = default_timer() - t_1
    t2 = t_1 = default_timer()
    lr = optimizer.param_groups[0]['lr']
    
    print('epoch {}, time {:.5f}, lr {:.2e}, train l2 step {:.5f} train l2 full {:.5f} data loss step {:.5f} data loss full {:.5f}, test10 l2 step [{}] test10 l2 full [{}], testfull l2 step ({}) testfull l2 full ({}), time train avg {:.5f} load avg {:.5f} test {:.5f} length {}'.format(
    ep, 
    t2 - t1, 
    lr, 
    train_l2_step_avg, 
    train_l2_full_avg, 
    data_loss_l2_step_avg, 
    data_loss_l2_full_avg,
    ', '.join(['{:.5f}'.format(val) for val in test_l2_steps]),
    ', '.join(['{:.5f}'.format(val) for val in test_l2_fulls]),
    ', '.join(['{:.5f}'.format(val) for val in test_l2_steps_full_data]),
    ', '.join(['{:.5f}'.format(val) for val in test_l2_fulls_full_data]),
    t_train / len(train_loader), 
    t_load / len(train_loader), 
    t_test,
    length
))
