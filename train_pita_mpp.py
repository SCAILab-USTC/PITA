import torch
import torch.nn as nn
import random
from test_function import evaluate_model
from einops import rearrange  
from types import SimpleNamespace
from utils.optimizer_utils import get_optimizer_and_scheduler
from models.avit import build_avit  
from solve_coef import run_STRidge_pipeline, run_STRidge_prediction
import sys
from utils.Stridge import *
import os
sys.path.append(['.','./../'])
os.environ['OMP_NUM_THREADS'] = '16'
import json
import time
import argparse
import torch
import numpy as np
import torch.nn as nn
from utils.grid import *
from timeit import default_timer
from torch.utils.tensorboard import SummaryWriter
from utils.utilities import count_parameters, get_grid, load_model_from_checkpoint, load_components_from_pretrained
from utils.criterion import SimpleLpLoss
from utils.griddataset import MixedTemporalDataset
from utils.AutomaticWeightedLoss import *
from utils.make_master_file import DATASET_DICT
from config.mpp_params import *
from config.mpp import get_args
args = get_args()
device = torch.device("cuda:{}".format(args.gpu))
awl = AutomaticWeightedLoss(3)
print(f"Current working directory: {os.getcwd()}")


############### load data and dataloader ###############
train_paths = args.train_paths
test_paths = args.test_paths
args.data_weights = [1] * len(args.train_paths) if len(args.data_weights) == 1 else args.data_weights
print('args',args)
train_dataset = MixedTemporalDataset(args.train_paths, args.ntrain_list, res=args.res, t_in = args.T_in, t_ar = args.T_ar,normalize=False,train=True, data_weights=args.data_weights)
if args.use_full_test == True:
    test_datasets = [MixedTemporalDataset(test_path, res=args.res, n_channels = train_dataset.n_channels,t_in = args.T_in, t_ar=-1, normalize=False, train=False) for i, test_path in enumerate(test_paths)]
else:
    test_datasets = [MixedTemporalDataset(test_path, res=args.res, n_channels = train_dataset.n_channels,t_in = args.T_in, t_ar=10, normalize=False, train=False, use_full_data = args.use_full_test) for i, test_path in enumerate(test_paths)]
test_datasets_full = [MixedTemporalDataset(test_path, res=args.res, n_channels = train_dataset.n_channels,t_in = args.T_in, t_ar=-1, normalize=False, train=False, use_full_data = True) for i, test_path in enumerate(test_paths)]
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=True, 
                                            num_workers=8)
test_loaders = [torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8) for test_dataset in test_datasets]
test_loaders_full = [torch.utils.data.DataLoader(test_datasets_full_, batch_size=args.batch_size, shuffle=False,num_workers=8) for test_datasets_full_ in test_datasets_full]

ntrain, ntests = len(train_dataset), [len(test_dataset) for test_dataset in test_datasets]

def load_model_and_print_structure(pretrained_ckpt_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_avit(params_mpp).to(device)  
    if pretrained_ckpt_path:  
        checkpoint = torch.load(pretrained_ckpt_path, map_location=device)
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)  
    return model

model_config = {
    'Ti': ('/opt/data/private/Project/ai4pde/multiple_physics_pretraining/pretrained/MPP_AViT_Ti', params_mpp_Ti),
    'S' : ('/opt/data/private/Project/ai4pde/multiple_physics_pretraining/pretrained/MPP_AViT_S', params_mpp_S),
    'B' : ('/opt/data/private/Project/ai4pde/multiple_physics_pretraining/pretrained/MPP_AViT_B', params_mpp_B),
    'L' : ('/opt/data/private/Project/ai4pde/multiple_physics_pretraining/pretrained/MPP_AViT_L', params_mpp_L),
}

try:
    pretrained_ckpt_path, params_mpp = model_config[args.model_size]
except KeyError:
    raise ValueError(f"Unknown model size: {args.model_size}. Supported sizes are {list(model_config.keys())}.")

model = load_model_and_print_structure(pretrained_ckpt_path=pretrained_ckpt_path)

def print_random_model_weight(model):
    parameters = list(model.named_parameters())
    chosen_param_name, chosen_param_tensor = random.choice(parameters)
    weight_values = chosen_param_tensor.data.cpu().numpy()
    print(f"Chosen parameter: {chosen_param_name}")
    print(f"Weight values shape: {weight_values.shape}")
    print("Weight values:", weight_values)
print_random_model_weight(model)

optimizer, scheduler = get_optimizer_and_scheduler(args, model, awl, train_loader)
def adjust_learning_rate(optimizer, epoch):
    """Set the learning rate to the initial value after warmup epochs."""
    if epoch < args.warmup_epochs:
        lr = args.lr * (epoch + 1) / args.warmup_epochs 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        scheduler.step()
        
        
############### Set up Random Seed ###############
torch.manual_seed(args.seed)
np.random.seed(args.seed)
comment = args.comment + '_{}_{}'.format(len(train_paths), ntrain)
log_path = f'./logs_{args.model}/'+ 'seed/'  + time.strftime('%m%d_%H_%M_%S') + args.train_paths[0]  + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
model_path = log_path + '/model.pth'

if args.use_writer:
    print(f"use log_path = {log_path}")
    writer = SummaryWriter(log_dir=log_path)
    fp = open(log_path + f'/logs_{args.train_paths[0]}.txt', 'w+',buffering=1)
    json.dump(vars(args), open(log_path + '/params.json', 'w'),indent=4)
    sys.stdout = fp
else:
    writer = None
for key, value in vars(args).items():
    print(f'{key}: {value}')
upper_bound_x, upper_bound_y, upper_bound_t, lower_bound_x, lower_bound_y, lower_bound_t = get_grid_bound_2D( dataset = args.train_paths[0],T_ar = args.T_ar)
lam,d_tol,maxit,STR_iters,normalize,l0_penalty ,print_best_tol,P= 1e-5, args.tol, 100,  10, 2, 1e-4,  False, args.poly_order


############### Solve the PDE Coefficients from the Ground Truth Data ###############
w_true, elapsed_time = run_STRidge_pipeline(train_loader=train_loader, device=device, args=args, upper_bound_x=upper_bound_x, upper_bound_y=upper_bound_y, upper_bound_t=upper_bound_t, lower_bound_x=lower_bound_x, lower_bound_y=lower_bound_y, lower_bound_t=lower_bound_t )
myloss = SimpleLpLoss(size_average=False)
clsloss = torch.nn.CrossEntropyLoss(reduction='sum')
iter_num = 0


############### Start Training ###############
for ep in range(args.epochs):
    model.train()
    t1 = t_1 = default_timer()
    t_load, t_train , train_l2_step, train_l2_full, data_loss_l2_step, data_loss_l2_full = 0., 0.,0.,0.,0.,0.
    loss_previous = np.inf
    torch.autograd.set_detect_anomaly(True) 
    
    for xx, yy, msk, cls in train_loader:
        t_load += default_timer() - t_1
        t_1 = default_timer()
        xx, yy, msk, cls = xx.to(device), yy.to(device), msk.to(device), cls.to(device)   ## B, n, n, T_in, C
        model.to(device)

        loss, physics_loss, coefficient_loss= 0. , 0. , 0.
        torch.cuda.empty_cache()
        ############### Auto-regressive Prediction ###############
        for t in range(0, yy.shape[-2], args.T_bundle): 
            
            torch.cuda.empty_cache()
            y = yy[..., t:t + args.T_bundle, :]
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
            
            ############### Governing Equations Discovery ###############
            w_best, phi_loss, elapsed_time = run_STRidge_prediction(  xx=xx, device=device, args=args, upper_bound_x=upper_bound_x, upper_bound_y=upper_bound_y, upper_bound_t=upper_bound_t, lower_bound_x=lower_bound_x,lower_bound_y=lower_bound_y, lower_bound_t=lower_bound_t, P=args.poly_order, lam=1e-5,d_tol=args.tol)
            
            ############### Physics-informed Constraints ################
            physics_loss = physics_loss + phi_loss.clone()  
            coefficient_loss = coefficient_loss + torch.mean((w_true.clone() - w_best.clone())**2) 
            
        data_loss_l2_step += loss.item()
        data_loss_l2_full += myloss(pred, yy, mask=msk).item()
        train_l2_step += (loss.item() + physics_loss + coefficient_loss)
        l2_full = myloss(pred, yy, mask=msk).item() + physics_loss + coefficient_loss
        train_l2_full += l2_full 

        optimizer.zero_grad()
        total_loss = awl(loss, physics_loss, coefficient_loss)
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        adjust_learning_rate(optimizer, ep) if args.lr_method == 'cosine' else scheduler.step()        train_l2_step_avg, train_l2_full_avg = train_l2_step / ntrain / (yy.shape[-2] / args.T_bundle), train_l2_full / ntrain
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
        
    ################ Complete Test Data ################
    test_l2_fulls_full_data, test_l2_steps_full_data = evaluate_model( model=model, test_loaders=test_loaders_full, ntests=ntests, test_paths=test_paths, ep=ep, log_path=log_path, args=args,writer=writer,save_tag="full_data")

    ################ Incomplete Test Data ################
    test_l2_fulls, test_l2_steps = evaluate_model( model=model, test_loaders=test_loaders, ntests=ntests, test_paths=test_paths, ep=ep, log_path=log_path, args=args,writer=writer,save_tag="incomplete_data")

    if args.use_writer:
        torch.save({'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, model_path)

    t_test = default_timer() - t_1
    t2 = t_1 = default_timer()
    lr = optimizer.param_groups[0]['lr']

    print('epoch {}, time {:.5f}, lr {:.2e}, train l2 step {:.5f} train l2 full {:.5f} data loss step {:.5f} data loss full {:.5f}, test10 l2 step [{}] test10 l2 full [{}], testfull l2 step ({}) testfull l2 full ({}), time train avg {:.5f} load avg {:.5f} test {:.5f} '.format(
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
            t_test
        ))
            
                    
        
