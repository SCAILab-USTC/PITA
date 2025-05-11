# config.py
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Training or pretraining for the same data type')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--model_size', type=str, default='Ti')
    parser.add_argument('--des', type=str, default='ns2d_fno_1e-3')
    parser.add_argument('--dataset',type=str, default='ns2d')
    parser.add_argument('--train_paths',nargs='+', type=str, default=['dr_pdb'])
    parser.add_argument('--test_paths',nargs='+',type=str, default=['dr_pdb'])
    parser.add_argument('--resume_path',type=str, default='')
    parser.add_argument('--ntrain_list', nargs='+', type=int, default=[900])
    parser.add_argument('--data_weights',nargs='+',type=int, default=[1])
    parser.add_argument('--use_writer', action='store_true',default=True)

    parser.add_argument('--warmup_steps',type=int, default=1000)
    parser.add_argument('--sched_epochs',type=int, default=500)
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

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_step_size', type=int, default=10)
    parser.add_argument('--opt',type=str, default='adam', choices=['adam','lamb'])
    parser.add_argument('--beta1',type=float,default=0.9)
    parser.add_argument('--beta2',type=float,default=0.999)
    parser.add_argument('--lr_method',type=str, default='step')
    parser.add_argument('--grad_clip',type=float, default=1.0)
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--step_gamma', type=float, default=0.5)
    parser.add_argument('--warmup_epochs',type=int, default=5)
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
    return parser.parse_args()
