"""Entry point."""

import argparse
import time
import torch
import os.path as osp
import trainer as trainer
import os


def build_args():
    parser = argparse.ArgumentParser(description='gcl')
    register_default_args(parser)
    args = parser.parse_args()

    return args


def register_default_args(parser):
    
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=False, required=False,
                        help="run in cuda mode")
    parser.add_argument("--dataset", type=str, default="Citeseer", required=False,
                        help="The input dataset.")
    parser.add_argument('--n_tasks', type=int, default=4)

    # Controller

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])
    parser.add_argument('--entropy_coeff', type=float, default=1e-4)
    parser.add_argument('--shared_rnn_max_length', type=int, default=35)

    parser.add_argument('--ema_baseline_decay', type=float, default=0.95)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--controller_max_step', type=int, default=5,
                        help='step for controller parameters')
    parser.add_argument('--controller_optim', type=str, default='adam')
    parser.add_argument('--controller_lr', type=float, default=3.5e-4,
                        help="will be ignored if --controller_lr_cosine=True")
    parser.add_argument('--controller_grad_clip', type=float, default=0)
    parser.add_argument('--tanh_c', type=float, default=2.5)
    parser.add_argument('--softmax_temperature', type=float, default=5.0)
    
    #buffer
    parser.add_argument('--buffer_size', type=int, default= 6000, required=False, 
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, default= 128, required=False,
                        help='The mini-batch size of the memory buffer.')
    parser.add_argument('--batch_size_nei', type=int, default= 100, required=False,
                        help='The batch size of the graph neighbour sampling.')
    parser.add_argument('--alpha', type=float, default = 0.5, required=False,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, default = 0.5 , required=False,
                        help='Penalty weight.')

    # child model
    parser.add_argument('--channels_gnn', nargs='+', type=int, default=[16,16])
    parser.add_argument('--channels_mlp', nargs='+', type=int, default=[5,6])
    parser.add_argument('--mp_nn', type=str, default='gcn', choices=['gcn', 'gat', 'sg'])


    parser.add_argument("--epochs", type=int, default=250,
                        help="number of training epochs")
    parser.add_argument("--heads", type=int, default=1,
                        help="number of heads")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--param_file", type=str, default="cora_test.pkl",
                        help="learning rate")
    parser.add_argument("--optim_file", type=str, default="opt_cora_test.pkl",
                        help="optimizer save path")
    parser.add_argument('--max_param', type=float, default=5E6)
    parser.add_argument('--logger_file', type=str, default=f"logger_file_{time.time()}.txt")
    parser.add_argument('--task_override', type=bool, default=False)

def main(args):

    if args.cuda and torch.cuda.is_available():  # cuda is  available
        args.cuda = True
        print("\n\nTraining with cuda...\n")
    else:
        args.cuda = False
        print("\n\nTraining with cpu...\n")
    # args.epochs = 4
    # args.controller_max_step = 2

    # Sanity check
    if not args.task_override:
        if args.dataset == "Cora":
            args.n_tasks = 3
            
        elif args.dataset == "Citeseer":
            args.n_tasks = 3
            
        elif args.dataset == "Reddit":
            args.n_tasks = 8
        
        elif args.dataset == 'ENZYMES':
            args.n_tasks = 3


    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # utils.makedirs(args.dataset)
    
    print(f"\nArguments = {args}\n\n")

    if not osp.exists(osp.join(f'results/', f'{args.dataset}',  f'{args.dataset}_{args.mp_nn}.csv')):
        os.makedirs(osp.join(f'results/', f'{args.dataset}'))

    with open(osp.join(f'results/', f'{args.dataset}',  f'{args.dataset}_{args.mp_nn}.csv') , 'w') as f:
        f.write(f'{args.dataset} dataset\n\n')
    
    for times in range(5):
        print('\n'*10,' Trial no. ',times,'\n'*10)

        trnr = trainer.Trainer(args,times)
        trnr.train()

if __name__ == "__main__":
    args = build_args()
    main(args)