"""Entry point."""

import argparse
import numpy as np
import random
import torch
import os.path as osp
import trainer as trainer
import os
import json


def build_args():
    parser = argparse.ArgumentParser(description='gcl')
    register_default_args(parser)
    args = parser.parse_args()

    return args


def register_default_args(parser):

    parser.add_argument("--dataset", type=str, default="Cora", required=False, 
                                        choices = ['Cora','CoraFull','Computers','Citeseer'] ,help="The input dataset.")
    parser.add_argument('--setting', type=str, default='task', choices=['task','class'], help='Type of continual learning')
    
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=False, required=False,
                        help="Set True for cuda")
    parser.add_argument('--n_tasks', type=int, default=4)

    # Controller

    parser.add_argument('--entropy_coeff', type=float, default=1e-4)
    parser.add_argument('--ema_baseline_decay', type=float, default=0.95)
    parser.add_argument('--controller_max_step', type=int, default=4, 
                                    help='step for controller parameters')
    parser.add_argument('--controller_optim', type=str, default='adam')
    parser.add_argument('--controller_lr', type=float, default=3.5e-4,
                        help="Controller lr")
    parser.add_argument('--controller_grad_clip', type=float, default=0)
    parser.add_argument('--tanh_c', type=float, default=2.5)
    parser.add_argument('--softmax_temperature', type=float, default=2.0)
    
    #Buffer
    parser.add_argument('--buffer_size', type=int, default= 1000, required=False,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, default= 16, required=False,
                        help='The mini-batch size of the memory buffer.')
    parser.add_argument('--alpha', type=float, default = 0.5, required=False,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, default = 0.5 , required=False,
                        help='Penalty weight.')

    # child model
    parser.add_argument('--channels_gnn', nargs='+', type=int, default=[20,20])
    parser.add_argument('--mp_nn', type=str, default='gcn', choices=['gcn', 'gat', 'sg'])
    parser.add_argument('--batch_size_nei', type=int, default= 16, required=False, help='The batch size of the graph neighbour sampling.')
    parser.add_argument("--epochs", type=int, default=250, help="number of training epochs")
    parser.add_argument("--heads", type=int, default=2,help="number of heads")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument('--task_override', type=bool, default=False)


    # Ablation study    
    parser.add_argument("--abl", type=int, default=3, required=False, choices=[0,1,2,3] , 
                                            help="0 : vanilla gnn, 1 : ER, 2: Only controller, 3 : Controller + ER ")
    parser.add_argument("--gpu_id", type=int, default=1, required=False)


def main(args):

    if args.cuda and torch.cuda.is_available():  # cuda is  available
        args.cuda = True
        print("\n\nTraining with cuda...\n")
    else:
        args.cuda = False
        print("\n\nTraining with cpu...\n")

    # Sanity check
    if not args.task_override:
        if args.dataset == "Cora":
            args.n_tasks = 3
            
        elif args.dataset == "Citeseer":
            args.n_tasks = 3
            
        elif args.dataset == 'CoraFull':
            args.n_tasks = 9

        elif args.dataset == 'Computers':
            args.n_tasks = 5

    
    print(f"\nArguments = {args}\n\n")

    if not osp.exists(osp.join(f'results/', f'{args.dataset}')):
        os.makedirs(osp.join(f'results/', f'{args.dataset}'))

    path = f'results/{args.dataset}/{args.dataset}_{args.setting}_{args.epochs}_{args.mp_nn}_{args.controller_max_step}_{args.abl}_'+'_'.join(list(map(str, args.channels_gnn)))

    with open(osp.join(f'results/', f'{args.dataset}',f'{args.dataset}_args.txt'),'w') as f:
        json.dump(args.__dict__, f, indent=2)
            
    with open(f'{path}.csv' , 'w') as f:
        f.write(f'{args.dataset} dataset\n\n')
    
    for times in range(5):
        torch.manual_seed(args.random_seed+times)
        np.random.seed(args.random_seed+times)
        random.seed(args.random_seed+times)
        if args.cuda:
            torch.cuda.manual_seed(args.random_seed+times)
        print('\n'*5,' Trial no. ',times,'\n'*5)

        trnr = trainer.Trainer(args,times, path)
        trnr.train()

if __name__ == "__main__":
    args = build_args()
    main(args)