import pandas as pd
import numpy as np
import os.path as osp

# Error present

def average_forgeting(n_tasks, score_matrix):
    backward = []
    for t in range(n_tasks-1):
        b = score_matrix[n_tasks-1][t]-score_matrix[t][t]
        backward.append(round(b*100, 2))
    mean_backward = round(np.mean(backward),2)

    return mean_backward  


def result_file(args, score_matrix, times):
    # mean_backward = average_forgeting(args.n_tasks, score_matrix)
    # print(f'\n\n Average forgetting = {mean_backward}',end='\n\n')
    # print(f'\n\n Average accuracy = {score_matrix}',end='\n\n')

    # af = pd.DataFrame(mean_backward, index=list(range(args.n_tasks-1)), columns=['Average forgetting'])
    # af.to_csv(osp.join(f'{args.dataset}' ,f'{args.dataset}_{args.mp_nn}_forgetting.csv'), )

    rc = ['Task '+str(i) for i in range(1,args.n_tasks+1)]

    with open(osp.join(f'results/', f'{args.dataset}',  f'{args.dataset}_{args.mp_nn}.csv') , 'a') as f:
        f.write(f'Trial no. {times}\n')
        pd.DataFrame(score_matrix, index=rc, columns=rc).to_csv(f)
        f.write('\n')
