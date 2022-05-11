import pandas as pd
import numpy as np


def average_forgeting(n_tasks, score_matrix):
    backward = []
    for t in range(n_tasks-1):
        b = score_matrix[n_tasks-1][t]-score_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward),2)

    return mean_backward

def average_acc(score_matrix):
    return round( np.mean(score_matrix[-1]), 2)


def result_file(args, score_matrix, times,path):

    af = average_forgeting(args.n_tasks, score_matrix)
    aa = average_acc(score_matrix)
    
    rc = ['Task '+str(i) for i in range(1,args.n_tasks+1)]

    with open(f'{path}.csv' , 'a') as f:
        f.write(f'Trial no. {times}\n')
        pd.DataFrame(score_matrix, index=rc, columns=rc).to_csv(f)
        f.write(f'\n\tAverage accuracy = {aa}\n\tAverage forgetting = {af}\n')