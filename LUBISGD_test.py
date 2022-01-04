from data import ImplicitData
import pandas as pd
import numpy as np
from recommenders_implicit import LocalUBISGD,UBISGD,BISGD,ISGD
from eval_implicit import EvalPrequential
from datetime import datetime
import getopt
import sys

argv = sys.argv[1:]

opts, args = getopt.getopt(argv, 'd:f:i:l:r:n:')

print(opts)

for opt, arg in opts:
    if opt in ['-d']:
        dataset = arg
    elif opt in ['-f']:
        num_factors = int(arg)
    elif opt in ['-i']:
        num_iter = int(arg)
    elif opt in ['-l']:
        learn_rate = float(arg)
    elif opt in ['-r']:
        regularization = float(arg)
    elif opt in ['-n']:
        interleave = int(arg)

data = pd.read_csv(dataset,"\t",header=None)
stream = ImplicitData(data[0],data[1])

num_nodes = [12,14,16,18,20,24,28,32]


# In[4]:

for nn in num_nodes:
    print("## ", nn, " ##")
    model = LocalUBISGD(ImplicitData([],[]), num_factors, num_iter, nn, learn_rate = learn_rate, u_regularization = regularization, i_regularization = regularization, random_seed = 10)

    eval = EvalPrequential(model,stream, metrics = ["Recall@20"])

    start_recommend = datetime.now()
    print('start time', start_recommend)

    results=eval.EvaluateTime(0,stream.size, interleave)

    print('npmean(resuls[Recall@20])', np.mean(results['Recall@20']))

    end_recommend = datetime.now()
    print('end time', end_recommend)

    tempo = end_recommend - start_recommend

    print('run time', tempo)
    print('')
    print('get tuple',np.mean(results['time_get_tuple']))
    print('recommend',np.mean(results['time_recommend']))
    print('eval_point',np.mean(results['time_eval_point']))
    print('update',np.mean(results['time_update']))


