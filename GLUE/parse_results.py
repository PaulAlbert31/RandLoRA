import sys
import json
import numpy as np
root = sys.argv[1]
nseeds = int(sys.argv[2])

datasets = ['sst2', 'mrpc', 'cola', 'qnli', 'rte', 'stsb']
metric = ['eval_accuracy', 'eval_accuracy', 'eval_matthews_correlation', 'eval_accuracy', 'eval_accuracy', 'eval_combined_score']


avg = [[] for _ in range(len(datasets))]
for i, (d, m) in enumerate(zip(datasets, metric)):
    for seed in range(1, nseeds+1):
        fname = f"{root}/{seed}/{d}/eval_results.json"
        f = json.load(open(fname))
        avg[i].append(f[m]*100)
        
avg = np.array(avg)
avg_mean = avg.mean(axis=1)
avg_std = avg.std(axis=1)
print(' '.join(datasets) + ' avg')
string = f"{root.split('/')[-1]}"
string += ' | '.join([f'{m:.1f} ± {s:.1f}' for m,s in zip(avg_mean, avg_std)])
string += f' | {avg.mean():.1f} ± {avg.mean(axis=0).std():.1f} \\\\' 
print(string)
