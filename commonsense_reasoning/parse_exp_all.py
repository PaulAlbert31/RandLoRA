import sys
import json
import os
datasets = ["boolq.txt", "piqa.txt", "social_i_qa.txt",  "hellaswag.txt",  "winogrande.txt", "ARC-Easy.txt", "ARC-Challenge.txt", "openbookqa.txt"]


accuracies = []
for d in datasets:
    fi = os.path.join(sys.argv[1], d)
    if os.path.isfile(fi):
        with open(fi, 'r') as f:
            lines = f.readlines()
            if len(lines) < 10:
                accuracies.append(0)
            else:
                for i in range(10):
                    l = lines[-(i+2)]
                    if l[:4] == 'test':
                        accuracies.append(float(l.split(' ')[-1]))
                        break
    else:
        accuracies.append(0)
                    

print(' & ' + ' & '.join([d.split('.')[0] for d in datasets]) + '& Average \\\\')
      
print(' & ' + ' & '.join([f'{a*100:.2f}' for a in accuracies]) + f' & {sum(accuracies)/len(accuracies) * 100.:.2f} \\\\')
