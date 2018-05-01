import os
import csv


def save_log(save_path, *args, **kwargs):
    columns = list(kwargs.keys())
    
    if not(os.path.exists(save_path)):
        with open(save_path, 'a') as f:
            csvWriter = csv.writer(f)
            csvWriter.writerow(columns)

    with open(save_path, 'a') as f:
        csvWriter = csv.writer(f)
        logs = list(kwargs.values())
        csvWriter.writerow(logs)

        
def confusion_values(predicted, labels):
    TP = 0
    FP = 0
    for c in range(150):
        TP += ((predicted == c + 1) & (labels.data == c + 1)).float().sum(1).sum(2)[:,0,0]
        FP += ((predicted != c + 1) & (labels.data == c + 1)).float().sum(1).sum(2)[:,0,0]

    TN = ((predicted == 0) & (labels.data == 0)).float().sum(1).sum(2)[:,0,0]
    FN = ((predicted != 0) & (labels.data == 0)).float().sum(1).sum(2)[:,0,0]
    
    return TP, FP, TN, FN