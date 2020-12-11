import numpy as np
import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem

def open_file(path, skipfirst=True, split=False):
    with  open(path) as smi: 
        if skipfirst:           # gzip.open(path) as smi:
            smi.readline()
        lines = smi.readlines()
        # for i in range(len(lines)):
        if split==True:
            for i in range(len(lines)):
                lines[i] = lines[i].split()
                lines[i][1] = float(lines[i][1])
            print(str(lines) + "\n")
    return lines

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset to train")
# parser.add_argument("--mean", help="mean")
# parser.add_argument("--std", help="std")
args = parser.parse_args()
dataset = args.dataset
# mean = args.mean
# std = args.std

mean_std=open_file(dataset+'_mean_std.txt',False,True)
mean = mean_std[0][1]
std = mean_std[1][1]

delaney = pd.read_csv(dataset + "_train.csv", skiprows=1,
                      names=['id', 'measured', 'predicted', 'SMILES'])

delaney['measured'] = delaney['measured'] * float(std) + float(mean)
delaney.to_csv("unity_"+dataset+'.csv', index=False)