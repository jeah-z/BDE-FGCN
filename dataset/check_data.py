import numpy as np
import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset to train")
args = parser.parse_args()
dataset = args.dataset
dataset = pd.read_csv(
    dataset, names=['smi', 'atm1', 'atm2', 'bde', 'type', 'res1', 'res2'])
invalid = []
input_type = 'H'  # option: 'H' "other"


def neighbour(atoms, set_in):
    neighb = []
    for indx in set_in:
        atom = atoms[indx]
        bonds = atom.GetBonds()
        for bond in bonds:
            idxBegin = bond.GetBeginAtomIdx()
            idxEnd = bond.GetEndAtomIdx()
            neighb.append(idxBegin)
            neighb.append(idxEnd)
    # print(neighb)
    neighb = list(set(neighb))
    # print(neighb)
    H_id = ''
    for idx in neighb:
        atom = atoms[idx]
        if atom.GetSymbol() == 'H':
            print(idx)
            H_id = idx
            break
    if H_id == '':
        print("No bonded H was found, please check the input!")
    return H_id


for idx, col in dataset.iterrows():
    print(str(idx))
    try:
        smi = col['smi']
        # print(smi)
        atm1 = int(col['atm1'])
        atm2 = int(col['atm2'])
        # print(atm1)
        # print(atm2)
        mol = Chem.MolFromSmiles(smi)
        mol = rdkit.Chem.rdmolops.AddHs(mol)
        rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)
        AllChem.Compute2DCoords(mol)
        atoms = mol.GetAtoms()
        symbol1 = atoms[atm1].GetSymbol()
        symbol2 = atoms[atm2].GetSymbol()
        if input_type == 'H':
            H_id = neighbour(atoms, [atm1, atm2])
            if H_id == '':
                print(str(col)+" is not correct! ")
            # else:
            #     print("H_id= %s" % (H_id))
        else:

            if col['type'] not in [symbol1+'-'+symbol2, symbol2+'-'+symbol1]:
                print(str(col)+" is not correct! ")
            try:
                bond = mol.GetBondBetweenAtoms(atm1, atm2)
                bd_type = bond.GetBondType()
            except:
                print(str(col)+" is not correct! ")
    except:
        "Bug!"
        continue
