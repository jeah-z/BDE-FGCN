import numpy as np
import pandas as pd
import argparse
from rdkit import Chem
import rdkit
from rdkit.Chem import AllChem


validation_split = .1
shuffle_dataset = True
random_seed = 42
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset to train")
args = parser.parse_args()
dataset = args.dataset


input_csv = pd.read_csv(dataset + ".csv", skiprows=0,
                      names=['SMILES', 'id1', 'id2', 'exp_val','reserv1','reserv2'])
input_csv=input_csv[['SMILES', 'id1', 'id2', 'exp_val']]
dataset_size = len(input_csv)
invalid_id = []
for i in range(dataset_size):
    smi = input_csv.loc[i]['SMILES']
    try:
        mol = Chem.MolFromSmiles(smi)
        AllChem.Compute2DCoords(mol)

    except:
        print(smi + "was not valid SMILES\n")
        invalid_id.append(i)
input_csv.drop(labels=invalid_id, axis=0)

#    Split the compounds into fragments
def canonicalize_smiles(smiles):
    """ Return a consistent SMILES representation for the given molecule """
    mol = rdkit.Chem.MolFromSmiles(smiles)
    return rdkit.Chem.MolToSmiles(mol)
def fragment(smiles, ids):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    mol = rdkit.Chem.rdmolops.AddHs(mol)
    rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)
    mh = rdkit.Chem.RWMol(mol)
    a1 = int(ids[0])
    a2 = int(ids[1])
    mh.RemoveBond(a1, a2)

    mh.GetAtomWithIdx(a1).SetNoImplicit(True)
    mh.GetAtomWithIdx(a2).SetNoImplicit(True)

    # Call SanitizeMol to update radicals
    rdkit.Chem.SanitizeMol(mh)

    # Convert the two molecules into a SMILES string
    fragmented_smiles = rdkit.Chem.MolToSmiles(mh)
    print(fragmented_smiles.split('.'))
    # Split fragment and canonicalize
    if len(fragmented_smiles.split('.'))==1:
        frag1 = fragmented_smiles.split('.')[0]
        frag2 = frag1
    else:
        frag1, frag2 = sorted(fragmented_smiles.split('.'))
    frag1 = canonicalize_smiles(frag1)
    frag2 = canonicalize_smiles(frag2)
    print("Frag1=%s; \t Frag2=%s"%(frag1, frag2))
    return frag1, frag2
#fragment_op = open(dataset+'_fragment')
fragments = []
for i in range(len(input_csv)):
    item = input_csv.loc[i]
    smi = item['SMILES']
    id1 = item['id1']
    id2 = item['id2']
    fragments.append([smi, id1, id2, fragment(smi, [id1, id2])])
fragment_pd = pd.DataFrame(fragments, columns=['SMI', 'id1','id2','Frags'])
fragment_pd.to_csv(dataset+'_fragment.csv', index=None)


