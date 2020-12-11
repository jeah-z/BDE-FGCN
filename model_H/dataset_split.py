import numpy as np
import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem

normalization = 0
validation_split = .1
shuffle_dataset = True
random_seed = 42
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset to train")
args = parser.parse_args()
dataset = args.dataset


input_csv = pd.read_csv(dataset + ".csv", skiprows=1,
                        names=['SMILES', 'id1', 'id2', 'exp_val', 'type', 'reserv1'])
input_csv = input_csv[['SMILES', 'id1', 'id2', 'exp_val', 'type']]
dataset_size = len(input_csv)
print(f'The total dataset size: {dataset_size}')
invalid_id = []
small_id = []
for i in range(dataset_size):
    smi = input_csv.loc[i]['SMILES']
    try:
        mol = Chem.MolFromSmiles(smi)
        AllChem.Compute2DCoords(mol)
        atoms = mol.GetAtoms()
        natom = len(atoms)
        if natom <= 5:
            small_id.append(i)

    except:
        print(smi + "was not valid SMILES\n")
        invalid_id.append(i)

tmp_csv = input_csv.copy(deep=True)
train_csv = tmp_csv.iloc[small_id]
input_csv.drop(labels=invalid_id+small_id, axis=0, inplace=True)
tmp_index = list(range(len(input_csv)))
input_csv.index = tmp_index
# Creating data indices for training and validation splits:
dataset_size = len(input_csv)
print('dataset_size= %s' % (dataset_size))
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
# print(train_indices)

# Creating PT data samplers and loaders:
train_sampler = input_csv.loc[train_indices]
train_sampler = pd.concat([train_csv, train_sampler])
train_target_mean = train_sampler['exp_val'].mean()
train_target_std = train_sampler['exp_val'].std()
if normalization > 0:
    train_sampler['exp_val'] = (
        train_sampler['exp_val'] - train_target_mean) / train_target_std

valid_sampler = input_csv.loc[val_indices]
if normalization > 0:
    valid_sampler['exp_val'] = (
        valid_sampler['exp_val']-train_target_mean)/train_target_std

    # train_target_mean = train_sampler['NMR'].mean()
    # train_target_std = train_sampler['NMR'].std()
    # train_sampler['NMR'] = (
    #     train_sampler['NMR'] - train_target_mean) / train_target_std

    # valid_sampler['NMR'] = (valid_sampler['NMR']-train_target_mean)/train_target_std

    mean_file = open(dataset + '_mean_std.txt', 'w')
    # mean_file.writelines('train_parm_mean= %s\n' % (train_parm_mean))
    # mean_file.writelines('train_parm_std= %s\n' % (train_parm_std))
    mean_file.writelines('train_target_mean= %s\n' % (train_target_mean))
    mean_file.writelines('train_target_std= %s' % (train_target_std))

print(f'The total train dataset size: {len(train_sampler)}')
print(f'The total validation dataset size: {len(valid_sampler)}')
train_sampler.to_csv(dataset+"_train.csv", index=False)
valid_sampler.to_csv(dataset+"_valid.csv", index=False)


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
    a1 = ids[0]
    a2 = ids[1]
    mh.RemoveBond(a1, a2)

    mh.GetAtomWithIdx(a1).SetNoImplicit(True)
    mh.GetAtomWithIdx(a2).SetNoImplicit(True)

    # Call SanitizeMol to update radicals
    rdkit.Chem.SanitizeMol(mh)

    # Convert the two molecules into a SMILES string
    fragmented_smiles = rdkit.Chem.MolToSmiles(mh)

    # Split fragment and canonicalize
    frag1, frag2 = sorted(fragmented_smiles.split('.'))
    frag1 = canonicalize_smiles(frag1)
    frag2 = canonicalize_smiles(frag2)
    return [frag1, frag2]
