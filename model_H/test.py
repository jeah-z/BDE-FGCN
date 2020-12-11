import rdkit
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem import AllChem
# import dgl
# def print_atom(mol):
#     num_atoms = mol.GetNumAtoms()
#     bonds = mol.GetBonds()
#     num_bonds = mol.GetNumBonds()
#     atoms = mol.GetAtoms()
#     for i in range(num_atoms):
#         symbol = atoms[i].GetSymbol()
#         print("id=%s; symbol=%s "%(i, symbol))
#     for i in range(num_bonds):
#         bond = bonds[i]
#         bond_type = bond.GetBondType()
#         a1 = bond.GetBeginAtomIdx()
#         a2 = bond.GetEndAtomIdx()
#         print("bond_type = %s; atom1= %s; atom2 = %s; "%(bond_type, a1, a2))


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
    print(neighb)
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
# smi="CC(=O)C(C)C"
# id1=0
# id2=1
# mol= Chem.MolFromSmiles(smi)
# AllChem.Compute2DCoords(mol)
# mol = rdkit.Chem.rdmolops.AddHs(mol)
# atoms = mol.GetAtoms()
# print(smi)
# print_atom(mol)
# smi_cano =  rdkit.Chem.MolToSmiles(mol)
# mol_cano= Chem.MolFromSmiles(smi_cano)
# print(smi_cano)
# print_atom(mol_cano)
# set1=([0,1])
# set2=neighbour(atoms,set1)
# set3=neighbour(atoms,set2)


smi = "Cc1ccc(NN)cc1"
id1 = 0
id2 = 1
mol = Chem.MolFromSmiles(smi)
AllChem.Compute2DCoords(mol)
mol = rdkit.Chem.rdmolops.AddHs(mol)
rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)
atoms = mol.GetAtoms()
neighb_list = neighbour(atoms, [6])
