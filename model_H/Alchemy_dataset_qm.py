# -*- coding:utf-8 -*-
"""Example dataloader of Tencent Alchemy Dataset
https://alchemy.tencent.com/
"""
import os
import zipfile
import os.path as osp
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem import AllChem
import dgl
from dgl.data.utils import download
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pathlib
import pandas as pd
import numpy as np
import rdkit
from functools import partial
#from multiprocessing import Pool
from pathos.multiprocessing import Pool
from tqdm.auto import tqdm


class AlchemyBatcher:
    def __init__(self, graph=None, graph0=None, graph1=None, graph2=None, graph3=None, graph4=None, graph5=None, graph6=None, graph7=None, label=None, feature=None):
        self.graph = graph
        self.graph0 = graph0
        self.graph1 = graph1
        self.graph2 = graph2
        self.graph3 = graph3
        self.graph4 = graph4
        self.graph5 = graph5
        self.graph6 = graph6
        self.graph7 = graph7
        self.label = label
        self.feature = feature


def batcher():
    def batcher_dev(batch):
        graphs, graph0s, graph1s, graph2s, graph3s, graph4s, graph5s, graph6s, graph7s, labels, features = zip(
            *batch)
        batch_graphs = dgl.batch(graphs)
        batch_graph0s = dgl.batch(graph0s)
        batch_graph1s = dgl.batch(graph1s)
        batch_graph2s = dgl.batch(graph2s)
        batch_graph3s = dgl.batch(graph3s)
        batch_graph4s = dgl.batch(graph4s)
        batch_graph5s = dgl.batch(graph5s)
        batch_graph6s = dgl.batch(graph6s)
        batch_graph7s = dgl.batch(graph7s)
        labels = torch.stack(labels, 0)
        features = torch.stack(features, 0)
        return AlchemyBatcher(graph=batch_graphs, graph0=batch_graph0s, graph1=batch_graph1s, graph2=batch_graph2s, graph3=batch_graph3s, graph4=batch_graph4s, graph5=batch_graph5s, graph6=batch_graph6s, graph7=batch_graph7s, label=labels, feature=features)

    return batcher_dev


class TencentAlchemyDataset(Dataset):
    file_path = ""

    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    def alchemy_nodes(self, mol, set):
        """Featurization for all atoms in a molecule. The atom indices
        will be preserved.

        Args:
            mol : rdkit.Chem.rdchem.Mol
              RDKit molecule object
        Returns
            atom_feats_dict : dict
              Dictionary for atom features
        """
        atom_feats_dict = defaultdict(list)
        is_donor = defaultdict(int)
        is_acceptor = defaultdict(int)

        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        mol_feats = mol_featurizer.GetFeaturesForMol(mol)
        mol_conformers = mol.GetConformers()
        assert len(mol_conformers) == 1
        geom = mol_conformers[0].GetPositions()

        for i in range(len(mol_feats)):
            if mol_feats[i].GetFamily() == 'Donor':
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_donor[u] = 1
            elif mol_feats[i].GetFamily() == 'Acceptor':
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_acceptor[u] = 1

        num_atoms = mol.GetNumAtoms()
        num_set = len(set)
        for u in set:
            atom = mol.GetAtomWithIdx(u)
            symbol = atom.GetSymbol()
            atom_type = atom.GetAtomicNum()
            aromatic = atom.GetIsAromatic()
            hybridization = atom.GetHybridization()
            num_h = atom.GetTotalNumHs()
            atom_feats_dict['pos'].append(torch.FloatTensor(geom[u]))
            atom_feats_dict['node_type'].append(atom_type)

            h_u = []
            h_u += [
                int(symbol == x) for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Si', 'Br']
            ]
            h_u.append(atom_type)
            h_u.append(is_acceptor[u])
            h_u.append(is_donor[u])
            h_u.append(int(aromatic))
            h_u += [
                int(hybridization == x)
                for x in (Chem.rdchem.HybridizationType.SP,
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3)
            ]
            h_u.append(num_h)
            atom_feats_dict['n_feat'].append(torch.FloatTensor(h_u))

        atom_feats_dict['n_feat'] = torch.stack(atom_feats_dict['n_feat'],
                                                dim=0)
        atom_feats_dict['pos'] = torch.stack(atom_feats_dict['pos'], dim=0)
        atom_feats_dict['node_type'] = torch.LongTensor(
            atom_feats_dict['node_type'])
        return atom_feats_dict

    def alchemy_edges(self, mol, set, self_loop=True):
        """Featurization for all bonds in a molecule. The bond indices
        will be preserved.

        Args:
          mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object

        Returns
          bond_feats_dict : dict
              Dictionary for bond features
        """
        bond_feats_dict = defaultdict(list)

        mol_conformers = mol.GetConformers()
        assert len(mol_conformers) == 1
        geom = mol_conformers[0].GetPositions()

        num_atoms = mol.GetNumAtoms()
        num_set = len(set)
        for u in set:
            for v in set:
                if u == v and not self_loop:
                    continue

                e_uv = mol.GetBondBetweenAtoms(u, v)
                if e_uv is None:
                    bond_type = None
                else:
                    bond_type = e_uv.GetBondType()
                bond_feats_dict['e_feat'].append([
                    float(bond_type == x)
                    for x in (Chem.rdchem.BondType.SINGLE,
                              Chem.rdchem.BondType.DOUBLE,
                              Chem.rdchem.BondType.TRIPLE,
                              Chem.rdchem.BondType.AROMATIC, None)
                ])
                bond_feats_dict['distance'].append(
                    np.linalg.norm(geom[u] - geom[v]))

        bond_feats_dict['e_feat'] = torch.FloatTensor(
            bond_feats_dict['e_feat'])
        bond_feats_dict['distance'] = torch.FloatTensor(
            bond_feats_dict['distance']).reshape(-1, 1)

        return bond_feats_dict

    def atom_mask(self, mol, ids):
        num_atoms = mol.GetNumAtoms()
        mask = [[0]]*num_atoms
        for id in ids:
            mask[id] = [1]
        return mask

    def h_bonded(self, atoms, set_in):
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
            if atom.GetAtomicNum() == 1:
                # print(idx)
                H_id = idx
                # print("H_id= %s" % (H_id))
                break
        if H_id == '':
            print("No bonded H was found, please check the input!")
        return H_id

    def feats_cal(self, mol, ids):
        features = []
        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        atoms = mol.GetAtoms()
        # Calculation of the properties.
        Chem.rdPartialCharges.ComputeGasteigerCharges(
            mol, throwOnParamFailure=True)
        (CrippenlogPs, CrippenMRs) = zip(
            *(Chem.rdMolDescriptors._CalcCrippenContribs(mol)))
        TPSAs = Chem.rdMolDescriptors._CalcTPSAContribs(mol)
        (LaASAs, x) = Chem.rdMolDescriptors._CalcLabuteASAContribs(mol)

        # Both bonded atomic features

        for idxA in ids:
            AtomicNum = atoms[idxA].GetAtomicNum()  # 1.
            symbol = atoms[idxA].GetSymbol()      # 2
            # 3.
            GasteigerCharge = atoms[idxA].GetDoubleProp("_GasteigerCharge")
            TotalDegree = atoms[idxA].GetTotalDegree()  # 4.
            MinRingSize = [0]*6
            for n in range(3, 8):
                if(atoms[idxA].IsInRingSize(n)):
                    MinRingSize[n-3] = 1  # 5
                    break
            CrippenlogP = CrippenlogPs[idxA]  # 6
            CrippenMR = CrippenMRs[idxA]  # 7
            TPSA = TPSAs[idxA]  # 8
            LaASA = LaASAs[idxA]  # 9
            aromatic = atoms[idxA].GetIsAromatic()  # 10
            hybridization = atoms[idxA].GetHybridization()  # 11
            num_h = atoms[idxA].GetTotalNumHs()  # 12
            #  appending features to single list
            features.append(AtomicNum)  # 1
            features += [
                int(symbol == x) for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Si', 'Br']
            ]  # 2
            features.append(GasteigerCharge)  # 3
            features.append(TotalDegree)  # 4
            features += MinRingSize  # 5
            features.append(CrippenlogP)  # 6
            features.append(CrippenMR)  # 7
            features.append(TPSA)  # 8
            features.append(LaASA)  # 9
            features += [
                int(hybridization == x)
                for x in (Chem.rdchem.HybridizationType.SP,
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3)
            ]  # 10
            features.append(int(aromatic))  # 11 aromatic
            features.append(num_h)  # 12 num_h
        # Bond features
        bond = mol.GetBondBetweenAtoms(ids[0], ids[1])
        # print("id1=%s,id2=%s" % (ids[0], ids[1]))
        bond_type = bond.GetBondType()  # b 1

        mol_conformers = mol.GetConformers()
        geom = mol_conformers[0].GetPositions()
        distance = np.linalg.norm(geom[ids[1]] - geom[ids[0]])  # b 2

        features += [float(bond_type == x)
                     for x in (Chem.rdchem.BondType.SINGLE,
                               Chem.rdchem.BondType.DOUBLE,
                               Chem.rdchem.BondType.TRIPLE,
                               Chem.rdchem.BondType.AROMATIC, None)
                     ]
        features.append(distance)
        return features

    def canonicalize_smiles(self, smiles):
        """ Return a consistent SMILES representation for the given molecule """
        mol = rdkit.Chem.MolFromSmiles(smiles)
        return rdkit.Chem.MolToSmiles(mol)

    def fragment(self, smiles, ids):
        def neighb_set(atoms, set_in):
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
            return neighb

        def frag_cal(mol, set0):
            mh = rdkit.Chem.RWMol(mol)
            a1 = int(set0[0])
            a2 = int(set0[1])
            mh.RemoveBond(a1, a2)
            mh.GetAtomWithIdx(a1).SetNoImplicit(True)
            mh.GetAtomWithIdx(a2).SetNoImplicit(True)

            # Call SanitizeMol to update radicals
            rdkit.Chem.SanitizeMol(mh)
            # Convert the two molecules into a SMILES string
            fragmented_smiles = rdkit.Chem.MolToSmiles(mh)
            print(fragmented_smiles.split('.'))
            # Split fragment and canonicalize
            if len(fragmented_smiles.split('.')) == 1:
                frag1 = fragmented_smiles.split('.')[0]
                frag2 = frag1
            else:
                frag1, frag2 = sorted(fragmented_smiles.split('.'))
            frag1 = self.canonicalize_smiles(frag1)
            frag2 = self.canonicalize_smiles(frag2)
            print("Frag1=%s; \t Frag2=%s" % (frag1, frag2))
            return frag1, frag2

        mol = rdkit.Chem.MolFromSmiles(smiles)
        mol = rdkit.Chem.rdmolops.AddHs(mol)
        rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)
        atoms = mol.GetAtoms()
        a1 = int(ids[0])
        a2 = int(ids[1])
        set0 = [a1, a2]
        set1 = neighb_set(atoms, set0)
        set2 = neighb_set(atoms, set1)
        set3 = neighb_set(atoms, set2)
        set4 = neighb_set(atoms, set3)
        set5 = neighb_set(atoms, set4)
        set6 = neighb_set(atoms, set5)
        set7 = neighb_set(atoms, set6)
        return set0, set1, set2, set3, set4, set5, set6, set7

    def smi_graph(self, frag, set=[], self_loop=False):

        mol = rdkit.Chem.MolFromSmiles(frag)
        mol = rdkit.Chem.rdmolops.AddHs(mol)
        rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)
        AllChem.Compute2DCoords(mol)
        g = dgl.DGLGraph()
        # add nodes
        num_atoms = mol.GetNumAtoms()
        if set == []:
            set = list(range(num_atoms))
        atom_feats = self.alchemy_nodes(mol, set)
        num_set = len(set)
        g.add_nodes(num=num_set, data=atom_feats)
        if self_loop:
            g.add_edges(
                [i for i in range(num_set) for j in range(num_set)],
                [j for i in range(num_set) for j in range(num_set)])
        else:
            g.add_edges(
                [i for i in range(num_set) for j in range(num_set - 1)], [
                    j for i in range(num_set)
                    for j in range(num_set) if i != j
                ])

        bond_feats = self.alchemy_edges(mol, set, self_loop)
        g.edata.update(bond_feats)
        return g

    def smi_to_dgl(self, smi_id, self_loop=False):
        """
        Read sdf file and convert to dgl_graph
        Args:
            sdf_file: path of sdf file
            self_loop: Whetaher to add self loop
        Returns:
            g: DGLGraph
            l: related labels
        """
        try:

            smi = smi_id[0]
            print(f'Processing {smi}:')
            id1 = smi_id[1]
            id2 = smi_id[2]
            exp_val = smi_id[3]
            # smi = self.target.loc[index]['SMILES']
            # print(smi)
            # id1 = int(self.target.loc[index]['id1'])
            # id2 = int(self.target.loc[index]['id2'])

            mol = rdkit.Chem.MolFromSmiles(smi)
            mol = rdkit.Chem.rdmolops.AddHs(mol)
            rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)
            AllChem.Compute2DCoords(mol)
            atoms = mol.GetAtoms()
            ids = [id1, id2]
            # print("ids= %s" % (str(ids)))
            if ids[0] == ids[1]:
                ids[1] = self.h_bonded(atoms, ids)
            # print("ids= %s" % (str(ids)))
            set0, set1, set2, set3, set4, set5, set6, set7 = self.fragment(
                smi, ids)
            g = self.smi_graph(smi)
            g0 = self.smi_graph(smi, set0)
            g1 = self.smi_graph(smi, set1)
            g2 = self.smi_graph(smi, set2)
            g3 = self.smi_graph(smi, set3)
            g4 = self.smi_graph(smi, set4)
            g5 = self.smi_graph(smi, set5)
            g6 = self.smi_graph(smi, set6)
            g7 = self.smi_graph(smi, set7)

            features = self.feats_cal(mol, ids)
            # print(len(features))
            # for val/test set, labels are molecule ID
            features = torch.FloatTensor(features)
            # exp_val = float(self.target.loc[index]['exp_val'])
            l = torch.FloatTensor([exp_val])

            # mask = self.atom_mask(mol, ids)
            # mask = ids
            # mask = torch.IntTensor(mask)
            # features = self.feats_cal(mol, ids)
            # print(len(features))
            # features = torch.FloatTensor(features)
            # l = torch.FloatTensor(self.target.loc[int(sdf_file.stem)].tolist()) \
            #     if self.mode == 'dev' else torch.LongTensor([int(sdf_file.stem)])
            print(f'Processing {smi} : finished!')
            # result_list = [g, g0, g1, g2, g3, g4, g5, g6, g7, l, features]
            return (g, g0, g1, g2, g3, g4, g5, g6, g7, l, features)
        except:
            print(
                f'Processing {smi} {id1} {id2} {exp_val}: error encountered!')
            return None

    def __init__(self, mode='Train', transform=None):
        assert mode in ['Train', 'valid',
                        'Test'], "mode should be dev/valid/test"
        self.mode = mode
        self.transform = transform
        self.file_path = './'
        # self.file_dir = pathlib.Path('./Alchemy_data', mode)
        # self.zip_file_path = pathlib.Path('./Alchemy_data', '%s.zip' % mode)
        # download(_urls['Alchemy'] + "%s.zip" % mode,
        #          path=str(self.zip_file_path))
        # if not os.path.exists(str(self.file_dir)):
        #     archive = zipfile.ZipFile(self.zip_file_path)
        #     archive.extractall('./Alchemy_data')
        #     archive.close()

        # self._load()
    def read_file(file_path):
        """
        Reads a SMILES file.
        """
        with open(file_path, "r") as file:
            return [smi.rstrip().split() for smi in file]

    def _load(self):

        def h_bonded(atoms, set_in):
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
                if atom.GetAtomicNum() == 1:
                    # print(idx)
                    H_id = idx
                    # print("H_id= %s" % (H_id))
                    break
            if H_id == '':
                print("No bonded H was found, please check the input!")
            return H_id

        def alchemy_edges(mol, set, self_loop=True):
            """Featurization for all bonds in a molecule. The bond indices
            will be preserved.

            Args:
            mol : rdkit.Chem.rdchem.Mol
                RDKit molecule object

            Returns
            bond_feats_dict : dict
                Dictionary for bond features
            """
            bond_feats_dict = defaultdict(list)

            mol_conformers = mol.GetConformers()
            assert len(mol_conformers) == 1
            geom = mol_conformers[0].GetPositions()

            num_atoms = mol.GetNumAtoms()
            num_set = len(set)
            for u in set:
                for v in set:
                    if u == v and not self_loop:
                        continue

                    e_uv = mol.GetBondBetweenAtoms(u, v)
                    if e_uv is None:
                        bond_type = None
                    else:
                        bond_type = e_uv.GetBondType()
                    bond_feats_dict['e_feat'].append([
                        float(bond_type == x)
                        for x in (Chem.rdchem.BondType.SINGLE,
                                  Chem.rdchem.BondType.DOUBLE,
                                  Chem.rdchem.BondType.TRIPLE,
                                  Chem.rdchem.BondType.AROMATIC, None)
                    ])
                    bond_feats_dict['distance'].append(
                        np.linalg.norm(geom[u] - geom[v]))

            bond_feats_dict['e_feat'] = torch.FloatTensor(
                bond_feats_dict['e_feat'])
            bond_feats_dict['distance'] = torch.FloatTensor(
                bond_feats_dict['distance']).reshape(-1, 1)

            return bond_feats_dict

        def alchemy_nodes(mol, set):
            """Featurization for all atoms in a molecule. The atom indices
            will be preserved.

            Args:
                mol : rdkit.Chem.rdchem.Mol
                RDKit molecule object
            Returns
                atom_feats_dict : dict
                Dictionary for atom features
            """
            atom_feats_dict = defaultdict(list)
            is_donor = defaultdict(int)
            is_acceptor = defaultdict(int)

            fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
            mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
            mol_feats = mol_featurizer.GetFeaturesForMol(mol)
            mol_conformers = mol.GetConformers()
            assert len(mol_conformers) == 1
            geom = mol_conformers[0].GetPositions()

            for i in range(len(mol_feats)):
                if mol_feats[i].GetFamily() == 'Donor':
                    node_list = mol_feats[i].GetAtomIds()
                    for u in node_list:
                        is_donor[u] = 1
                elif mol_feats[i].GetFamily() == 'Acceptor':
                    node_list = mol_feats[i].GetAtomIds()
                    for u in node_list:
                        is_acceptor[u] = 1

            num_atoms = mol.GetNumAtoms()
            num_set = len(set)
            for u in set:
                atom = mol.GetAtomWithIdx(u)
                symbol = atom.GetSymbol()
                atom_type = atom.GetAtomicNum()
                aromatic = atom.GetIsAromatic()
                hybridization = atom.GetHybridization()
                num_h = atom.GetTotalNumHs()
                atom_feats_dict['pos'].append(torch.FloatTensor(geom[u]))
                atom_feats_dict['node_type'].append(atom_type)

                h_u = []
                h_u += [
                    int(symbol == x) for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Si', 'Br']
                ]
                h_u.append(atom_type)
                h_u.append(is_acceptor[u])
                h_u.append(is_donor[u])
                h_u.append(int(aromatic))
                h_u += [
                    int(hybridization == x)
                    for x in (Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3)
                ]
                h_u.append(num_h)
                atom_feats_dict['n_feat'].append(torch.FloatTensor(h_u))

            atom_feats_dict['n_feat'] = torch.stack(atom_feats_dict['n_feat'],
                                                    dim=0)
            atom_feats_dict['pos'] = torch.stack(atom_feats_dict['pos'], dim=0)
            atom_feats_dict['node_type'] = torch.LongTensor(
                atom_feats_dict['node_type'])
            return atom_feats_dict

        def canonicalize_smiles(smiles):
            """ Return a consistent SMILES representation for the given molecule """
            mol = rdkit.Chem.MolFromSmiles(smiles)
            return rdkit.Chem.MolToSmiles(mol)

        def fragment(smiles, ids):
            def neighb_set(atoms, set_in):
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
                return neighb

            def frag_cal(mol, set0):
                mh = rdkit.Chem.RWMol(mol)
                a1 = int(set0[0])
                a2 = int(set0[1])
                mh.RemoveBond(a1, a2)
                mh.GetAtomWithIdx(a1).SetNoImplicit(True)
                mh.GetAtomWithIdx(a2).SetNoImplicit(True)

                # Call SanitizeMol to update radicals
                rdkit.Chem.SanitizeMol(mh)
                # Convert the two molecules into a SMILES string
                fragmented_smiles = rdkit.Chem.MolToSmiles(mh)
                print(fragmented_smiles.split('.'))
                # Split fragment and canonicalize
                if len(fragmented_smiles.split('.')) == 1:
                    frag1 = fragmented_smiles.split('.')[0]
                    frag2 = frag1
                else:
                    frag1, frag2 = sorted(fragmented_smiles.split('.'))
                frag1 = canonicalize_smiles(frag1)
                frag2 = canonicalize_smiles(frag2)
                print("Frag1=%s; \t Frag2=%s" % (frag1, frag2))
                return frag1, frag2

            mol = rdkit.Chem.MolFromSmiles(smiles)
            mol = rdkit.Chem.rdmolops.AddHs(mol)
            rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)
            atoms = mol.GetAtoms()
            a1 = int(ids[0])
            a2 = int(ids[1])
            set0 = [a1, a2]
            set1 = neighb_set(atoms, set0)
            set2 = neighb_set(atoms, set1)
            set3 = neighb_set(atoms, set2)
            set4 = neighb_set(atoms, set3)
            set5 = neighb_set(atoms, set4)
            set6 = neighb_set(atoms, set5)
            set7 = neighb_set(atoms, set6)
            return set0, set1, set2, set3, set4, set5, set6, set7

        def smi_graph(frag, set=[], self_loop=False):
            mol = rdkit.Chem.MolFromSmiles(frag)
            mol = rdkit.Chem.rdmolops.AddHs(mol)
            rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)
            AllChem.Compute2DCoords(mol)
            g = dgl.DGLGraph()
            # add nodes
            num_atoms = mol.GetNumAtoms()
            if set == []:
                set = list(range(num_atoms))
            atom_feats = self.alchemy_nodes(mol, set)
            num_set = len(set)
            g.add_nodes(num=num_set, data=atom_feats)
            if self_loop:
                g.add_edges(
                    [i for i in range(num_set) for j in range(num_set)],
                    [j for i in range(num_set) for j in range(num_set)])
            else:
                g.add_edges(
                    [i for i in range(num_set) for j in range(num_set - 1)], [
                        j for i in range(num_set)
                        for j in range(num_set) if i != j
                    ])

            bond_feats = self.alchemy_edges(mol, set, self_loop)
            g.edata.update(bond_feats)
            return g

        def feats_cal(mol, ids):
            features = []
            fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
            chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(
                fdef_name)
            atoms = mol.GetAtoms()
            # Calculation of the properties.
            Chem.rdPartialCharges.ComputeGasteigerCharges(
                mol, throwOnParamFailure=True)
            (CrippenlogPs, CrippenMRs) = zip(
                *(Chem.rdMolDescriptors._CalcCrippenContribs(mol)))
            TPSAs = Chem.rdMolDescriptors._CalcTPSAContribs(mol)
            (LaASAs, x) = Chem.rdMolDescriptors._CalcLabuteASAContribs(mol)

            # Both bonded atomic features

            for idxA in ids:
                AtomicNum = atoms[idxA].GetAtomicNum()  # 1.
                symbol = atoms[idxA].GetSymbol()      # 2
                # 3.
                GasteigerCharge = atoms[idxA].GetDoubleProp("_GasteigerCharge")
                TotalDegree = atoms[idxA].GetTotalDegree()  # 4.
                MinRingSize = [0]*6
                for n in range(3, 8):
                    if(atoms[idxA].IsInRingSize(n)):
                        MinRingSize[n-3] = 1  # 5
                        break
                CrippenlogP = CrippenlogPs[idxA]  # 6
                CrippenMR = CrippenMRs[idxA]  # 7
                TPSA = TPSAs[idxA]  # 8
                LaASA = LaASAs[idxA]  # 9
                aromatic = atoms[idxA].GetIsAromatic()  # 10
                hybridization = atoms[idxA].GetHybridization()  # 11
                num_h = atoms[idxA].GetTotalNumHs()  # 12
                #  appending features to single list
                features.append(AtomicNum)  # 1
                features += [
                    int(symbol == x) for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Si', 'Br']
                ]  # 2
                features.append(GasteigerCharge)  # 3
                features.append(TotalDegree)  # 4
                features += MinRingSize  # 5
                features.append(CrippenlogP)  # 6
                features.append(CrippenMR)  # 7
                features.append(TPSA)  # 8
                features.append(LaASA)  # 9
                features += [
                    int(hybridization == x)
                    for x in (Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3)
                ]  # 10
                features.append(int(aromatic))  # 11 aromatic
                features.append(num_h)  # 12 num_h
            # Bond features
            bond = mol.GetBondBetweenAtoms(ids[0], ids[1])
            # print("id1=%s,id2=%s" % (ids[0], ids[1]))
            bond_type = bond.GetBondType()  # b 1

            mol_conformers = mol.GetConformers()
            geom = mol_conformers[0].GetPositions()
            distance = np.linalg.norm(geom[ids[1]] - geom[ids[0]])  # b 2

            features += [float(bond_type == x)
                         for x in (Chem.rdchem.BondType.SINGLE,
                                   Chem.rdchem.BondType.DOUBLE,
                                   Chem.rdchem.BondType.TRIPLE,
                                   Chem.rdchem.BondType.AROMATIC, None)
                         ]
            features.append(distance)
            return features

        # @wraps(lsmi_to_dgl)
        def lsmi_to_dgl(smi_id, self_loop=False):
            try:
                smi = smi_id[0]
                # print(f'Processing {smi}:')
                id1 = smi_id[1]
                id2 = smi_id[2]
                exp_val = smi_id[3]
                mol = rdkit.Chem.MolFromSmiles(smi)
                mol = rdkit.Chem.rdmolops.AddHs(mol)
                rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)
                AllChem.Compute2DCoords(mol)
                atoms = mol.GetAtoms()
                ids = [id1, id2]
                # print("ids= %s" % (str(ids)))
                if ids[0] == ids[1]:
                    ids[1] = h_bonded(atoms, ids)
                # print("ids= %s" % (str(ids)))
                set0, set1, set2, set3, set4, set5, set6, set7 = fragment(
                    smi, ids)
                g = smi_graph(smi)
                g0 = smi_graph(smi, set0)
                g1 = smi_graph(smi, set1)
                g2 = smi_graph(smi, set2)
                g3 = smi_graph(smi, set3)
                g4 = smi_graph(smi, set4)
                g5 = smi_graph(smi, set5)
                g6 = smi_graph(smi, set6)
                g7 = smi_graph(smi, set7)

                features = feats_cal(mol, ids)
                # print(len(features))
                # for val/test set, labels are molecule ID
                features = torch.FloatTensor(features)
                # exp_val = float(self.target.loc[index]['exp_val'])
                l = torch.FloatTensor([exp_val])
                # print(f'Processing {smi} : finished!')
                # result_list = [g, g0, g1, g2, g3, g4, g5, g6, g7, l, features]
                return (g, g0, g1, g2, g3, g4, g5, g6, g7, l, features)
            except:
                print(
                    f'Processing {smi} {id1} {id2} {exp_val}: error encountered!')
                return None
        if self.mode == 'Train':
            file_dir = pathlib.Path('./')
            # target_file = pathlib.Path(file_dir, "delaney.csv")
            target_file = self.file_path
            print('target_file= '+target_file)
            self.target = pd.read_csv(target_file, skiprows=1,
                                      names=['SMILES', 'id1', 'id2', 'exp_val', 'type', 'res1'])
            # self.target = read_file(target_file)
            # self.target = pd.DataFrame(target, columns=['smiles','id1',
            #     'exp_val','id2','shift','res'])

        self.graphs, self.graph0s,  self.graph1s, self.graph2s, self.graph3s, self.graph4s, self.graph5s, self.graph6s, self.graph7s, self.labels, self.features = [
        ], [], [], [], [], [], [], [], [], [], []
        # t_index = list(self.target.index)
        smi_id_list = []
        for index in self.target.index:
            # if index > 100:
            #     break
            smi = self.target.loc[index]['SMILES']
            id1 = int(self.target.loc[index]['id1'])
            id2 = int(self.target.loc[index]['id2'])
            exp_val = self.target.loc[index]['exp_val']
            smi_id_list.append([smi, id1, id2, exp_val])
        print('The input was loaded!')

        pool = Pool(20)

        # smi_to_dgl_p = partial(smi_to_dgl, self_loop=False)
        # results = []
        # for itm in smi_id_list:
        #     results.append(lsmi_to_dgl(itm))

        # for x in pool.imap(smi_to_dgl, smi_id_list):
        #     print('hi')
        #     results.append(x)
        # results = [x for x in tqdm(
        #     pool.imap(lsmi_to_dgl, smi_id_list), total=len(smi_id_list), miniters=100)]
        results = pool.map(lsmi_to_dgl, smi_id_list)
        pool.close()
        pool.join()
        # with Pool(20) as pool:
        #     smi_dgl_p = partial(self.smi_to_dgl, self_loop=False)
        #     results = [x for x in tqdm(pool.map(smi_dgl_p, smi_id_list), total=len(smi_id_list), miniters=100)
        #                if x is not None]
        for result in results:
            # result = self.smi_to_dgl(index)
            if result is None:
                continue
            self.graphs.append(result[0])
            self.graph0s.append(result[1])
            self.graph1s.append(result[2])
            self.graph2s.append(result[3])
            self.graph3s.append(result[4])
            self.graph4s.append(result[5])
            self.graph5s.append(result[6])
            self.graph6s.append(result[7])
            self.graph7s.append(result[8])
            self.labels.append(result[9])
            self.features.append(result[10])
        self.normalize()
        print(len(self.graphs), "loaded!")

    def normalize(self, mean=None, std=None):
        labels = np.array([i.numpy() for i in self.labels])
        if mean is None:
            mean = np.mean(labels, axis=0)
        if std is None:
            std = np.std(labels, axis=0)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g, g0, g1, g2, g3, g4, g5, g6, g7, l, fts = self.graphs[idx], self.graph0s[idx],  self.graph1s[idx], self.graph2s[
            idx], self.graph3s[idx], self.graph4s[idx], self.graph5s[idx], self.graph6s[idx], self.graph7s[idx],  self.labels[idx], self.features[idx]
        if self.transform:
            g = self.transform(g)
            g0 = self.transform(g0)
            g1 = self.transform(g1)
            g2 = self.transform(g2)
            g3 = self.transform(g3)
            g4 = self.transform(g4)
            g5 = self.transform(g5)
            g6 = self.transform(g6)
            g7 = self.transform(g7)

        return g, g0, g1, g2, g3, g4, g5, g6, g7, l, fts


if __name__ == '__main__':
    alchemy_dataset = TencentAlchemyDataset()
    # device = torch.device('cpu')
    # To speed up the training with multi-process data loader,
    # the num_workers could be set to > 1 to
    alchemy_loader = DataLoader(dataset=alchemy_dataset,
                                batch_size=20,
                                collate_fn=batcher(),
                                shuffle=False,
                                num_workers=0)

    for step, batch in enumerate(alchemy_loader):
        print("bs =", batch.graph.batch_size)
        print('feature size =', batch.graph.ndata['n_feat'].size())
        print('pos size =', batch.graph.ndata['pos'].size())
        print('edge feature size =', batch.graph.edata['e_feat'].size())
        print('edge distance size =', batch.graph.edata['distance'].size())
        print('label size=', batch.label.size())
        print(dgl.sum_nodes(batch.graph, 'n_feat').size())
        break
