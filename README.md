# BDE-FGCN : A fragments based model for predicting experimental bond dissociation energy

![image](https://github.com/jeah-z/BDE-FGCN/tree/main/images/Figure_S1_00.png)
Fig. 1 Architecture of BDE-FGCN model

# data format
Example:

SMILES,atom1,atom2,experimental_BDE,bond_type
C/C=C\CCC,2,3,-0.5240901979171866,C-C

This code adopts SMILES as input with the index of two atoms in the rdkit. Above atomic index usually is the same as the index in the SMILES string. 

Example:

SMILES,atom1,atom2,experimental_BDE,bond_type
C/C=C\CCC,2,2,-0.5240901979171866,C-H

If the the bond involve implicit hydrogen, users could input the heavy atom's index twice, this script will detect implicit hydrogen index automatically.



# to train the model 

```
python model_H/train_qm.py --model sch_qm --saved_model pretrained_model --epochs 2000 --train_file dataset/full_train.csv --test_file dataset/full_valid.csv
```

# to evals the model 

```
python model_H/eval.py --model sch_qm --saved_model pretrained_model/model_400 --output ./application-OH.txt --test_file Dataset/Application_OH_GCN.csv
```


# dependency

- rdkit
- dgl
- pytorch
- python==3.6
- numpy 
- pandas
- zipfile
- os
- pathlib
- tqdm
- pathos
- argparse

# related repository

This code was based on https://github.com/tencent-alchemy/Alchemy. If this script is of any help to you, please cite them.

- K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.
SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017) [link](http://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions)  
```
- @article{chen2019alchemy,
  title={Alchemy: A Quantum Chemistry Dataset for Benchmarking AI Models},
  author={Chen, Guangyong and Chen, Pengfei and Hsieh, Chang-Yu and Lee, Chee-Kong and Liao, Benben and Liao, Renjie and Liu, Weiwen and Qiu, Jiezhong and Sun, Qiming and Tang, Jie and Zemel, Richard and Zhang, Shengyu},
  journal={arXiv preprint arXiv:1906.09427},
  year={2019}
}
```