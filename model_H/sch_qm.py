# -*- coding:utf-8 -*-

import dgl
import torch as th
import torch.nn as nn
import numpy as np
from layers import AtomEmbedding, Interaction, ShiftSoftplus, RBFLayer


class SchNetModel(nn.Module):
    """
    SchNet Model from:
        Schütt, Kristof, et al.
        SchNet: A continuous-filter convolutional neural network
        for modeling quantum interactions. (NIPS'2017)
    """

    def __init__(self,
                 dim=64,
                 cutoff=5.0,
                 output_dim=1,
                 width=1,
                 n_conv=3,
                 norm=False,
                 atom_ref=None,
                 pre_train=None):
        """
        Args:
            dim: dimension of features
            output_dim: dimension of prediction
            cutoff: radius cutoff
            width: width in the RBF function
            n_conv: number of interaction layers
            atom_ref: used as the initial value of atom embeddings,
                      or set to None with random initialization
            norm: normalization
        """
        super().__init__()
        self.name = "SchNet"
        self._dim = dim
        self.cutoff = cutoff
        self.width = width
        self.n_conv = n_conv
        self.atom_ref = atom_ref
        self.norm = norm
        self.activation = ShiftSoftplus()

        if atom_ref is not None:
            self.e0 = AtomEmbedding(1, pre_train=atom_ref)
        if pre_train is None:
            self.embedding_layer = AtomEmbedding(dim)
        else:
            self.embedding_layer = AtomEmbedding(pre_train=pre_train)
        self.rbf_layer = RBFLayer(0, cutoff, width)
        self.conv_layers = nn.ModuleList(
            [Interaction(self.rbf_layer._fan_out, dim) for i in range(n_conv)])

        self.atom_dense_layer1 = nn.Linear(dim, 64)
        self.atom_dense_layer2 = nn.Linear(64, output_dim)
        self.fc = nn.Sequential(nn.Linear(69, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(64, 32),
                                nn.ReLU(),
                                nn.Linear(32, 1),
                                )

    def set_mean_std(self, mean, std, device="cpu"):
        self.mean_per_atom = th.tensor(mean, device=device)
        self.std_per_atom = th.tensor(std, device=device)

    def forward(self, g, g0,  g1, g2, g3, g4, g5, g6, g7, fts):
        def forward_g(g):
            """g is the DGL.graph"""

            self.embedding_layer(g)
            if self.atom_ref is not None:
                self.e0(g, "e0")
            self.rbf_layer(g)
            for idx in range(self.n_conv):
                self.conv_layers[idx](g)

            atom = self.atom_dense_layer1(g.ndata["node"])
            atom = self.activation(atom)
            res = self.atom_dense_layer2(atom)
            g.ndata["res"] = res

            if self.atom_ref is not None:
                g.ndata["res"] = g.ndata["res"] + g.ndata["e0"]

            if self.norm:
                g.ndata["res"] = g.ndata[
                    "res"] * self.std_per_atom + self.mean_per_atom
            # print('res before *mask %s'%(g.ndata["res"]))
            # print('mask =  %s'%(mask.squeeze(0)))
            res = dgl.sum_nodes(g, "res")
            return res
        # print('res =  %s'%(res))
        res = forward_g(g)
        res0 = forward_g(g0)
        res1 = forward_g(g1)
        res2 = forward_g(g2)
        res3 = forward_g(g3)
        res4 = forward_g(g4)
        res5 = forward_g(g5)
        res6 = forward_g(g6)
        res7 = forward_g(g7)

        res_np = res.cpu().detach().numpy()
        res0_np = res0.cpu().detach().numpy()
        res1_np = res1.cpu().detach().numpy()
        res2_np = res2.cpu().detach().numpy()
        res3_np = res3.cpu().detach().numpy()
        res4_np = res4.cpu().detach().numpy()
        res5_np = res5.cpu().detach().numpy()
        res6_np = res6.cpu().detach().numpy()
        res7_np = res7.cpu().detach().numpy()
        with open('graph_bit.txt', 'a') as graph_bit:
            for i in range(len(res_np)):
                graph_bit.write(
                    f'{res0_np[i][0]},{res1_np[i][0]},{res2_np[i][0]},{res3_np[i][0]},{res4_np[i][0]},{res5_np[i][0]},{res6_np[i][0]},{res7_np[i][0]},{res_np[i][0]}\n')
        print(
            f'{res}')

        dense_input = th.cat((res, res0), 1)
        dense_input = th.cat((dense_input, res1), 1)
        dense_input = th.cat((dense_input, res2), 1)
        dense_input = th.cat((dense_input, res3), 1)
        dense_input = th.cat((dense_input, res4), 1)
        dense_input = th.cat((dense_input, res5), 1)
        dense_input = th.cat((dense_input, res6), 1)
        dense_input = th.cat((dense_input, res7), 1)
        dense_input = th.cat((dense_input, fts), 1)
        # print('res after *mask %s'%(g.ndata["res"]))
        # res = dgl.sum_nodes(g, "res")
        # res_qm = th.cat((res, qm), 1)
        # print('res_qm =  %s'%(res_qm))
        # pred = self.activation(self.dense_layer1(dense_input))

        # pred = self.activation(self.dense_layer2(pred))
        # pred = self.activation(self.dense_layer3(pred))

        # pred = self.activation(self.dense_layer4(pred))
        # pred = self.dropout(pred)
        # pred = self.activation(self.dense_layer5(pred))
        # pred = self.relu(self.dense_layer4(pred))

        pred = self.fc(dense_input)

        # print('res after sum_nodes %s'%(res))
        return pred


if __name__ == "__main__":
    g = dgl.DGLGraph()
    g.add_nodes(2)
    g.add_edges([0, 0, 1, 1], [1, 0, 1, 0])
    g.edata["distance"] = th.tensor([1.0, 3.0, 2.0, 4.0]).reshape(-1, 1)
    g.ndata["node_type"] = th.LongTensor([1, 2])
    model = SchNetModel(dim=1)
    atom = model(g)
    print(atom)
