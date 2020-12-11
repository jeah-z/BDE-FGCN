# -*- coding:utf-8 -*-
"""Sample training code
"""
import numpy as np
import pandas as pd
import argparse
import torch as th
import torch.nn as nn
from sch_qm import SchNetModel
# from mgcn import MGCNModel
# from mpnn import MPNNModel
from torch.utils.data import DataLoader
from Alchemy_dataset_qm import TencentAlchemyDataset, batcher


def eval(model="sch", device=th.device("cpu"), test_file='', saved_model='', output=''):
    print("test start")

    test_dataset = TencentAlchemyDataset()
    # test_dir = train_dir
    # test_file = dataset+"_valid.csv"
    test_dataset.mode = "Train"
    test_dataset.transform = None
    test_dataset.file_path = test_file
    test_dataset._load()

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=50,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=0,
    )

    if model == "sch_qm":
        model = SchNetModel(norm=False, output_dim=1)

    print(model)
    # if model.name in ["MGCN", "SchNet"]:
    #     model.set_mean_std(alchemy_dataset.mean, alchemy_dataset.std, device)
    model.load_state_dict(
        th.load(saved_model))

    model.to(device)
    model.train()

    def print_res(label, res, op):
        size = len(res)
        for i in range(size):
            line = "%s,%s\n" % (label[i][0], res[i][0])
            op.writelines(line)

    w_loss, w_mae = 0, 0
    res_op = open(output, 'w')
    for idx, batch in enumerate(test_loader):
        batch.graph.to(device)
        batch.label = batch.label.to(device)
        batch.graph0 = batch.graph0.to(device)
        batch.graph1 = batch.graph1.to(device)
        batch.graph2 = batch.graph2.to(device)
        batch.graph3 = batch.graph3.to(device)
        batch.graph4 = batch.graph4.to(device)
        batch.graph5 = batch.graph5.to(device)
        batch.graph6 = batch.graph6.to(device)
        batch.graph7 = batch.graph7.to(device)
        batch.feature = batch.feature.to(device)

        res = model(batch.graph, batch.graph0, batch.graph1, batch.graph2,
                    batch.graph3, batch.graph4, batch.graph5, batch.graph6, batch.graph7, batch.feature)
        l = batch.label.cpu().detach().numpy()
        r = res.cpu().detach().numpy()
        print_res(l, r, res_op)
    res_op.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-M",
                        "--model",
                        help="model name (sch_qm)",
                        default="sch_qm")
    parser.add_argument(
        "--output", help="path to save the evalidation results", default='')
    parser.add_argument("--test_file", help="dataset to test", default="")
    parser.add_argument(
        "--saved_model", help="path of saved_model", default="")
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    assert args.model in ["sch_qm"]
    # dataset_split("delaney.csv")
    eval(model=args.model, device=device,
         test_file=args.test_file, saved_model=args.saved_model, output=args.output)
