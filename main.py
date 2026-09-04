import argparse
import os
import random

import numpy as np
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_dir",
    type=str,
    default="Data/input/SWaT_Dataset_Attack_v0.csv",
)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--output_dir", type=str, default="./checkpoint/")
parser.add_argument("--name", type=str, default="SWaT")
parser.add_argument("--graph", type=str, default="None")
parser.add_argument("--model", type=str, default="MAF")
parser.add_argument("--n_blocks", type=int, default=1)
parser.add_argument("--n_components", type=int, default=1)
parser.add_argument("--hidden_size", type=int, default=32)
parser.add_argument("--n_hidden", type=int, default=1)
parser.add_argument("--input_size", type=int, default=1)
parser.add_argument("--batch_norm", action="store_true")
parser.add_argument("--train_split", type=float, default=0.6)
parser.add_argument("--stride_size", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=640)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--window_size", type=int, default=60)
parser.add_argument("--lr", type=float, default=5e-3)

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


import torch
from torch.nn.utils import clip_grad_value_
from models.GraphMoE import GraphMoE

args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")
print("Using device:", device)

if args.cuda:
    print("GPU:", torch.cuda.get_device_name(0))


for seed in range(15, 20):
    args.seed = seed
    print("\nCurrent arguments:")
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    print("Loading dataset")
    print("Dataset:", args.name)

    from Dataset import (
        load_smd_smap_msl,
        loader_PSM,
        loader_SWat,
        loader_WADI,
    )

    if args.name == "SWaT":
        train_loader, val_loader, test_loader, n_sensor = loader_SWat(
            args.data_dir,
            args.batch_size,
            args.window_size,
            args.stride_size,
            args.train_split,
        )
        print("train_loader:", train_loader)

    elif args.name == "Wadi":
        train_loader, val_loader, test_loader, n_sensor = loader_WADI(
            args.data_dir,
            args.batch_size,
            args.window_size,
            args.stride_size,
            args.train_split,
        )

    elif (
        args.name == "SMAP"
        or args.name == "MSL"
        or args.name.startswith("machine")
    ):
        train_loader, val_loader, test_loader, n_sensor = load_smd_smap_msl(
            args.name,
            args.batch_size,
            args.window_size,
            args.stride_size,
            args.train_split,
        )

    elif args.name == "PSM":
        train_loader, val_loader, test_loader, n_sensor = loader_PSM(
            args.name,
            args.batch_size,
            args.window_size,
            args.stride_size,
            args.train_split,
        )

    else:
        raise ValueError(f"Unsupported dataset name: {args.name}")

    model = GraphMoE(
        args.n_blocks,
        args.input_size,
        args.hidden_size,
        args.n_hidden,
        args.window_size,
        n_sensor,
        dropout=0.0,
        model=args.model,
        batch_norm=args.batch_norm,
    )

    model = model.to(device)

    save_path = os.path.join(args.output_dir, args.name)
    os.makedirs(save_path, exist_ok=True)

    roc_max = 0.0

    optimizer = torch.optim.Adam(
        [
            {
                "params": model.parameters(),
                "weight_decay": args.weight_decay,
            }
        ],
        lr=args.lr,
        weight_decay=0.0,
    )

    for epoch in range(80):
        print(f"\nEpoch: {epoch + 1}/80")

        loss_train = []

        model.train()

        for batch in train_loader:
            x, _, idx = batch
            x = x.to(device)
            optimizer.zero_grad()
            loss = -model(x).mean()
            loss.backward()
            clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()
            loss_train.append(loss.item())
        mean_train_loss = np.mean(loss_train)

        
        loss_test = []

        model.eval()

        with torch.no_grad():
            for batch in test_loader:
                x, _, idx = batch
                x = x.to(device)              
                log_prob_loss = -model(x)
                loss_test.append(
                    log_prob_loss.detach().cpu().numpy()
                )

        loss_test = np.concatenate(loss_test)     
        labels = np.asarray(
            test_loader.dataset.label[: len(loss_test)],
            dtype=int,
        )
        roc_test = roc_auc_score(labels, loss_test)

        
        if roc_test > roc_max:
            roc_max = roc_test
            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": args.seed,
                    "epoch": epoch + 1,
                    "roc_auc": roc_test,
                },
                os.path.join(save_path, "model.pth"),
            )
        print(
            #f"train loss: {mean_train_loss:.6f}, "
            f"ROC: {roc_max:.6f}"
        )