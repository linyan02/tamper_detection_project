import argparse
from train.train_resnet import train_resnet
from train.train_pscc import train_pscc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["pscc_net", "resnet18"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.model == "resnet18":
        train_resnet(args.epochs)
    else:
        train_pscc(args.epochs)