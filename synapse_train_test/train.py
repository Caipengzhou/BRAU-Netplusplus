import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] ='1'
import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath
import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.bra_unet import BRAUnet
from trainer import trainer_synapse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="/home/cqut/Data/medical_seg_data/Synapse_zheng/train_npz",
    help="root dir for train data",
)
parser.add_argument(
    "--test_path",
    type=str,
    default="/home/cqut/Data/medical_seg_data/Synapse_zheng/test_vol_h5",
    help="root dir for test data",
)
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse", help="list dir")
parser.add_argument("--num_classes", type=int, default=9, help="output channel of network")
parser.add_argument("--output_dir", type=str, default="./save_models", help="output dir")
parser.add_argument("--max_iterations", type=int, default=90000, help="maximum epoch number to train")
parser.add_argument("--max_epochs", type=int, default=400, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument("--num_workers", type=int, default=8, help="num_workers")
parser.add_argument("--eval_interval", type=int, default=20, help="eval_interval")
parser.add_argument("--model_name", type=str, default="synapse", help="model_name")
parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
parser.add_argument("--deterministic", type=int, default=1, help="whether to use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation network base learning rate")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
parser.add_argument("--z_spacing", type=int, default=1, help="z_spacing")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--zip", action="store_true", help="use zipped dataset instead of folder dataset")
parser.add_argument(
    "--cache-mode",
    type=str,
    default="part",
    choices=["no", "full", "part"],
    help="no: no cache, "
    "full: cache all data, "
    "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
)
parser.add_argument("--resume", help="resume from checkpoint")
parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
parser.add_argument(
    "--use-checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory"
)
parser.add_argument(
    "--amp-opt-level",
    type=str,
    default="O1",
    choices=["O0", "O1", "O2"],
    help="mixed precision opt level, if O0, no amp is used",
)
parser.add_argument("--tag", help="tag of experiment")
parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
parser.add_argument("--throughput", action="store_true", help="Test throughput only")
parser.add_argument(
    "--module", help="The module that you want to load as the network, e.g. networks.DAEFormer.DAEFormer"
)

args = parser.parse_args()



if __name__ == "__main__":
    # setting device on GPU if available, else CPU
    # transformer = locate(args.module)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        "Synapse": {
            "root_path": args.root_path,
            "list_dir": args.list_dir,
            "num_classes": 9,
        },
    }
    if args.batch_size != 24 and args.batch_size % 5 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.root_path = dataset_config[dataset_name]["root_path"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    net = BRAUnet(img_size=224,in_chans=3, num_classes=9, n_win=7).cuda(0)
    net.load_from()
    trainer = {
        "Synapse": trainer_synapse,
    }
    trainer[dataset_name](args, net, args.output_dir)
