import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18,resnet34,resnet50
import time
import copy
import numpy as np
import random
from tqdm import trange

from utils.options import args_parser
from utils.sampling import noniid
from utils.dataset import load_data, LeNet5
from utils.test import test_img
from utils.byzantine_fl import GPU_krum,krum, trimmed_mean, fang, dummy_contrastive_aggregation
from utils.attack import compromised_clients, untargeted_attack
from src.aggregation import fedavg
from src.update import BenignUpdate, CompromisedUpdate


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    args.dataset = "CIFAR10"
    #args.num_classes = 14
    if args.dataset in ["CIFAR10", "MNIST", "FaMNIST","SVHN"]:
    # Change the package  [/home/syjiang/anaconda3/lib/python3.11/site-packages/torchvision/models/resnet.py] Line 197 3 ==> 1 in MNIST and FaMNIST
        args.num_classes = 10
    print("args.num_classes",args.num_classes)
    args.tsboard=True

    if args.tsboard:
        writer = SummaryWriter(f'runs/data')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.num_clients = 4          #change client
   # args.method = 'trimmed_mean'   #change method

    dataset_train, dataset_test, dataset_val = load_data(args)


    # early stopping hyperparameters
    cnt = 0
    check_acc = 0

    # sample users
    dict_users = noniid(dataset_train, args)


    if args.dataset in ["MNIST","pathmnist","pneumoniamnist","tissuemnist"] :
        net_glob = LeNet5().to(args.device)  #change model
    elif args.dataset in ["FaMNIST","chestmnist","retinamnist","organamnist"]:
        net_glob = resnet18(num_classes = args.num_classes).to(args.device)  #change model
    elif args.dataset in ["CIFAR10","dermamnist","breastmnist","organcmnist"]:
        net_glob = resnet34(num_classes = args.num_classes).to(args.device)  #change model
    elif args.dataset in ["SVHN","octmnist","bloodmnist","organsmnist"]:
        net_glob = resnet50(num_classes = args.num_classes).to(args.device)  #change model



    net_glob.train()

    # copy weights
    print(args.device)
    w_glob = net_glob.state_dict()

    if args.c_frac > 0:
        compromised_idxs = compromised_clients(args)
    else:
        compromised_idxs = []

    local_traintime=0
    for iter in trange(args.global_ep):
        w_locals = []
        selected_clients = max(int(args.frac * args.num_clients), 1)
        compromised_num = int(args.c_frac * selected_clients)
        idxs_users = np.random.choice(range(args.num_clients), selected_clients, replace=False)

        for idx in idxs_users:
            if idx in compromised_idxs:
                if args.p == "untarget":
                    w_locals.append(copy.deepcopy(untargeted_attack(net_glob.state_dict(), args)))
                else:
                    local = CompromisedUpdate(args = args, dataset = dataset_train, idxs = dict_users[idx])
                    w = local.train(net = copy.deepcopy(net_glob).to(args.device))
                    w_locals.append(copy.deepcopy(w))

            else:
                local = BenignUpdate(args = args, dataset = dataset_train, idxs = dict_users[idx])
                starttime = time.time()
                w = local.train(net = copy.deepcopy(net_glob).to(args.device))
                endtime = time.time()
                local_traintime+=endtime-starttime

                w_locals.append(copy.deepcopy(w))
        print("local train time", local_traintime)

        # update global weights
        if args.method == 'fedavg':
            w_glob = fedavg(w_locals)
        elif args.method == 'krum':
            #w_glob, _ = krum(w_locals, compromised_num, args)
            w_glob, _ = GPU_krum(w_locals, compromised_num, args)
        elif args.method == 'trimmed_mean':
            w_glob = trimmed_mean(w_locals, compromised_num, args)
        elif args.method == 'fang':
            w_glob = fang(w_locals, dataset_val, compromised_num, args)
        elif args.method == 'dca':
            w_glob = dummy_contrastive_aggregation(w_locals, compromised_num, copy.deepcopy(net_glob), args)
        else:
            exit('Error: unrecognized aggregation technique')

        # copy weight to net_glob

        net_glob.load_state_dict(w_glob)
        test_acc, test_loss = test_img(net_glob.to(args.device), dataset_test, args)

        args.debug=True
        if args.debug:
            print(f"Round: {iter}")
            print(f"Test accuracy: {test_acc}")
            print(f"Test loss: {test_loss}")
            print(f"Check accuracy: {check_acc}")
            print(f"patience: {cnt}")

        if check_acc == 0:
            check_acc = test_acc
        elif test_acc < check_acc + args.delta:
            cnt += 1
        else:
            check_acc = test_acc
            cnt = 0

        # early stopping
        if cnt == args.patience:
            print('Early stopped federated training!')
            break

        # tensorboard
        args.tsboard=True

        if args.tsboard:
            writer.add_scalar(f'testacc/{args.method}_{args.p}_cfrac_{args.c_frac}_alpha_{args.alpha}', test_acc, iter)
            writer.add_scalar(f'testloss/{args.method}_{args.p}_cfrac_{args.c_frac}_alpha_{args.alpha}', test_loss, iter)

    if args.tsboard:
        writer.close()
