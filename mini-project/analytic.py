import time
import pickle
import heapq
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from bayes_cnn import train, KerasMNISTCNN

class FrozenBasisNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv1  = model.conv1
        self.conv2  = model.conv2
        self.drop1  = model.drop1
        self.dense1 = model.dense1

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = self.dense1(x)
        x = F.relu(x)
        return x

def Phi_Y(frozen_nn, loader, device, D):
    Phi_list = []
    Y_list = []
    frozen_nn.eval()
    with torch.inference_mode():
        for data, target in loader:
            data = data.to(device)
            Phi_list.append(frozen_nn(data)) # varphi(x_n)
            one_hot = F.one_hot(target.to(device), num_classes=D).float()
            Y_list.append(one_hot) # Y_n
    Phi = torch.cat(Phi_list, dim=0)
    Y = torch.cat(Y_list, dim=0)
    return Phi, Y

def inference(Phi, Y, sigma2, s2):
    K = Phi.shape[1]
    I_K = torch.eye(K, device=Phi.device)
    A = (1.0 / sigma2) * (Phi.T @ Phi) + (1.0 / s2) * I_K
    B = (1.0 / sigma2) * (Phi.T @ Y)
    Sigma_prime = torch.linalg.inv(A)
    M_prime = Sigma_prime @ B
    return M_prime, Sigma_prime

def inference_mfvi(Phi, Y, sigma2, s2):
    K = Phi.shape[1]
    I_K = torch.eye(K, device=Phi.device)
    A = (Phi.T @ Phi)+(sigma2/s2)*I_K 
    B = (Phi.T @ Y)
    M_star = torch.linalg.inv(A) @ B
    phi_sq_sum = torch.sum(Phi * Phi, dim=0)
    S_star = 1.0 / ((1.0 / s2) + (1.0 / sigma2) * phi_sq_sum)
    return M_star, S_star

def predictive_var_mfvi(varphi_star, S_star, sigma2, D=10):
    diag = (varphi_star * varphi_star) @ (S_star * S_star)
    return D * sigma2 + torch.diag(diag)

def predictive_var(varphi_star, cov, sigma2, D=10):
    return D*sigma2 + D * (varphi_star @ cov @ varphi_star.T) # is the trace

def acquire(frozen_nn, cov, sigma2, pool_idxs, train_idxs, train_mnist, num_acquire, acq_fn):
    heap = []
    pool_subset_idxs = torch.randperm(len(pool_idxs)).tolist()[:2000]
    pool = Subset(train_mnist, [pool_idxs[j] for j in pool_subset_idxs])
    pool_loader = DataLoader(dataset=pool)
    with torch.inference_mode():
        g_idx = 0
        for data, _ in pool_loader:
            data = data.to(next(frozen_nn.parameters()).device)
            varphi_xstar = frozen_nn(data)
            acq_vals = acq_fn(varphi_xstar, cov, sigma2).tolist()
            for j, acq_val in enumerate(acq_vals):
                idx = j + g_idx
                if len(heap) < num_acquire: heapq.heappush(heap, (acq_val, pool_subset_idxs[idx]))
                elif acq_val > heap[0][0]: heapq.heapreplace(heap, (acq_val, pool_subset_idxs[idx]))
            g_idx += len(acq_vals)
    assert len(heap) == num_acquire
    heap.sort(key=lambda x: x[1], reverse=True)
    for _, i in heap: train_idxs.append(pool_idxs.pop(i))
    assert set(train_idxs).isdisjoint(set(pool_idxs))

def test(frozen_nn, M_prime, device, test_loader, D):
    frozen_nn.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for data, target in test_loader:
            data, target= data.to(device), target.to(device)
            target = F.one_hot(target, num_classes=D).float()
            mu_star = frozen_nn(data) @ M_prime
            total += torch.sum((mu_star - target) ** 2)
            count += target.numel()
    rmse = torch.sqrt(total / count)
    return rmse

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="analytic", choices=["analytic", "mfvi"]) 
    args = parser.parse_args()

    # hyperparameters
    NUM_ACQUIRE, NUM_CLASSES, SEED, NUM_SEEDS, VERBOSE = 10, 10, 10, 1, False
    sigma2, s2 = 1.0, 1.0

    torch.manual_seed(SEED)
    device = "mps" if torch.mps.is_available() else "cpu"
    print(f"{device=}, {args.a=}, {NUM_ACQUIRE=}, {NUM_CLASSES=}, {SEED=}, {NUM_SEEDS=}, {sigma2=}, {s2=}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_mnist = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST("./data", train=False, transform=transform)
    test_loader = DataLoader(dataset=test_mnist, batch_size=128)

    plotting = {args.a: {"rmse": []}}

    start = time.time()
    for _ in range(NUM_SEEDS):
        # INIT BALANCED TRAINING SET
        train_idxs = []
        for i in range(NUM_CLASSES):
            class_examples = (train_mnist.targets == i).nonzero().flatten()
            nums = torch.randperm(len(class_examples)).tolist()[:2]
            train_idxs.extend(class_examples[nums].tolist())
        train_dataset = Subset(train_mnist, train_idxs)
        BATCH_SIZE = len(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        assert len(train_dataset) == 20

        # INIT VAL
        s_train_idxs = set(train_idxs)
        open_idxs = torch.tensor([i for i in range(len(train_mnist)) if i not in s_train_idxs])
        val_idxs = open_idxs[torch.randperm(len(open_idxs))[:100]].tolist()
        s_val_idxs = set(val_idxs)

        # INIT POOL
        pool_idxs = [i for i in range(len(train_mnist)) if i not in s_train_idxs and i not in s_val_idxs]

        # INIT MODEL
        model = KerasMNISTCNN().to(device)
        optimizer = optim.Adam(model.parameters())
        train(model, device, train_loader, optimizer)
        
        # INIT FROZEN
        frozen_nn = FrozenBasisNN(model).to(device)
        frozen_nn.eval()
        for p in frozen_nn.parameters(): p.requires_grad_(False) # freeze

        # INFERENCE
        Phi, Y = Phi_Y(frozen_nn, train_loader, device, NUM_CLASSES)
        if args.a == "analytic":
            M_prime, Sigma_prime = inference(Phi, Y, sigma2, s2)
            cov = Sigma_prime
            acq_fn = lambda phi, cov, s2_: predictive_var(phi, cov, s2_, D=NUM_CLASSES)
        else:
            M_prime, v = inference_mfvi(Phi, Y, sigma2, s2)
            cov = v
            acq_fn = lambda phi, cov, s2_: predictive_var_mfvi(phi, cov, s2_, D=NUM_CLASSES)

        rmse = test(frozen_nn, M_prime, device, test_loader, NUM_CLASSES)
        plotting[args.a]["rmse"].append(rmse)

        for s in tqdm(range(100)):
            acquire(frozen_nn=frozen_nn, cov=cov, sigma2=sigma2, pool_idxs=pool_idxs, train_idxs=train_idxs, train_mnist=train_mnist, num_acquire=NUM_ACQUIRE, acq_fn=acq_fn)

            assert set(train_idxs).isdisjoint(set(val_idxs))
            assert set(train_idxs).isdisjoint(set(pool_idxs))
            assert set(val_idxs).isdisjoint(set(pool_idxs))

            train_dataset = Subset(train_mnist, train_idxs)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            model = KerasMNISTCNN().to(device)
            optimizer = optim.Adam(model.parameters())
            train(model, device, train_loader, optimizer)
            
            # INIT FROZEN
            frozen_nn = FrozenBasisNN(model).to(device)
            frozen_nn.eval()
            for p in frozen_nn.parameters(): p.requires_grad_(False) # freeze

            Phi, Y = Phi_Y(frozen_nn, train_loader, device, NUM_CLASSES)
            if args.a == "analytic":
                M_prime, Sigma_prime = inference(Phi, Y, sigma2, s2)
                cov = Sigma_prime
            else:
                M_prime, v = inference_mfvi(Phi, Y, sigma2, s2)
                cov = v

            rmse = test(frozen_nn, M_prime, device, test_loader, NUM_CLASSES)

            plotting[args.a]["rmse"].append(rmse)
            N = len(train_dataset) - 20
            if VERBOSE: print(f"{rmse=:.6f}, {N=}")
    end = time.time()
    final_rmse = sum(plotting[args.a]["rmse"][-NUM_SEEDS:]) / float(NUM_SEEDS)
    print(f"SUMMARY {args.a=}: final_mean_rmse={final_rmse:.6f}, TIME:{(end-start)/60}m")
    assert len(plotting[args.a]["rmse"]) == 101 * NUM_SEEDS
    filename = f"frozen_{args.a}_new_s{SEED}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(plotting, f)
