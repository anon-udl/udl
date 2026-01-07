import time
import pickle
import heapq
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm 

# https://github.com/Riashat/Active-Learning-Bayesian-Convolutional-Neural-Networks/blob/master/active_learning/mnist_N1000/Dropout_Uncertainty_Model_Averaging/Dropout_Model_Averaging_Bald_Q10_N1000.py    
# https://github.com/yaringal/acquisition_example/blob/master/acquisition_example.ipynb
class KerasMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 32, 4)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.dense1 = nn.LazyLinear(128)
        self.dense2 = nn.LazyLinear(10)

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
        x = self.drop2(x)
        x = self.dense2(x)
        x = F.softmax(x, dim=-1)
        return x

def mc_dropout(model, data):
    data_rep = data.repeat(T, 1, 1, 1) # broadcasts T steps (T, 1, 28, 28)
    preds = model(data_rep) # (B*T, C)
    B = data.shape[0]
    preds = preds.view(T, B, -1) # (T, B, C)
    res = torch.sum(preds, dim=0) / T # (B, C)
    return res, preds

def random_acquisition(pool):
    return torch.randperm(len(pool)).tolist()[:NUM_ACQUIRE]

def max_entropy(preds): # preds shape = (T, B, C)
    scaled_sum_phat_tc = torch.sum(preds, dim=0) / T # (B, C)
    return -torch.sum(scaled_sum_phat_tc*torch.log2(scaled_sum_phat_tc+1e-8),dim=-1) # (B)

def variation_ratio(preds):
    p = torch.sum(preds, dim=0) / T # (B, C)
    maxy = torch.max(p, dim=-1).values # (B)
    return 1.0 - maxy

def mean_std(preds):
    p = torch.sum(preds, dim=0) / T # (B, C)
    p2 = torch.sum(preds**2, dim=0) / T # (B, C)
    sigmas = torch.sqrt(p2-p**2) # (B, C)
    return torch.sum(sigmas, dim=-1) / NUM_CLASSES # (B)

def bald(preds):
    H = max_entropy(preds)
    sum_t = torch.sum(preds * torch.log2(preds+1e-8),dim=0) # TODO: what shape? 
    EH = torch.sum(sum_t, dim=-1) / T
    return H + EH # (B)

def margin_sampling(preds):
    m = preds.mean(dim=0)
    top2 = torch.topk(m, k=2, dim=-1).values # (B, 2)
    margin = top2[:,0]-top2[:,1] 
    return 1.0-margin # (B)

# computes acquisition statistic and adds NUM_ACQUIRE datapoints to training set
def acquire(acq_func, pool_idxs, train_idxs):
    heap = []
    pool_subset_idxs = torch.randperm(len(pool_idxs)).tolist()[:2000] # get subset of 2000
    pool = Subset(train_mnist, [pool_idxs[j] for j in pool_subset_idxs])
    if acq_func == random_acquisition: # add random idx to pool
        for idx in random_acquisition(pool): heap.append((pool_subset_idxs[idx], pool_subset_idxs[idx]))
    else:
        pool_loader = DataLoader(dataset=pool, batch_size=BATCH_SIZE)
        with torch.inference_mode():
            g_idx = 0
            for data, _ in pool_loader:
                # add idx via acquisition function
                data = data.to(device)
                _, preds = mc_dropout(model, data)
                acq_vals = acq_func(preds).tolist()
                # maintain heap of top NUM_ACQUIRE datapoints
                for j, acq_val in enumerate(acq_vals):
                    idx = j + g_idx
                    if len(heap) < NUM_ACQUIRE: heapq.heappush(heap, (acq_val, pool_subset_idxs[idx]))
                    elif acq_val > heap[0][0]: heapq.heapreplace(heap, (acq_val, pool_subset_idxs[idx]))
                g_idx += len(acq_vals)
    # add to training set
    assert len(heap) == NUM_ACQUIRE
    heap.sort(key=lambda x: x[1], reverse=True) # sort by idx to pop largest first
    for _, i in heap: train_idxs.append(pool_idxs.pop(i))
    assert set(train_idxs).isdisjoint(set(pool_idxs))
        
def train(model, device, train_loader, optimizer, NUM_EPOCHS=50):
    model.train()
    for _ in range(NUM_EPOCHS):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # loss function identical to keras categorical_crossentropy: https://www.tensorflow.org/api_docs/python/tf/keras/losses/categorical_crossentropy
            loss = F.nll_loss((output + 1e-8).log(), target)
            # c = 3.5
            # weight_decay = c / float(len(train_loader.dataset))
            # loss += weight_decay * model.dense1.weight.pow(2).sum() # mimics keras weight decay in first dense layer
            loss.backward()
            optimizer.step()

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.inference_mode():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = mc_dropout(model, data)
            correct += output.argmax(dim=-1).eq(target).sum().item()
    return correct

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="random_acquisition")
    args = parser.parse_args()
    if args.a == "all": 
        funcs =  [margin_sampling, random_acquisition, mean_std, max_entropy, variation_ratio, bald]
    else: 
        funcs = {"margin_sampling": margin_sampling, "random_acquisition": random_acquisition, "mean_std": mean_std, "max_entropy": max_entropy, "variation_ratio": variation_ratio, "bald": bald}
        funcs = [funcs[args.a]]

    # hyperparameters
    NUM_ACQUIRE, NUM_EPOCHS, NUM_CLASSES, BATCH_SIZE, SEED, NUM_SEEDS, T, VERBOSE = 10, 50, 10, 128, 1, 3, 100, False
    torch.manual_seed(SEED)
    device = "mps" if torch.mps.is_available() else "cpu"
    print(f"{device=}, {NUM_EPOCHS=}, {NUM_ACQUIRE=}, {NUM_CLASSES=}, {BATCH_SIZE=}, {SEED=}, {NUM_SEEDS=}, {T=}")

    # from offical pytorch mnist example
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(dataset=test_mnist, batch_size=BATCH_SIZE)
    plotting = {a:[] for a in funcs}
    
    for acq_func in funcs:
        start = time.time()
        total_5e = 0
        total_10e = 0
        for _ in range(NUM_SEEDS):
            # INIT BALANCED TRAINING SET
            train_idxs = []
            for i in range(NUM_CLASSES):
                class_examples = (train_mnist.targets==i).nonzero().flatten()
                nums = torch.randperm(len(class_examples)).tolist()[:2]
                train_idxs.extend(class_examples[nums].tolist())
            train_dataset = Subset(train_mnist, train_idxs)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            assert len(train_dataset) == 20

            # INIT VAL
            s_train_idxs = set(train_idxs)
            open_idxs = torch.tensor([i for i in range(len(train_mnist)) if i not in s_train_idxs])
            val_idxs = open_idxs[torch.randperm(len(open_idxs))[:100]].tolist() # pick 100 random idxs
            val_dataset = Subset(train_mnist, val_idxs)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2)

            # INIT POOL 
            s_train_idxs = set(train_idxs)
            s_val_idxs = set(val_idxs)
            pool_idxs = [i for i in range(len(train_mnist)) if i not in s_train_idxs and i not in s_val_idxs]

            # INIT MODEL
            model = KerasMNISTCNN().to(device)
            optimizer = optim.Adam(model.parameters())

            # INITIAL TRAIN
            train(model, device, train_loader, optimizer)
            tl = test(model, device, test_loader)
            plotting[acq_func].append(tl)

            did_10e, did_5e = False, False
            for s in tqdm(range(100)): # 100 acquisition steps
                # ACQUIRE FROM ORACLE
                acquire(acq_func, pool_idxs, train_idxs)

                # update datasets after acquiring
                train_dataset = Subset(train_mnist, train_idxs)
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

                assert set(train_idxs).isdisjoint(set(val_idxs))
                assert set(train_idxs).isdisjoint(set(pool_idxs))
                assert set(val_idxs).isdisjoint(set(pool_idxs))

                # model = KerasMNISTCNN().to(device)
                # optimizer = optim.Adam(model.parameters())

                # train and test
                train(model, device, train_loader, optimizer)
                # vl = test(model, device, val_loader)
                tl = test(model, device, test_loader)
                plotting[acq_func].append(tl)
                N = len(train_dataset)-20
                # print(f"{vl}/100, {tl}/10_000, {N=}")
                if VERBOSE: print(f"{tl}/10_000, {N=}")

                # CHECK FOR CONVERGENCE
                if tl >= 9_000 and not did_10e:
                    total_10e += N
                    did_10e = True
                    if VERBOSE: print(f"reached 10% error with {N} datapoints")
                if tl >= 9_500 and not did_5e:
                    total_5e += N
                    did_5e = True
                    if VERBOSE: print(f"reached 5% error with {N} datapoints")
                    # break
        end = time.time()
        print(f"SUMMARY {acq_func=}: 10% error: {total_10e/NUM_SEEDS}, 5% error: {total_5e/NUM_SEEDS}, TIME:{(end-start)/60}m")
    assert len(plotting[acq_func]) == 101 * NUM_SEEDS
    filename = f"NO_RETRAIN_NOVEL_exp_k3_{funcs[0].__name__}_s{SEED}_dFalse_STD.pkl" if len(funcs) == 1 else f"NO_RETRAIN_NOVEL_exp_k3_all_s{SEED}_dFalse.pkl"
    with open(filename, "wb") as f:
        pickle.dump(plotting, f)
