import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, f1_score
from data import HandWritten
from loss_function import get_loss,forward_label
from model import RCMCL

np.set_printoptions(precision=4, suppress=True)

def calculate_mean_and_std(numbers):
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    import math
    std = math.sqrt(variance)
    return mean, std

def Normalize(x, power):
    norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
    out = x.div(norm)
    return out

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--annealing_step', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate')
    args = parser.parse_args()

    dataset = HandWritten()
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims

    vartheta=0.1
    zeta=0.5
    beta = 0.5

    test_time = 5
    evidence_vir = []
    results = []
    p_results=[]
    f_results=[]
    for test_t in range(test_time):
        index = np.arange(num_samples)
        np.random.shuffle(index)
        train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]
        #All comparison algorithms employ this conflict construction method.
        dataset.postprocessing(test_index, addNoise=True, sigma=0.5, ratio_noise=0.1, addConflict=True,
                               ratio_conflict=0.4)
        train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_index), batch_size=args.batch_size, shuffle=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = RCMCL(num_views, dims, num_classes, device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        model.to(device)
        print(f'{test_t+1}: training...')

        model.train()
        for epoch in range(1, args.epochs + 1):
            if epoch % (args.epochs/10) == 0:
                print(f'====> {epoch}')
            for X, Y, indexes in train_loader:
                for v in range(num_views):
                    X[v] = X[v].to(device)
                Y = Y.to(device)
                evidences, evidence_a, evidence_con, evidence_div,dissonances,vacuity,alpha_a,S_a,ps= model(X)
                loss = get_loss(dissonances,evidences, evidence_con, evidence_div, Y, epoch, num_classes, args.annealing_step, vartheta, zeta, device)
                con_label_loss = []
                lambda_min=0
                lambda_max=1
                for v in range(len(evidences)):
                    for w in range(v + 1, len(evidences)):
                        con_label_loss.append(forward_label(evidences[v], evidences[w],dissonances[:,v],dissonances[:,w], device))
                loss1 = sum(con_label_loss)
                loss = loss + beta *loss1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        num_correct, num_sample = 0, 0
        all_Y = []
        all_Y_pre = []
        for X, Y, indexes in test_loader:
            for v in range(num_views):
                X[v] = X[v].to(device)
            Y = Y.to(device)
            with torch.no_grad():
                evidences, evidence_a, evidence_con, evidence_div,dissonances,vacuity,alpha_a,S_a,ps = model(X)
                _, Y_pre = torch.max(evidence_a, dim=1)
                num_correct += (Y_pre == Y).sum().item()
                num_sample += Y.shape[0]
                evidence_vir.append(torch.mean(evidence_con).item())
                all_Y.extend(Y.cpu().numpy())
                all_Y_pre.extend(Y_pre.cpu().numpy())

        print('====> acc: {:.4f}'.format(num_correct / num_sample))
        print('====> precision: {:.4f}'.format(precision_score(all_Y, all_Y_pre, average='macro')))
        print('====> f1: {:.4f}'.format(f1_score(all_Y, all_Y_pre, average='macro')))
        results.append(num_correct / num_sample)
        p_results.append(precision_score(all_Y, all_Y_pre, average='macro'))
        f_results.append(f1_score(all_Y, all_Y_pre, average='macro'))
    mean, stdDev = calculate_mean_and_std(results)
    mean1, stdDev1 = calculate_mean_and_std(p_results)
    mean2, stdDev2 = calculate_mean_and_std(f_results)
    print('===========================================')
    print(f'{test_time} times test finish.')
    print('vartheta = {:.2f}'.format(vartheta))
    print('zeta = {:.2f}'.format(zeta))
    print(results)
    print('acc_mean = {:.4f}'.format(mean))
    print('acc_std = {:.4f}'.format(stdDev))
    print(p_results)
    print('precision_mean = {:.4f}'.format(mean1))
    print('precision_std = {:.4f}'.format(stdDev1))
    print(f_results)
    print('f1_mean = {:.4f}'.format(mean2))
    print('f1_std = {:.4f}'.format(stdDev2))


