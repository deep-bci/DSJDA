import argparse
import torch
import torch.nn.functional as F
import numpy as np
import copy
import math
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from torch.utils.tensorboard import SummaryWriter
import utils
import models


threshold = 0.0014
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DSJDA():
    def __init__(self, model=models.DSJDANet_deap(), source_loaders=0, target_loader=0, batch_size=256, iteration=10000, lr=0.01, momentum=0.9, log_interval=10):
        self.model = model
        self.model.to(device)
        self.source_loaders = source_loaders
        self.target_loader = target_loader
        self.batch_size = batch_size
        self.iteration = iteration
        self.lr = lr
        self.momentum = momentum
        self.log_interval = log_interval


    def __getModel__(self):
        return self.model

    def train(self):
        source_iters = []
        for i in range(len(self.source_loaders)):
            source_iters.append(iter(self.source_loaders[i]))
        target_iter = iter(self.target_loader)
        correct = 0
        best_true_labels = []
        best_predicted_labels = []

        for i in range(1, self.iteration + 1):
            self.model.train()
            LEARNING_RATE = self.lr
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=LEARNING_RATE)

            for j in range(len(source_iters)):
                try:
                    source_data, source_label = next(source_iters[j])
                except Exception as err:
                    source_iters[j] = iter(self.source_loaders[j])
                    source_data, source_label = next(source_iters[j])
                try:
                    target_data, _ = next(target_iter)
                except Exception as err:
                    target_iter = iter(self.target_loader)
                    target_data, _ = next(target_iter)
                source_data, source_label = source_data.to(
                    device), source_label.to(device)
                target_data = target_data.to(device)

                optimizer.zero_grad()
                cls_loss, mmd_loss, disc_loss, lsd_loss = self.model(source_data, number_of_source=len(
                    source_iters), data_tgt=target_data, label_src=source_label, mark=j)
                gamma = 2 / (1 + math.exp(-10 * (i) / (self.iteration))) - 1
                beta = gamma / 100
                loss = cls_loss + gamma * mmd_loss + beta * (disc_loss + lsd_loss)


                loss.backward()
                optimizer.step()

            if i % (log_interval * 50) == 0:
                t_correct= self.test(i)
                if t_correct > correct:
                    correct = t_correct
                correct_log = correct.item()
                print('interation ' + str(i), "max_acc:", 100. * correct_log / len(self.target_loader.dataset))


        return 100. * correct / len(self.target_loader.dataset)

    def test(self, i):
        self.model.eval()
        test_loss = 0
        correct = 0
        corrects = []
        for i in range(len(self.source_loaders)):
            corrects.append(0)
        with torch.no_grad():
            for data, target in self.target_loader:
                data = data.to(device)
                target = target.to(device)
                preds = self.model(data, len(self.source_loaders))
                for i in range(len(preds)):
                    preds[i] = F.softmax(preds[i], dim=1)
                pred = sum(preds) / len(preds)
                test_loss += F.nll_loss(F.log_softmax(pred,
                                                      dim=1), target.squeeze()).item()
                pred = pred.data.max(1)[1]
                correct += pred.eq(target.data.squeeze()).cpu().sum()

                for j in range(len(self.source_loaders)):
                    pred = preds[j].data.max(1)[1]
                    corrects[j] += pred.eq(target.data.squeeze()).cpu().sum()

            test_loss /= len(self.target_loader.dataset)
        return correct




def cross_subject(data, label, subject_id, batch_size, iteration, lr, momentum, log_interval,select_idxs):
    data=data
    label=label
    train_idxs = select_idxs
    test_idx = subject_id
    target_data, target_label = copy.deepcopy(data[test_idx]), copy.deepcopy(label[test_idx])
    source_data, source_label = copy.deepcopy(data[train_idxs]), copy.deepcopy(label[train_idxs])
    del label
    del data
    source_loaders = []
    for j in range(len(source_data)):
        source_loaders.append(torch.utils.data.DataLoader(dataset=utils.CustomDataset(source_data[j], source_label[j]),
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          drop_last=True))
    target_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(target_data, target_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)


    model = DSJDA(model=models.DSJDANet_deap(pretrained=False, number_of_source=len(source_loaders)),
                    source_loaders=source_loaders,
                    target_loader=target_loader,
                    batch_size=batch_size,
                    iteration=iteration,
                    lr=lr,
                    momentum=momentum,
                    log_interval=log_interval)
    # print(model.__getModel__())

    acc= model.train()
    print('Target_subject_id: {}, acc: {}'.format(test_idx, acc))
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSJDA parameters')
    parser.add_argument('--dataset', type=str, default='deap',
                        help='the dataset used for DSJDA, "seed3" or "seed4" or "deap"')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='size for one batch, integer')
    parser.add_argument('--epoch', type=int, default=200,
                        help='training epoch, integer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    args = parser.parse_args()
    dataset_name = args.dataset
    bn = args.norm_type

    # data preparation
    print('Model name: DSJDA. Dataset name: ', dataset_name)
    if dataset_name == 'deap':
        data, label_A, label_V = utils.load_deap()
    else:
        data, label = utils.load_data(dataset_name)

    #####----------CHOOSE --A   ---V
    label=label_V


    data_tmp = copy.deepcopy(data)
    label_tmp = copy.deepcopy(label)

    # training settings
    batch_size = args.batch_size
    epoch = args.epoch
    lr = args.lr
    print('BS: {}, epoch: {}'.format(batch_size, epoch))
    momentum = 0.9
    log_interval = 10
    iteration = 0
    iteration = math.ceil(epoch * 2400 / batch_size)
    print('Iteration: {}'.format(iteration))



    # store the results
    csub = []
    csub_all = []
    csesn = []
    best_confusion_matrix_all = []
    all_confusion_matrix = 0
    one_time=[]

    for subject_id_main in range(32):
        js = []
        ks = []
        js_select_1 = []
        select_idxs_1 = []
        select_idxs = []
        js_select_2 = []
        sorted_tuples = []
        ks_select = []
        select_idxs_2 = []

        target_data = data_tmp[subject_id_main]
        train_idxs = list(range(32))
        del train_idxs[subject_id_main]
        for value in train_idxs:
            source_data = data_tmp[value]
            if np.any(source_data <= 0):
                source_data[source_data <= 0] = 1e-6
            source_pdf = entropy(source_data, base=math.e)
            if np.any(target_data <= 0):
                target_data[target_data <= 0] = 1e-6
            target_pdf = entropy(target_data, base=math.e)
            js_divergence = jensenshannon(source_pdf, target_pdf)
            js.append(js_divergence)


        for idx, js_divergence in enumerate(js):
            if js_divergence < threshold:
                js_select_1.append((js_divergence, idx))

        if not js_select_1:
            sorted_js = sorted(js)
            min_values = sorted_js[:7]
            for value in min_values:
                select_idxs.append(js.index(value))
        else:
            sorted_tuples = sorted(js_select_1, key=lambda x: x[0])
            select_idxs = [tup[1] for tup in sorted_tuples]

        print(f"目标{subject_id_main}按阈值_1选出来索引【{len(sorted_tuples)}】个: {select_idxs}")


        acc= cross_subject(data_tmp, label_tmp, subject_id_main, batch_size, iteration, lr,
                                                      momentum, log_interval, select_idxs)
        csub.append(acc)
        print("traget_id:", subject_id_main, "   Cross-subject-acc: ", csub)

    one_session_all = csub
    sessions_data = {
        'acc': one_session_all
    }
    #scipy.io.savemat(f'yourfilepath.mat', sessions_data)
    print("one_session_cross_subject_MEAN: ", np.mean(one_session_all), "std: ", np.std(one_session_all))


