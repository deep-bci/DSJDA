

# For SEED data loading
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import copy
import os
import scipy.io as scio
from scipy.io import loadmat
import numpy as np
import random

random.seed(0)


dataset_path = {'seed4': 'yourpath', 'seed3': 'yourpath','deap':'yourpath'}

'''
Tools
'''


def norminx(data):
    '''
    description: norm in x dimension
    param {type}:
        data: array
    return {type} 
    '''
    for i in range(data.shape[0]):
        data[i] = normalization(data[i])
    return data


def norminy(data):
    dataT = data.T
    for i in range(dataT.shape[0]):
        dataT[i] = normalization(dataT[i])
    return dataT.T


def normalization(data):
    '''
    description: 
    param {type} 
    return {type} 
    '''
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# package the data and label into one class


class CustomDataset(Dataset):
    # initialization: data and label
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    # get the size of data

    def __len__(self):
        return len(self.Data)
    # get the data and label

    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.LongTensor(self.Label[index])
        return data, label

# mmd loss and guassian kernel


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1) % batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


def EntropyLoss(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))



def get_number_of_label_n_trial(dataset_name):
    '''
    description: get the number of categories, trial number and the corresponding labels
    param {type} 
    return {type}:
        trial: int
        label: int
        label_xxx: list 3*15
    '''
    # global variables
    label_seed4 = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                   [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                   [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]
    label_seed3 = [[2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]]
    #SEED3:labels (-1 for negative, 0 for neutral and +1 for positive)
    if dataset_name == 'seed3':
        label = 3
        trial = 15
        return trial, label, label_seed3
    elif dataset_name == 'seed4':
        label = 4
        trial = 24
        return trial, label, label_seed4
    else:
        print('Unexcepted dataset name')


def reshape_data(data, label):
    '''
    description: reshape data and initiate corresponding label vectors
    param {type}:
        data: list
        label: list
    return {type}:
        reshape_data: array, x*310
        reshape_label: array, x*1
    '''
    reshape_data = None
    reshape_label = None

    for i in range(len(data)):
        one_data = np.reshape(np.transpose(
            data[i], (1, 2, 0)), (-1, 310), order='F')
        one_label = np.full((one_data.shape[0], 1), label[i])
        if reshape_data is not None:
            reshape_data = np.vstack((reshape_data, one_data))
            reshape_label = np.vstack((reshape_label, one_label))
        else:
            reshape_data = one_data
            reshape_label = one_label
    return reshape_data, reshape_label


def get_data_label_frommat(mat_path, dataset_name, session_id):
    '''
    description: load data from mat path and reshape to 851*310
    param {type}:
        mat_path: String
        session_id: int
    return {type}: 
        one_sub_data, one_sub_label: array (851*310, 851*1)
    '''
    _, _, labels = get_number_of_label_n_trial(dataset_name)

    mat_data = scio.loadmat(mat_path)
    mat_de_data = {key: value for key,
                   value in mat_data.items() if key.startswith('de_LDS')}
    mat_de_data = list(mat_de_data.values())

    one_sub_data, one_sub_label = reshape_data(mat_de_data, labels[session_id])
    return one_sub_data, one_sub_label


def load_data_split_category(mat_path):
    '''
    description: load data from mat path and reshape to 851*310
    param {type}:
        mat_path: String
        session_id: int
    return {type}:
        one_sub_data, one_sub_label: array (851*310, 851*1)
    '''

    labels_pos = [2, 2, 2, 2, 2]

    labels_neg = [0, 0, 0, 0, 0]

    labels_neu = [1, 1, 1, 1, 1]

    mat_data = scio.loadmat(mat_path)

    mat_de_data_pos_keys = ['de_LDS1', 'de_LDS6', 'de_LDS9', 'de_LDS10', 'de_LDS14']
    mat_de_data_neu_keys = ['de_LDS2', 'de_LDS5', 'de_LDS8', 'de_LDS11', 'de_LDS13']
    mat_de_data_neg_keys = ['de_LDS3', 'de_LDS4', 'de_LDS7', 'de_LDS12', 'de_LDS15']


    mat_de_data_pos = [mat_data[key] for key in mat_data if key in mat_de_data_pos_keys]
    mat_de_data_neu = [mat_data[key] for key in mat_data if key in mat_de_data_neu_keys]
    mat_de_data_neg = [mat_data[key] for key in mat_data if key in mat_de_data_neg_keys]

    one_sub_data_pos, one_sub_label_pos = reshape_data(mat_de_data_pos, labels_pos)
    one_sub_data_neu, one_sub_label_neu = reshape_data(mat_de_data_neu, labels_neu)
    one_sub_data_neg, one_sub_label_neg = reshape_data(mat_de_data_neg, labels_neg)

    print("one_sub_data_pos, one_sub_label_pos",one_sub_data_pos.shape, one_sub_label_pos.shape)
    print("one_sub_data_neu, one_sub_label_neu",one_sub_data_neu.shape, one_sub_label_neu.shape)
    print("one_sub_data_neg, one_sub_label_neg",one_sub_data_neg.shape, one_sub_label_neg.shape)
    return one_sub_data_pos, one_sub_label_pos,one_sub_data_neu, one_sub_label_neu,one_sub_data_neg, one_sub_label_neg

def sample_by_value(list, value, number):
    '''
    @Description: sample the given list randomly with given value
    @param {type}:
        list: list
        value: int {0,1,2,3}
        number: number of sampling
    @return:
        result_index: list
    '''
    result_index = []
    index_for_value = [i for (i, v) in enumerate(list) if v == value]
    result_index.extend(random.sample(index_for_value, number))
    return result_index

'''
For loading data
'''


def get_allmats_name(dataset_name):
    '''
    description: get the names of all the .mat files
    param {type}
    return {type}:
        allmats: list (3*15)

    '''
    path = dataset_path[dataset_name]
    sessions = os.listdir(path)
    sessions.sort()
    allmats = []
    for session in sessions:
        if session != '.DS_Store':
            mats = os.listdir(path + '/' + session)
            mats.sort()
            mats_list = []
            for mat in mats:
                mats_list.append(mat)
            allmats.append(mats_list)

    return path, allmats


def load_data(dataset_name):
    '''
    description: get all the data from one dataset
    param {type} 
    return {type}:
        data: list 3(sessions) * 15(subjects), each data is x * 310
        label: list 3*15, x*1
    '''
    path, allmats = get_allmats_name(dataset_name)
    data = [([0] * 15) for i in range(3)]
    label = [([0] * 15) for i in range(3)]
    for i in range(len(allmats)):
        for j in range(len(allmats[0])):
            mat_path = path + '/' + str(i+1) + '/' + allmats[i][j]

            one_data, one_label = get_data_label_frommat(
                mat_path, dataset_name, i)
            data[i][j] = one_data.copy()
            label[i][j] = one_label.copy()
    return np.array(data), np.array(label)



def load_deap():
    '''
    description:
    param {type}
    return {type}
    '''
    path = dataset_path['deap']
    mats = os.listdir(path)
    mats.sort()
    data=[]
    label_A = []
    label_V = []
    for mat_file in mats:
        if mat_file.endswith('.mat'):
            temp_mat_file = loadmat(os.path.join(path, mat_file), squeeze_me=True)
            temp_data, temp_label_A, temp_label_V = temp_mat_file['data_2'], temp_mat_file['arousal_labels'], \
                                                    temp_mat_file['valence_labels']


            temp_data = np.reshape(np.transpose(temp_data, (1, 2, 0)), (-1, 160), order='F')
            temp_label_A = temp_label_A.reshape(-1, 1)
            temp_label_V = temp_label_V.reshape(-1, 1)

            data.append(temp_data)
            label_A.append(temp_label_A)
            label_V.append(temp_label_V)

    data = np.array(data)
    label_A = np.array(label_A)
    label_V = np.array(label_V)

    print('data, label_A, label_V shapes:', data.shape, label_A.shape, label_V.shape)
    return data, label_A, label_V



def pick_one_data(dataset_name, session_id=1, cd_count=4, sub_id=0):
    '''
    @Description: pick one data from session 2 (or from other sessions), 
    @param {type}:
        session_id: int
        cd_count: int (to indicate the number of calibration data)
    @return: 
        832 for session 1, 851 for session 0
        cd_data: array (x*310, x is determined by cd_count)
        ud_data: array ((832-x)*310, the rest of that sub data)
        cd_label: array (x*1)
        ud_label: array ((832-x)*1)              
    '''
    path, allmats = get_allmats_name(dataset_name)
    mat_path = path + "/" + str(session_id+1) + \
        "/" + allmats[session_id][sub_id]
    mat_data = scio.loadmat(mat_path)
    mat_de_data = {key: value for key,
                   value in mat_data.items() if key.startswith('de_LDS')}
    mat_de_data = list(mat_de_data.values())  # 24 * 62 * x * 5
    cd_list = []
    ud_list = []
    number_trial, number_label, labels = get_number_of_label_n_trial(
        dataset_name)
    session_label_one_data = labels[session_id]
    for i in range(number_label):
        # 根据给定的label值从label链表中拿到全部的index后根据数量随机采样
        cd_list.extend(sample_by_value(
            session_label_one_data, i, int(cd_count/number_label)))
    ud_list.extend([i for i in range(number_trial) if i not in cd_list])
    cd_label_list = copy.deepcopy(cd_list)
    ud_label_list = copy.deepcopy(ud_list)
    for i in range(len(cd_list)):
        cd_list[i] = mat_de_data[cd_list[i]]
        cd_label_list[i] = labels[session_id][cd_label_list[i]]
    for i in range(len(ud_list)):
        ud_list[i] = mat_de_data[ud_list[i]]
        ud_label_list[i] = labels[session_id][ud_label_list[i]]

    # reshape
    cd_data, cd_label = reshape_data(cd_list, cd_label_list)
    ud_data, ud_label = reshape_data(ud_list, ud_label_list)

    return cd_data, cd_label, ud_data, ud_label



def data_loader(data,label,batch_size, session_id=1,subject_id=0):
    one_session_data, one_session_label = copy.deepcopy(data[session_id]), copy.deepcopy(label[session_id])
    data, label = copy.deepcopy(one_session_data[subject_id]), copy.deepcopy(one_session_label[subject_id])
    data_loader = torch.utils.data.DataLoader(dataset=CustomDataset(data, label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
    return data_loader

def get_one_data_and_label(data,label,session_id=1,subject_id=0): #get subject0‘s data and label in session1
    one_session_data, one_session_label = copy.deepcopy(data[session_id]), copy.deepcopy(label[session_id])
    one_data, one_label = copy.deepcopy(one_session_data[subject_id]), copy.deepcopy(one_session_label[subject_id])
    return one_data, one_label

def get_one_data_and_label_allsession(data,label,subject_id=0): #get subject0‘s data and label in session1
    one_data, one_label = copy.deepcopy(data[subject_id]), copy.deepcopy(label[subject_id])
    return one_data, one_label


