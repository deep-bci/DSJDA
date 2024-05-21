import numpy as np
import torch

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
    kernel_num: 表示的是多核的数量
    fix_sigma: 表示是否使用固定的标准差
        return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                        矩阵，表达形式:
                        [   K_ss K_st
                            K_ts K_tt ]
    这个函数主要目的是计算Gram核矩阵，用于度量源域和目标域之间的相似性。
    这个核矩阵考虑了不同核的多尺度特性（通过 kernel_mul 和 kernel_num 控制多核的数量和带宽）。
    它将源域和目标域的样本数据合并，并计算了每对样本之间的高斯核值，最后将多个核的结果相加以得到最终的 Gram 核矩阵。
    这个核矩阵在域自适应领域中常用于度量域间的相似性，以帮助域自适应算法进行优化
    """
    # source.size: torch.Size([30, 64])
    # target.size: torch.Size([3, 64])

    # 计算样本总数（包括源域和目标域）
    n_samples = int(source.size()[0])+int(target.size()[0])
    # 合并源域和目标域的数据
    total = torch.cat([source, target], dim=0)

    # 创建两个 total 的扩展矩阵
    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    # 计算 L2 距离矩阵，即每对样本之间的欧氏距离的平方
    L2_distance = ((total0-total1)**2).sum(2) # 计算高斯核中的|x-y|

    # 计算多核中每个核的带宽（bandwidth）
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    # 计算多个高斯核的核矩阵值
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val) # 将多个核合并在一起，得到最终的 Gram 核矩阵# 将多个核合并在一起，得到最终的 Gram 核矩阵

def coefficient(category_1, category_2, sample1_label, sample2_label):
    # 创建布尔掩码以标识属于 category_1 和 category_2 的样本
    cls_bool1 = (sample1_label == category_1)
    cls_bool2 = (sample2_label == category_2)
    # 打印维度
    # cls_bool1.shape: torch.Size([128, 1])
    # cls_bool2.shape: torch.Size([38])
    # 将布尔掩码展平为一维
    cls_bool1 = cls_bool1.view(-1, 1)
    cls_bool2 = cls_bool2.view(-1, 1)

    # 合并两个掩码以获取同时属于 category_1 和 category_2 的样本
    total_cls = torch.cat([cls_bool1, cls_bool2], dim=0).int()

    # 将 total_cls 张量转换为1-D张量
    total_cls = total_cls.view(-1).int()

    # 使用 torch.ger 计算外积（outer product），得到类别之间的系数矩阵
    # 这个系数矩阵表示样本是否同时属于 category_1 和 category_2
    total_coef = torch.ger(total_cls.cpu(), total_cls.cpu()).cuda()

    return total_coef

def lsd(source, target, source_label, target_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # print("source.size",source.size()) # [128, 32]
    # print("target.size",target.size()) # [38, 32]
    # n 是源域样本数量，m 是目标域样本数量
    n = int(source.size()[0])
    m = int(target.size()[0])

    # 计算源域和目标域之间的高斯核矩阵,度量域间的相似性
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n] 
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]
    # print("XX:", XX.size(), ";", "XY:", XY.size(), ";", "YY:", YY.size(), ";", "YX:", YX.size())
    # XX: torch.Size([128, 128]) ; XY: torch.Size([128, 38]) ; YY: torch.Size([38, 38]) ; YX: torch.Size([38, 128])

    # 初始化类内差异和类间差异度量的空列表
    intra_lsd_val = []
    inter_lsd_val = []
    num_class = 3 # seed3 3个类别
    #num_class = 4  # seed4 有4个类别
    # 双重循环迭代每对类别的组合
    for c1 in range(num_class):
        for c2 in range(num_class):
            # 计算类别之间的系数
            coef_val = coefficient(c1, c2, source_label, target_label)
            # 计算不同类别之间的高斯核项，并进行归一化
            e_ss = torch.div(coef_val[:n, :n] * XX, (coef_val[:n, :n]).sum()+1e-5)
            e_st = torch.div(coef_val[:n, n:] * XY, (coef_val[:n, n:]).sum()+1e-5)
            e_ts = torch.div(coef_val[n:, :n] * YX, (coef_val[n:, :n]).sum()+1e-5)
            e_tt = torch.div(coef_val[n:, n:] * YY, (coef_val[n:, n:]).sum()+1e-5)
            # e_ss[e_ss != e_ss] = 0
            # e_st[e_st != e_st] = 0
            # e_ts[e_ts != e_ts] = 0
            # e_tt[e_tt != e_tt] = 0

            lsd_val = e_ss.sum() + e_tt.sum() - e_st.sum() - e_ts.sum()
            """if lsd_val.is_nan():
                    continue
            else:"""
            if c1 == c2: # 将度量值添加到相应的列表中
                intra_lsd_val.append(lsd_val) # 类内差异
            elif c1 != c2:
                inter_lsd_val.append(lsd_val) # 类间差异
    # 计算域内外分布分离度的损失值
    #print(len(inter_lsd_val))
    #loss = sum(intra_lsd_val)/len(intra_lsd_val) - sum(inter_lsd_val)/(len(inter_lsd_val)*(len(inter_lsd_val)-1))
    loss = sum(intra_lsd_val) / len(intra_lsd_val) - sum(inter_lsd_val) / len(inter_lsd_val)
    #loss = sum(intra_lsd_val) / num_class - sum(inter_lsd_val) / (num_class*(num_class-1))
    return loss

