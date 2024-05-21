import torch.nn as nn
import torch.nn.functional as F
import torch
from lsd import lsd
from lsd_seed4 import lsd_seed4
from lsd_deap import lsd_deap
import utils

class CFE(nn.Module):
    def __init__(self):
        super(CFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(310, 256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.module(x)
        return x


def pretrained_CFE(pretrained=False):
    model = CFE()
    if pretrained:
        pass
    return model


class CFE_deap(nn.Module):
    def __init__(self):
        super(CFE_deap, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(160, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.module(x)
        return x


def pretrained_CFE_deap(pretrained=False):
    model = CFE_deap()
    if pretrained:
        pass
    return model



class DSFE(nn.Module):
    def __init__(self):
        super(DSFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.module(x)
        return x


class Classifier(nn.Module):
    def __init__(self,input_features=64):
        super(Classifier,self).__init__()
        self.fc=nn.Linear(input_features,3)

    def forward(self,input):
        return F.softmax(self.fc(input),dim=1)

class ClassClassifier(nn.Module):
    def __init__(self, number_of_category):
        super(ClassClassifier, self).__init__()
        self.classifier = nn.Linear(64, number_of_category)

    def forward(self,input):
        return F.softmax(self.fc(input),dim=1)



class DSJDANet(nn.Module):
    def __init__(self, pretrained=False, number_of_source=15, number_of_category=3):
        super(DSJDANet, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        for i in range(number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) +
                 '=nn.Linear(32,' + str(number_of_category) + ')')

    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        '''
        description: take one source data and the target data in every forward operation.
            the mmd loss is calculated between the source data and the target data (both after the DSFE)
            the discrepency loss is calculated between all the classifiers' results (test on the target data)
            the cls loss is calculated between the ground truth label and the prediction of the mark-th classifier
        param {type}:
            mark: int, the order of the current source
            data_src: take one source data each time
            number_of_source: int
            label_Src: corresponding label
            data_tgt: target data
        return {type}
        '''
        mmd_loss = 0
        disc_loss = 0
        data_tgt_DSFE = []
        if self.training == True:
            # common feature extractor
            data_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE = self.sharedNet(data_tgt)
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)
                data_tgt_DSFE.append(data_tgt_DSFE_i)
            # Use the specific feature extractor
            DSFE_name = 'self.DSFE' + str(mark)
            data_src_DSFE = eval(DSFE_name)(data_src_CFE)
            mmd_loss += utils.mmd_linear(data_src_DSFE, data_tgt_DSFE[mark])

            for i in range(len(data_tgt_DSFE)):
                if i != mark:
                    disc_loss += torch.mean(torch.abs(
                        F.softmax(data_tgt_DSFE[mark], dim=1) -
                        F.softmax(data_tgt_DSFE[i], dim=1)
                    ))

            # domain specific classifier and cls_loss
            DSC_name = 'self.cls_fc_DSC' + str(mark)
            pred_src = eval(DSC_name)(data_src_DSFE)
            cls_loss = F.nll_loss(F.log_softmax(
                pred_src, dim=1), label_src.squeeze())

            #lsd_loss
            max_prob, pseudo_label = torch.max(F.softmax(pred_src, dim=1), dim=1)
            confident_bool = max_prob >= 0.60
            confident_example = data_tgt_DSFE[mark][confident_bool]
            confident_label = pseudo_label[confident_bool]
            lsd_loss = lsd(data_src_DSFE, confident_example, label_src, confident_label)

            return cls_loss, mmd_loss, disc_loss, lsd_loss

        else:
            data_CFE = self.sharedNet(data_src)
            pred = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                pred.append(eval(DSC_name)(feature_DSFE_i))
            return pred

class DSJDANet_seed4(nn.Module):
    def __init__(self, pretrained=False, number_of_source=15, number_of_category=4):
        super(DSJDANet_seed4, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        for i in range(number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) +
                 '=nn.Linear(32,' + str(number_of_category) + ')')

    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        '''
        description: take one source data and the target data in every forward operation.
            the mmd loss is calculated between the source data and the target data (both after the DSFE)
            the discrepency loss is calculated between all the classifiers' results (test on the target data)
            the cls loss is calculated between the ground truth label and the prediction of the mark-th classifier
        param {type}:
            mark: int, the order of the current source
            data_src: take one source data each time
            number_of_source: int
            label_Src: corresponding label
            data_tgt: target data
        return {type}
        '''
        mmd_loss = 0
        disc_loss = 0
        data_tgt_DSFE = []
        if self.training == True:
            # common feature extractor
            data_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE = self.sharedNet(data_tgt)
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)
                data_tgt_DSFE.append(data_tgt_DSFE_i)
            DSFE_name = 'self.DSFE' + str(mark)
            data_src_DSFE = eval(DSFE_name)(data_src_CFE)
            mmd_loss += utils.mmd_linear(data_src_DSFE, data_tgt_DSFE[mark])
            for i in range(len(data_tgt_DSFE)):
                if i != mark:
                    disc_loss += torch.mean(torch.abs(
                        F.softmax(data_tgt_DSFE[mark], dim=1) -
                        F.softmax(data_tgt_DSFE[i], dim=1)
                    ))

            # domain specific classifier and cls_loss
            DSC_name = 'self.cls_fc_DSC' + str(mark)
            pred_src = eval(DSC_name)(data_src_DSFE)
            cls_loss = F.nll_loss(F.log_softmax(
                pred_src, dim=1), label_src.squeeze())

            # lsd_loss
            max_prob, pseudo_label = torch.max(F.softmax(pred_src, dim=1), dim=1)
            confident_bool = max_prob >= 0.60
            confident_example = data_tgt_DSFE[mark][confident_bool]
            confident_label = pseudo_label[confident_bool]
            lsd_loss = lsd_seed4(data_src_DSFE, confident_example, label_src, confident_label)

            return cls_loss, mmd_loss, disc_loss, lsd_loss

        else:
            data_CFE = self.sharedNet(data_src)
            pred = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                pred.append(eval(DSC_name)(feature_DSFE_i))
            return pred


class DSJDANet_deap(nn.Module):
    def __init__(self, pretrained=False, number_of_source=32):
        super(DSJDANet_deap, self).__init__()
        self.sharedNet = pretrained_CFE_deap(pretrained=pretrained)
        for i in range(number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) +
                 '=nn.Linear(32,' + str(2) + ')')

    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        '''
        description: take one source data and the target data in every forward operation.
            the mmd loss is calculated between the source data and the target data (both after the DSFE)
            the discrepency loss is calculated between all the classifiers' results (test on the target data)
            the cls loss is calculated between the ground truth label and the prediction of the mark-th classifier
        param {type}:
            mark: int, the order of the current source
            data_src: take one source data each time
            number_of_source: int
            label_Src: corresponding label
            data_tgt: target data
        return {type}
        '''
        mmd_loss = 0
        disc_loss = 0
        lsd_loss = 0  # 添加 LSD 损失
        data_tgt_DSFE = []
        if self.training == True:
            # common feature extractor
            data_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE = self.sharedNet(data_tgt)

            # Each domian specific feature extractor
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)
                data_tgt_DSFE.append(data_tgt_DSFE_i)

            # Use the specific feature extractor
            DSFE_name = 'self.DSFE' + str(mark)
            data_src_DSFE = eval(DSFE_name)(data_src_CFE)
            mmd_loss += utils.mmd_linear(data_src_DSFE, data_tgt_DSFE[mark])

            for i in range(len(data_tgt_DSFE)):
                if i != mark:
                    disc_loss += torch.mean(torch.abs(
                        F.softmax(data_tgt_DSFE[mark], dim=1) -
                        F.softmax(data_tgt_DSFE[i], dim=1)
                    ))

            # domain specific classifier and cls_loss
            DSC_name = 'self.cls_fc_DSC' + str(mark)
            pred_src = eval(DSC_name)(data_src_DSFE)
            cls_loss = F.nll_loss(F.log_softmax(
                pred_src, dim=1), label_src.squeeze())

            #lsd_loss
            max_prob, pseudo_label = torch.max(F.softmax(pred_src, dim=1), dim=1)
            confident_bool = max_prob >= 0.60
            confident_example = data_tgt_DSFE[mark][confident_bool]
            confident_label = pseudo_label[confident_bool]

            lsd_loss = lsd_deap(data_src_DSFE, confident_example, label_src, confident_label)

            return cls_loss, mmd_loss, disc_loss, lsd_loss

        else:
            data_CFE = self.sharedNet(data_src)
            pred = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                pred.append(eval(DSC_name)(feature_DSFE_i))
            return pred












