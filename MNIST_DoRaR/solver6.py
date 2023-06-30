# School: ISU
# Name: Dong Qin
# Creat Time: 1/16/2022 3:50 PM
import os
import sys
import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tensorflow as tf
import torchvision as tv
import torchvision.models as models
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from utils import cuda, Weight_EMA_Update, label2binary, save_batch, index_transfer, timeSince, UnknownDatasetError, \
    idxtobool, check_model_name
from return_data import return_data
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import data_extraction_module as dem
import neural_network_model_resample


class Solver6(object):

    def __init__(self, args):

        self.args = args
        self.test_model = args.test_model
        self.dataset = args.dataset
        self.epoch = args.epoch
        self.save_image = args.save_image
        self.save_checkpoint = args.save_checkpoint
        self.batch_size = args.batch_size
        self.lr = args.lr  # learning rate
        self.fixed_training = args.fixed_training
        self.alpha = args.alpha
        self.beta = args.beta
        self.cuda = args.cuda
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.num_avg = args.num_avg
        self.global_iter = 0
        self.global_epoch = 0
        self.env_name = os.path.splitext(args.checkpoint_name)[0] if args.env_name is 'main' else args.env_name
        self.start = time.time()
        self.K = args.K
        self.with_noise = self.args.with_noise
        # Dataset
        self.args.root = os.path.join(self.args.dataset, self.args.data_dir)
        self.args.load_pred = True
        self.data_loader = return_data(args=self.args)
        self.args.word_idx = None

        if 'mnist' in self.dataset:

            for idx, (x, _, _, _) in enumerate(self.data_loader['train']):
                xsize = x.size()
                if idx == 0: break

            self.d = torch.tensor(xsize[1:]).prod()
            self.x_type = self.data_loader['x_type']
            self.y_type = self.data_loader['y_type']

            sys.path.append("./" + self.dataset)

            self.original_ncol = 28
            self.original_nrow = 28
            self.args.chunk_size = self.args.chunk_size if self.args.chunk_size > 0 else 2
            self.chunk_size = self.args.chunk_size
            assert np.remainder(self.original_nrow, self.chunk_size) == 0
            self.filter_size = (self.chunk_size, self.chunk_size)
            self.idx_list = [0, 1, 2]

            ## load black box model
            # from mnist.original import Net
            # self.black_box = Net().to(self.device)
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=2)
                    self.conv2 = nn.Conv2d(10, 20, kernel_size=2, stride=2)
                    self.conv2_drop = nn.Dropout2d()
                    self.fc1 = nn.Linear(320, 20)
                    self.fc2 = nn.Linear(20, 10)

                def forward(self, xx):
                    xx = F.relu(F.max_pool2d(self.conv1(xx), 2))
                    x2 = self.conv2(xx)
                    x2 = F.relu(self.conv2_drop(x2))
                    x3 = F.max_pool2d(x2, 2, padding=1)
                    xx = F.relu(x3)
                    xx = xx.view(-1, 320)
                    xx = F.relu(self.fc1(xx))
                    xx = F.dropout(xx, training=self.training)
                    xx = self.fc2(xx)
                    # return F.log_softmax(x, dim=1)
                    return F.log_softmax(xx, dim=1), x2, x3

            self.black_box = Net().to(self.device)

        elif 'cifar10' in self.dataset:

            for idx, (x, _) in enumerate(self.data_loader['train']):
                xsize = x.size()
                if idx == 0: break

            self.d = torch.tensor(xsize[1:]).prod()
            self.x_type = self.data_loader['x_type']
            self.y_type = self.data_loader['y_type']

            sys.path.append("./" + self.dataset)

            self.original_ncol = 32
            self.original_nrow = 32
            self.args.chunk_size = self.args.chunk_size if self.args.chunk_size > 0 else 2
            self.chunk_size = self.args.chunk_size
            assert np.remainder(self.original_nrow, self.chunk_size) == 0
            self.filter_size = (self.chunk_size, self.chunk_size)
            self.idx_list = [0, 1, 2]

            # load black box model
            from dla_simple import SimpleDLA
            self.black_box = SimpleDLA().to(self.device)
            self.black_box = torch.nn.DataParallel(self.black_box)
            # self.black_box = SimpleDLA()
            # self.black_box = self.black_box.to(device)

            trainset = tv.datasets.CIFAR10(
                root='C:\\Users\\QD873\\PycharmProjects\\interpretableAI\\cifar-10-python',
                train=True,
                download=True)
            tt = torch.tensor(trainset.data[:, :, :, :]).cuda()
            self.mu = torch.mean(((tt / 255 - 0.5) / 0.5), dim=0)
            self.sigma = torch.std(((tt / 255 - 0.5) / 0.5), dim=0)
        else:

            raise UnknownDatasetError()

        #%%
            # Black box
        file_path = 'models'
        if 'cifar10' in self.dataset:
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            self.black_box.load_state_dict(checkpoint['net'])
        elif 'baseline' in self.dataset:
            model_name1 = Path(file_path).joinpath('mouse_original.ckpt')
            self.black_box.load_state_dict(torch.load(model_name1))
        elif 'mnist' in self.dataset:
            model_name1 = Path(file_path).joinpath('blackbox.ckpt')
            self.black_box.load_state_dict(torch.load(model_name1))

        if 'cifar10' in self.dataset:
            from featureselector_cifar10 import featureselector_cifar10
            from generative_model1_cifar10 import generative_model1_cifar10
            from generative_model2_cifar10 import generative_model2_cifar10
        else:
            from featureselector import featureselector
            from generative_model1 import generative_model1
            from generative_model2 import generative_model2

        # Network
        if self.test_model == 'RealX':
            self.RealX = tf.saved_model.load('C:/Users/QD873/PycharmProjects/real-x/test result', tags=None, options=None)

        if 'cifar10' in self.dataset:
            self.net1 = cuda(featureselector_cifar10(args=self.args), self.args.cuda)
            self.net2 = cuda(generative_model1_cifar10(args=self.args), self.args.cuda)
            self.net3 = cuda(generative_model2_cifar10(args=self.args), self.args.cuda)
        else:
            self.net1 = cuda(featureselector(args=self.args), self.args.cuda)
            self.net2 = cuda(generative_model1(args=self.args), self.args.cuda)
            self.net3 = cuda(generative_model2(args=self.args), self.args.cuda)

        # model_name1 = Path(file_path).joinpath('featureselector.ckpt')
        # self.net1.load_state_dict(torch.load(model_name1))
        self.net1.weight_init()

        model_name2 = Path(file_path).joinpath('generative_model1.ckpt')
        # self.net2.load_state_dict(torch.load(model_name2))
        self.net2.weight_init()

        # model_name3 = Path(file_path).joinpath('generative_model2.ckpt')
        # self.net3.load_state_dict(torch.load(model_name3))
        self.net3.weight_init()

        self.net_ema1 = Weight_EMA_Update(cuda(featureselector(args=self.args), self.args.cuda), self.net1.state_dict(),
                                         decay=0.999)
        self.net_ema2 = Weight_EMA_Update(cuda(generative_model1(args=self.args), self.args.cuda), self.net2.state_dict(),
                                         decay=0.999)
        self.net_ema3 = Weight_EMA_Update(cuda(generative_model2(args=self.args), self.args.cuda), self.net3.state_dict(),
                                         decay=0.999)

        # Optimizer
        self.optim1 = optim.Adam([{'params': self.net1.parameters()}], lr=self.lr * 1, betas=(0.5, 0.999))
        # , {'params': self.net2.parameters()}
        self.optim2 = optim.Adam(self.net2.parameters(), lr=self.lr * 1, betas=(0.5, 0.999))
        self.optim3 = optim.Adam(self.net3.parameters(), lr=self.lr * 1, betas=(0.5, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optim1, gamma=0.97)
        self.scheduler2 = lr_scheduler.ExponentialLR(self.optim2, gamma=0.97)
        self.scheduler3 = lr_scheduler.ExponentialLR(self.optim3, gamma=0.97)

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.net1.train()
            self.net_ema1.model.train()
            self.net2.train()
            self.net_ema2.model.train()
            self.net3.train()
            self.net_ema3.model.train()
        elif mode == 'eval':
            self.net1.eval()
            self.net_ema1.model.eval()
            self.net2.eval()
            self.net_ema2.model.eval()
            self.net3.eval()
            self.net_ema3.model.eval()
        else:
            raise ('mode error. It should be either train or eval')

    def train(self, test=False):
        self.set_mode('train')

        self.class_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.info_criterion = nn.KLDivLoss(reduction='sum')

        start = time.time()
        for e in range(self.epoch):

            self.global_epoch += 1

            for idx, batch in enumerate(self.data_loader['train']):
                if 'mnist' in self.dataset:

                    x_raw = batch[0]
                    y_raw = batch[2]

                elif 'cifar10' in self.dataset:

                    x_raw = batch[0]
                    y_raw = batch[1]
                else:
                    raise UnknownDatasetError()

                self.global_iter += 1

                x = Variable(cuda(x_raw, self.args.cuda)).type(self.x_type)
                y = Variable(cuda(y_raw, self.args.cuda)).type(self.y_type)

                p_i, Z_hat, Z_hat_fixed = self.net1(x)

                if self.args.fixed_training == 'not_fixed':
                    Z_hat0, fake_image, noised_input = self.net2(x, Z_hat)
                else:
                    Z_hat0, fake_image, noised_input = self.net2(x, Z_hat_fixed)
                logit,_,_ = self.black_box(fake_image)

                if self.args.fixed_training == 'not_fixed':
                    Z_hat02, fake_image2, _ = self.net3(x, Z_hat)
                else:
                    Z_hat02, fake_image2, _ = self.net3(x, Z_hat_fixed)
                logit2,_,_ = self.black_box(fake_image2)

                ## define loss
                y_class = y if len(y.size()) == 1 else torch.argmax(y, dim=-1)

                class_loss = self.class_criterion(logit, y_class.long()).div(math.log(10))
                class_loss2 = self.class_criterion(logit2, y_class.long()).div(math.log(10))

                reconstruction_loss = sum(sum(sum(sum(abs(fake_image - x))))) / 78400
                reconstruction_loss2 = sum(sum(sum(sum(abs(fake_image2 - x))))) / 78400


                if self.fixed_training == 'not_fixed':
                    if self.test_model == 'Ours':
                        total_loss = ((1 - self.alpha) * class_loss - self.alpha * class_loss2) + \
                                     self.beta * ((1 - self.alpha) * reconstruction_loss - self.alpha * reconstruction_loss2) \

                        self.optim1.zero_grad()
                        total_loss.backward(retain_graph = True)
                        self.optim1.step()
                        # self.net_ema1.update(self.net1.state_dict())

                        # retain_graph = True

                    total_loss1 = class_loss + self.beta * reconstruction_loss
                    self.optim2.zero_grad()
                    total_loss1.backward(inputs=list(self.net2.parameters()))
                    self.optim2.step()
                    # self.net_ema2.update(self.net2.state_dict())

                    total_loss2 = class_loss2 + self.beta * reconstruction_loss2
                    self.optim3.zero_grad()
                    total_loss2.backward(inputs=list(self.net3.parameters()))
                    self.optim3.step()
                    # self.net_ema3.update(self.net3.state_dict())
                else:
                    total_loss1 = class_loss + reconstruction_loss
                    self.optim2.zero_grad()
                    total_loss1.backward(inputs=list(self.net2.parameters()))
                    self.optim2.step()
                    # self.net_ema2.update(self.net2.state_dict())

                    total_loss2 = class_loss2 + reconstruction_loss2
                    self.optim3.zero_grad()
                    total_loss2.backward(inputs=list(self.net3.parameters()))
                    self.optim3.step()
                    # self.net_ema3.update(self.net3.state_dict())

                if self.global_iter % 1000 == 0:

                    prediction = torch.argmax(logit, dim=-1)
                    accuracy = torch.eq(prediction, y_class).float().mean()
                    prediction_fixed = torch.argmax(logit, dim=-1)
                    prediction_fixed2 = torch.argmax(logit2, dim=-1)
                    y_class, prediction, prediction_fixed, prediction_fixed2 = y_class.cpu(), prediction.cpu(), prediction_fixed.cpu(), prediction_fixed2.cpu()

                    print('\n\n[TRAINING RESULT]\n')
                    print('epoch {} Time since {}'.format(self.global_epoch, timeSince(self.start)), end="\n")
                    print('global iter {}'.format(self.global_iter), end="\n")

                    print('training_class_loss:')
                    print(class_loss)
                    print('training_class_loss2:')
                    print(class_loss2)
                    print('training_reconstruction_loss:')
                    print(reconstruction_loss)
                    print('training_reconstruction_loss2:')
                    print(reconstruction_loss2)


            self.val(test=test)

            if 'models' not in os.listdir('.'):
                os.mkdir('models')

            print("epoch:{}".format(e + 1))
            print('Time spent is {}'.format(time.time() - start))

        print(" [*] Training Finished!")
        file_path = './models'
        model_name = check_model_name('featureselector.ckpt', file_path)
        model_name = Path(file_path).joinpath(model_name)
        # torch.save(self.net_ema1.model.state_dict(), model_name)
        torch.save(self.net1.state_dict(), model_name)

        model_name = check_model_name('generative_model1.ckpt', file_path)
        model_name = Path(file_path).joinpath(model_name)
        # torch.save(self.net_ema2.model.state_dict(), model_name)
        torch.save(self.net2.state_dict(), model_name)

        model_name = check_model_name('generative_model2.ckpt', file_path)
        model_name = Path(file_path).joinpath(model_name)
        # torch.save(self.net_ema3.model.state_dict(), model_name)
        torch.save(self.net3.state_dict(), model_name)

    def val(self, test=False):
        print('test', test)

        self.set_mode('eval')
        self.class_criterion_val = nn.CrossEntropyLoss(reduction='sum')
        self.info_criterion_val = nn.KLDivLoss(reduction='sum')
        class_loss = 0
        class_loss2 = 0
        reconstruction_loss = 0
        reconstruction_loss2 = 0
        total_loss = 0

        correct = 0
        correct_fixed = 0
        correct_fixed2 = 0
        precision_macro = 0
        recall_macro = 0
        f1_macro = 0
        f1_micro = 0
        precision_fixed_macro = 0
        recall_fixed_macro = 0
        f1_fixed_macro = 0
        f1_fixed_micro = 0
        avg_correct = 0
        avg_correct_fixed = 0
        avg_correct_fixed2 = 0
        avg_precision_macro = 0
        avg_recall_macro = 0
        avg_f1_macro = 0
        avg_f1_micro = 0
        avg_precision_fixed_macro = 0
        avg_recall_fixed_macro = 0
        avg_f1_fixed_macro = 0
        avg_f1_fixed_micro = 0

        total_num = 0
        total_num_ind = 0

        with torch.no_grad():
            data_type = 'test' if test else 'valid'

            for idx, batch in enumerate(self.data_loader[data_type]):

                if 'mnist' in self.dataset:

                    x_raw = batch[0]
                    y_raw = batch[1]
                    y_ori = batch[1]

                elif 'cifar10' in self.dataset:
                    x_raw = batch[0]
                    y_raw = batch[1]
                else:
                    raise UnknownDatasetError()

                # %%
                ## model fit
                x = Variable(cuda(x_raw, self.args.cuda)).type(self.x_type)
                y = Variable(cuda(y_raw, self.args.cuda)).type(self.y_type)

                # %%

                # p_i, Z_hat, Z_hat_fixed = self.net_ema1.model(x)
                p_i, Z_hat, Z_hat_fixed = self.net1(x)

                # log_p_i, Z_hat0, fake_image, noised_input = self.net_ema2.model(x, Z_hat, Z_hat_fixed, p_i, False)
                Z_hat0, fake_image, noised_input = self.net2(x, Z_hat_fixed)
                logit,_,_ = self.black_box(fake_image)

                # log_p_i2, Z_hat02, fake_image2, noised_input2 = self.net_ema3.model(x, Z_hat, Z_hat_fixed, p_i, False)
                Z_hat02, fake_image2, noised_input2 = self.net3(x, Z_hat_fixed)
                logit2,_,_ = self.black_box(fake_image2)

                ## define loss
                y_class = y if len(y.size()) == 1 else torch.argmax(y, dim=-1)

                cla_loss = self.class_criterion_val(logit, y_class.long()).div(math.log(2)) / self.batch_size
                cla_loss2 = self.class_criterion_val(logit2, y_class.long()).div(math.log(2)) / self.batch_size

                class_loss += cla_loss
                class_loss2 += cla_loss2

                recon_loss = sum(sum(sum(sum(abs(fake_image - x))))) / 78400
                recon_loss2 = sum(sum(sum(sum(abs(fake_image2 - x))))) / 78400
                reconstruction_loss += recon_loss
                reconstruction_loss2 += recon_loss2

                total_num += 1
                total_num_ind += y_class.size(0)

                prediction = F.softmax(logit, dim=1).max(1)[1]
                correct += torch.eq(prediction, y_class).float().sum()

                prediction_fixed = F.softmax(logit, dim=1).max(1)[1]
                prediction_fixed2 = F.softmax(logit2, dim=1).max(1)[1]
                correct_fixed += torch.eq(prediction_fixed, y_class).float().sum()
                correct_fixed2 += torch.eq(prediction_fixed2, y_class).float().sum()

                # prediction, prediction_fixed = prediction.cpu(), prediction_fixed.cpu()
                y_class, prediction, prediction_fixed = y_class.cpu(), prediction.cpu(), prediction_fixed.cpu()
                precision_macro += precision_score(y_class, prediction, average='macro')
                precision_fixed_macro += precision_score(y_class, prediction_fixed, average='macro')
                recall_macro += recall_score(y_class, prediction, average='macro')
                recall_fixed_macro += recall_score(y_class, prediction_fixed, average='macro')
                #                recall_weighted += recall_score(y_class, prediction, average = 'weighted')
                f1_macro += f1_score(y_class, prediction, average='macro')
                f1_micro += f1_score(y_class, prediction, average='micro')
                f1_fixed_macro += f1_score(y_class, prediction_fixed, average='macro')
                f1_fixed_micro += f1_score(y_class, prediction_fixed, average='micro')

                # %% save image #
                if self.save_image and (self.global_epoch % 1 == 0 and self.global_epoch >= 1):
                    # print("SAVED!!!!")
                    if idx in self.idx_list:  # (idx == 0 or idx == 200):
                        new_index = torch.Tensor(self.batch_size, self.chunk_size * self.chunk_size * self.K)
                        # new_index = torch.Tensor(self.batch_size, self.K)
                        for i in range(x.size(0)):
                            a = Z_hat0.view(self.batch_size, 784)
                            b = torch.nonzero(a[i, :])
                            c = b.view(self.chunk_size * self.chunk_size * self.K)
                            # c = b.view(self.K)
                            new_index[i, :] = c
                        # filename
                        img_name, _ = os.path.splitext(self.checkpoint_name)
                        img_name = 'figure_' + img_name + '_' + str(self.global_epoch) + "_" + str(idx) + '.png'
                        img_name = Path(self.image_dir).joinpath(img_name)

                        draw_batch = save_batch(dataset=self.dataset,
                                                batch=x,
                                                label=y_ori, label_pred=y_class, label_approx=prediction_fixed,
                                                index=new_index,
                                                filename=img_name,
                                                is_cuda=self.cuda,
                                                word_idx=self.args.word_idx)
                        save_batch.mnist(draw_batch)

            ## Approximation Fidelity (prediction performance)
            accuracy_fixed = correct_fixed / total_num_ind
            accuracy_fixed2 = correct_fixed2 / total_num_ind

            class_loss /= total_num
            reconstruction_loss /= total_num
            total_loss /= total_num

            print('\n\n[VAL RESULT]\n')
            print('epoch {}'.format(self.global_epoch), end="\n")
            print('global iter {}'.format(self.global_iter), end="\n")

            print('acc_fixed:{:.4f}'
                  .format(accuracy_fixed), end='\n')
            print('acc_fixed2:{:.4f}'
                  .format(accuracy_fixed2), end='\n')
            print()
        self.set_mode('train')