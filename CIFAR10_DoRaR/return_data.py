import torch
import torchvision as tv
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from torchvision import transforms
from mnist.data_loader import MNIST_modified
from torchtext.legacy import data
from torchtext.vocab import GloVe
# from imdb.data_loader import IMDB_modified, tokenizer_twolevel
from torch.utils.data import Dataset
import os
import numpy as np

import data_extraction_module as dem
import neural_network_model_resample

class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"

def return_data(args):

    name = args.dataset
    root = args.root
    batch_size = args.batch_size
    data_loader = dict()
    device = 0 if args.cuda else -1

    if name in ['mnist', 'MNIST']:

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),])

        train_kwargs = {'root':root, 'mode':'train', 'transform':transform, 'download':True,
                        'load_pred': args.load_pred, 'model_name': args.model_name}
        valid_kwargs = {'root':root, 'mode':'valid', 'transform':transform, 'download':True,
                        'load_pred': args.load_pred, 'model_name': args.model_name}
        test_kwargs = {'root':root, 'mode':'test', 'transform':transform, 'download':False,
                       'load_pred': args.load_pred, 'model_name': args.model_name}
        # transform = transforms.Compose([transforms.ToTensor()])
        #
        # train_kwargs = {'root':root, 'mode':'train', 'transform':transform,  'download':True,
        #                 'load_pred': args.load_pred, 'model_name': args.model_name}
        # valid_kwargs = {'root':root, 'mode':'valid', 'transform':transform,  'download':True,
        #                 'load_pred': args.load_pred, 'model_name': args.model_name}
        # test_kwargs = {'root':root, 'mode':'test', 'transform':transform,  'download':False,
        #                'load_pred': args.load_pred, 'model_name': args.model_name}
        dset = MNIST_modified

        train_data = dset(**train_kwargs)
        valid_data = dset(**valid_kwargs)
        test_data = dset(**test_kwargs)

        # data loader
        num_workers = 0
        train_loader = DataLoader(train_data,
                                   batch_size = batch_size,
                                   shuffle = False,
                                   num_workers = num_workers,
                                   drop_last = True,
                                   pin_memory = True)

        valid_loader = DataLoader(valid_data,
                                   batch_size = batch_size,
                                   shuffle = False,
                                   num_workers = num_workers,
                                   drop_last = False,
                                   pin_memory = True)
        
        test_loader = DataLoader(test_data,
                                    batch_size = batch_size,
                                    shuffle = False,
                                    num_workers = num_workers,
                                    drop_last = False,
                                    pin_memory = True)
    
        data_loader['x_type'] = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        data_loader['y_type'] = torch.cuda.LongTensor if args.cuda else torch.LongTensor
        
    elif name in ['imdb', 'IMDB']:

        embedding_dim = 100
        max_total_num_words = 20000
        text = data.Field(tokenize = tokenizer_twolevel, 
                          batch_first = True)
        label = data.Field(lower = True)
        label_pred = data.Field(use_vocab = False, fix_length = 1)
        fname = data.Field(use_vocab = False, fix_length = 1)
        
        train, valid, test = IMDB_modified.splits(text, label, label_pred, fname,
                                                  root = root, model_name = args.model_name,
                                                  load_pred = args.load_pred)
        print("build vocab...")
        text.build_vocab(train, vectors = GloVe(name = '6B',
                                                dim = embedding_dim,
                                                cache = root), max_size = max_total_num_words)
        label.build_vocab(train)
        
        print("Create Iterator objects for multiple splits of a dataset...")
        train_loader, valid_loader, test_loader = data.Iterator.splits((train, valid, test),
                                                                       batch_size = batch_size,
                                                                       device = device,
                                                                       repeat = False)
        
        data_loader['word_idx'] = text.vocab.itos
        data_loader['x_type'] = torch.cuda.LongTensor if args.cuda else torch.LongTensor
        data_loader['y_type'] = torch.cuda.LongTensor if args.cuda else torch.LongTensor
        data_loader['max_total_num_words'] = max_total_num_words
        data_loader['embedding_dim'] = embedding_dim
        data_loader['max_num_words'] = 50
        data_loader['max_num_sents'] = int(next(iter(train_loader)).text.size(-1) / data_loader['max_num_words'])

    elif name in ['cifar10']:
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转为Tensor
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 归一化
        ])

        num_workers = 0
        trainset = tv.datasets.CIFAR10(
            root='C:\\Users\\QD873\\PycharmProjects\\interpretableAI\\cifar-10-python',
            train=True,
            download=True,
            transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  pin_memory=True)

        # 测试集
        testset = tv.datasets.CIFAR10(
            root='C:\\Users\\QD873\\PycharmProjects\\interpretableAI\\cifar-10-python',
            train=False,
            download=True,
            transform=transform)
        test_loader = torch.utils.data.DataLoader(testset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                drop_last=True,
                                pin_memory=True)
        valid_loader = test_loader

        testset_origin = tv.datasets.CIFAR10(
            root='C:\\Users\\QD873\\PycharmProjects\\interpretableAI\\cifar-10-python',
            train=False,
            download=True,
            transform=transforms.ToTensor())
        test_loader_origin = torch.utils.data.DataLoader(testset_origin,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers,
                                                  drop_last=True,
                                                  pin_memory=True)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        data_loader['x_type'] = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        data_loader['y_type'] = torch.cuda.LongTensor if args.cuda else torch.LongTensor

    elif name in ['Imagenet']:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        num_workers = 0
        trainset = tv.datasets.ImageFolder(
            root='C:\\Users\\QD873\\PycharmProjects\\interpretableAI\\Imagenet\\ILSVRC2012_img_train',
            transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  pin_memory=True)

        # 测试集
        testset = tv.datasets.ImageFolder(
            root='C:\\Users\\QD873\\PycharmProjects\\interpretableAI\\Imagenet\\ILSVRC2012_img_val',
            transform=transform)
        valid_loader = torch.utils.data.DataLoader(testset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                drop_last=True,
                                pin_memory=True)
        test_loader = valid_loader

        test_set_origin = tv.datasets.ImageFolder(
            root='C:\\Users\\QD873\\PycharmProjects\\interpretableAI\\Imagenet\\ILSVRC2012_img_val',
            transform=transforms.ToTensor())
        test_loader_origin = torch.utils.data.DataLoader(test_set_origin,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers,
                                                  drop_last=True,
                                                  pin_memory=True)

        data_loader['x_type'] = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        data_loader['y_type'] = torch.cuda.LongTensor if args.cuda else torch.LongTensor

    elif name in ['baseline']:

        userID = [606, 707, 808, 909, 1010, 1200, 1300, 1400, 1600, 1700,
                  800, 1900, 2000, 2100, 2300, 2400, 2600, 2700, 2800, 2900, 3000]
        end_game_base = [13, 12, 14, 14, 10, 10, 11, 10, 16, 7, 10,
                         8, 8, 6, 10, 4, 12, 5, 7, 10, 8]  # the game ID of the last baseline game
        end_game_offset = [47, 48, 49, 44, 44, 48, 43, 43, 50, 41,
                           40, 43, 47, 42, 52, 43, 49, 32, 37, 42, 25]  # the game ID of the last offset game
        end_row_base = [1420, 1283, 1523, 1527, 1100, 1094, 1208, 1100, 1756, 758, 960,
                        871, 860, 654, 1086, 440, 1258, 548, 770, 1096,
                        878]  # note: the 1420 th row data is actually the 1421 th row of the table.
        end_row_offset = [5121, 5174, 5156, 4807, 4834, 5125, 4528, 4693, 5471, 4439,
                          4133, 4696, 5097, 4612, 5510, 4724, 5153, 3492, 4067, 4561, 2748]
        game_last_seg = [27, 22, 29, 24, 24, 18, 28, 17, 30, 23, 15, 27, 11, 25, 37, 25, 10, 20, 10, 20, 14]

        n_trials_each_user = {}
        n_trials_each_user['baseline'] = [153, 132, 165, 164, 110, 121, 132, 121, 187, 88,
                                          76, 98, 97, 87, 142, 77, 136, 66, 88, 120, 77]
        n_trials_each_user['offset'] = np.array(game_last_seg) * 11

        NUM_MOV_PER_TRIAL = 10
        PHASE = 'new data'  # either 'baseline', 'offset', or 'washout','baseline_shiftpeak_noise(0.25)'

        folder_path = 'C:\\Users\\QD873\\PycharmProjects\\informationBN\\bias_mouse\\dizzy-mouse-analysis-master\\dizzy-mouse-analysis-master\\preprocess_by_matlab\\' + PHASE + '\\'
        file_name_ext = '.csv'
        model_saved_dir = 'C:\\Users\\QD873\\PycharmProjects\\informationBN\\bias_mouse\\dizzy-mouse-analysis-master\\dizzy-mouse-analysis-master\\model_saved_dir'

        # -----------------------------------------------------------------------------------------------------

        # data from which users are considered
        start_user = 1
        end_user = 21
        ignore_users = [16, 18, 21]
        num_users = end_user - start_user + 1 - len(ignore_users)
        num_class = 2

        AUC_summary = []
        # for i_iter in range(0, 21):
        i_iter = 0
        print('iteration:', i_iter, ',----------')

        # if (i_iter + 1) in ignore_users:
        #     continue

        legitimate_user = i_iter + 1  # the user ID of legitimate user

        testing_ratio = 0.2
        majority_downsample_ratio = 1
        n_fold = 5  # partion the training data into 5 folds and pick one for validation

        # v1: dem.read_procedural_feature_binary_class()
        (features_varLen, labels, lengths, max_len_global) = dem.read_feature_baseline_v2(
            start_user,
            end_user,
            ignore_users,
            legitimate_user,
            folder_path,
            n_trials_each_user)

        # pad each trial to max_len_global so that the data is in a form of tensor
        features = neural_network_model_resample.pad_data(np.array(features_varLen), 160)  # max_len_global [dx, dy, vx, vy]
        # print('feature shape: ', features.shape)

        # features = np.array(features_varLen)[:, :, :2]
        # features = features[:, :, :, :4]
        # features = features[:,:,:,:2] # only use dx and dy
        # features = dem.generate_dx_dy_vx_vy_data(features) # use dx, dy, vx, and vy
        # features = dem.generate_vx_vy_dt_data(features)

        # --------------separate the data into two parts: for training and testing------------------------
        features_neg, labels_neg, lengths_neg, features_pos, labels_pos, lengths_pos = dem.separate_data(features,
                                                                                                         labels,
                                                                                                         lengths)
        # features_neg, labels_neg, lengths_neg, features_pos, labels_pos, lengths_pos = dem.separate_data(features,labels,lengths)

        # separate negative data
        num_total_neg = len(features_neg)
        # print('num_total_neg', num_total_neg)
        # the number of legitimate trials used to train for this user
        testing_index_neg = [int(1 / testing_ratio) * n + 2 for n in range(num_total_neg // int(1 / testing_ratio))]
        training_index_neg = [x for x in range(num_total_neg) if x not in testing_index_neg]
        print('    testing index negative:', testing_index_neg[:10])

        features_test_neg = np.array([features_neg[i] for i in testing_index_neg])
        labels_test_neg = np.array([labels_neg[i] for i in testing_index_neg])
        lengths_test_neg = np.array([lengths_neg[i] for i in testing_index_neg])

        features_train_neg = np.array([features_neg[i] for i in training_index_neg])
        labels_train_neg = np.array([labels_neg[i] for i in training_index_neg])
        lengths_train_neg = np.array([lengths_neg[i] for i in training_index_neg])
        print('    #negative trials(train):', len(labels_train_neg), '(test):', len(labels_test_neg))

        # ---------- process positive data - approach 1 ----------------
        # #positive trials(train): 349
        # num_total_pos = len(labels_pos)
        # # num_pos_train = 149 #349 # this is deterimined by the offset experiment, 297 for 3 days data,
        # num_to_sample = 149 + 37 # training data + testing data in offset

        # interval_length = num_total_pos//num_to_sample # 2437/349 = 6.9828 ~~= 6
        # sample_idxs = [interval_length*(n+1)-1 for n in range(num_total_pos//interval_length)]
        # sample_features = np.array([features_pos[i] for i in sample_idxs])
        # sample_labels = np.array([labels_pos[i] for i in sample_idxs])
        # sample_lengths = np.array([lengths_pos[i] for i in sample_idxs])

        # # note: this is the index in sample_features
        # testing_index_pos = [int(1/testing_ratio)*(n+1)-2 for n in range(num_to_sample//int(1/testing_ratio))]
        # training_index_pos = [x for x in range(num_to_sample) if x not in testing_index_pos]

        # # training_index_pos = [interval_length*n+(interval_length-1) for n in range(num_pos_train)]
        # # testing_index_pos = [interval_length*n+0 for n in range(num_pos_train)]
        # # testing_index_pos = testing_index_pos[:round(num_pos_train/4)]

        # features_test_pos = np.array([sample_features[i] for i in testing_index_pos])
        # labels_test_pos = np.array([sample_labels[i] for i in testing_index_pos])
        # lengths_test_pos = np.array([sample_lengths[i] for i in testing_index_pos])

        # features_train_pos = np.array([sample_features[i] for i in training_index_pos])
        # labels_train_pos = np.array([sample_labels[i] for i in training_index_pos])
        # lengths_train_pos = np.array([sample_lengths[i] for i in training_index_pos])
        # print('    #positive trials(train):', len(labels_train_pos), '(test):', len(labels_test_pos))

        # ---------- process positive data - approach 2 --------------
        num_total_pos = len(labels_pos)
        testing_index_pos = [11 * n + 4 for n in range(num_total_pos // 11)] + [11 * n + 5 for n in
                                                                                range(num_total_pos // 11)]
        testing_index_pos.sort()
        print('    testing index positive:', testing_index_pos[:10])
        training_index_pos = [x for x in range(num_total_pos) if x not in testing_index_pos]

        features_test_pos = np.array([features_pos[i] for i in testing_index_pos])
        labels_test_pos = np.array([labels_pos[i] for i in testing_index_pos])
        lengths_test_pos = np.array([lengths_pos[i] for i in testing_index_pos])

        features_train_pos = np.array([features_pos[i] for i in training_index_pos])
        labels_train_pos = np.array([labels_pos[i] for i in training_index_pos])
        lengths_train_pos = np.array([lengths_pos[i] for i in training_index_pos])
        print('    #positive trials(train):', len(labels_train_pos), '(test):', len(labels_test_pos))
        # -----------------------------------------

        # combine negative and positive data
        features_train = np.concatenate((features_train_neg, features_train_pos), axis=0)
        labels_train = np.concatenate((labels_train_neg, labels_train_pos), axis=0)
        lengths_train = np.concatenate((lengths_train_neg, lengths_train_pos), axis=0)

        features_test = np.concatenate((features_test_neg, features_test_pos), axis=0)
        labels_test = np.concatenate((labels_test_neg, labels_test_pos), axis=0)
        lengths_test = np.concatenate((lengths_test_neg, lengths_test_pos), axis=0)

        # print('  testing idxs: ', testing_index[(-5):(-1)])

        # --------------------------------------------------------------------------------------------------

        class MyData(Dataset):
            def __init__(self, features, labels, lengths):
                self.features = features
                self.labels = labels
                self.lengths = lengths

            def __getitem__(self, idx):
                # features_reshape = self.features.reshape(self.features.shape[0],1,-1,self.features.shape[3])
                features_reshape = self.features
                sequence = features_reshape[idx,:,:,:]
                # sequence = features_reshape[idx]
                sequence_tensor = torch.tensor(sequence)
                truth = torch.tensor(self.labels)
                truth = torch.argmax(truth, dim = -1)
                truth = truth[idx]
                lengths = torch.tensor(self.lengths)
                lengths = lengths[idx]
                return sequence_tensor, truth, lengths, idx

            def __len__(self):
                return len(self.labels)

        train_data = MyData(features_train, labels_train, lengths_train)
        valid_data = MyData(features_test, labels_test, lengths_test)
        test_data = MyData(features_test, labels_test, lengths_test)

        # data loader
        def my_collate(batch):
            # sequence_tensor, truth, lengths, idx
            # _, idx_sort = torch.sort(batch[2], dim=0, descending=True)
            # t= [i[2] for i in batch]
            # t = np.array(t)
            # t = np.argsort(t)
            # t = t[::-1]
            # batch_new = []
            inputs = [data[0].tolist() for data in batch]
            inputs = torch.tensor(inputs)
            target = [data[1].tolist() for data in batch]
            target = torch.tensor(target)
            lengths = [data[2].tolist() for data in batch]
            lengths = torch.tensor(lengths)
            _, idx_sort = torch.sort(lengths, dim=0, descending=True)
            # idx = [data[3].tolist() for data in batch]
            # idx = torch.tensor(idx)
            # for i in range(batch.__len__()):
            #     batch_new.append(batch[t[i]])
            order_tensor_in = torch.index_select(inputs, dim=0, index=idx_sort)
            order_labels = torch.index_select(target, dim=0, index=idx_sort)
            order_seq_lengths = torch.index_select(lengths, dim=0, index=idx_sort)
            # idx = torch.index_select(idx, dim=0, index=t)
            # train_data.sort(key=lambda data: len(data), reverse=True)
            # data_length = [len(data) for data in train_data]
            # train_data = rnn_utils.pad_sequence(train_data, batch_first=True, padding_value=0)
            return order_tensor_in, order_labels, order_seq_lengths   # 对train_data增加了一维数据

        num_workers = 0
        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=False,
                                  pin_memory=True,
                                  # collate_fn=my_collate
                                  )

        valid_loader = DataLoader(valid_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=False,
                                  pin_memory=True,
                                  # collate_fn=my_collate
                                  )

        test_loader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 drop_last=False,
                                 pin_memory=True,
                                 # collate_fn=my_collate
                                 )

        data_loader['x_type'] = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        data_loader['y_type'] = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    elif name in ['baseline_synthetic']:

        userID = [606, 707, 808, 909, 1010, 1200, 1300, 1400, 1600, 1700,
                  800, 1900, 2000, 2100, 2300, 2400, 2600, 2700, 2800, 2900, 3000]
        end_game_base = [13, 12, 14, 14, 10, 10, 11, 10, 16, 7, 10,
                         8, 8, 6, 10, 4, 12, 5, 7, 10, 8]  # the game ID of the last baseline game
        end_game_offset = [47, 48, 49, 44, 44, 48, 43, 43, 50, 41,
                           40, 43, 47, 42, 52, 43, 49, 32, 37, 42, 25]  # the game ID of the last offset game
        end_row_base = [1420, 1283, 1523, 1527, 1100, 1094, 1208, 1100, 1756, 758, 960,
                        871, 860, 654, 1086, 440, 1258, 548, 770, 1096,
                        878]  # note: the 1420 th row data is actually the 1421 th row of the table.
        end_row_offset = [5121, 5174, 5156, 4807, 4834, 5125, 4528, 4693, 5471, 4439,
                          4133, 4696, 5097, 4612, 5510, 4724, 5153, 3492, 4067, 4561, 2748]
        game_last_seg = [27, 22, 29, 24, 24, 18, 28, 17, 30, 23, 15, 27, 11, 25, 37, 25, 10, 20, 10, 20, 14]

        n_trials_each_user = {}
        n_trials_each_user['baseline'] = [153, 132, 165, 164, 110, 121, 132, 121, 187, 88,
                                          76, 98, 97, 87, 142, 77, 136, 66, 88, 120, 77]
        n_trials_each_user['offset'] = np.array(game_last_seg) * 11

        NUM_MOV_PER_TRIAL = 10
        PHASE = 'new data'  # either 'baseline', 'offset', or 'washout','baseline_shiftpeak_noise(0.25)'

        folder_path = 'C:\\Users\\QD873\\PycharmProjects\\informationBN\\bias_mouse\\dizzy-mouse-analysis-master\\dizzy-mouse-analysis-master\\preprocess_by_matlab\\' + PHASE + '\\'
        file_name_ext = '.csv'
        model_saved_dir = 'C:\\Users\\QD873\\PycharmProjects\\informationBN\\bias_mouse\\dizzy-mouse-analysis-master\\dizzy-mouse-analysis-master\\model_saved_dir'

        # -----------------------------------------------------------------------------------------------------

        # data from which users are considered
        start_user = 1
        end_user = 21
        # ignore_users = [16, 18, 21]
        ignore_users = []
        num_users = end_user - start_user + 1 - len(ignore_users)
        num_class = 2

        AUC_summary = []
        # for i_iter in range(0, 21):
        i_iter = 0
        print('iteration:', i_iter, ',----------')

        # if (i_iter + 1) in ignore_users:
        #     continue

        legitimate_user = i_iter + 1  # the user ID of legitimate user

        testing_ratio = 0.2
        majority_downsample_ratio = 1
        n_fold = 5  # partion the training data into 5 folds and pick one for validation

        # v1: dem.read_procedural_feature_binary_class()
        (features_varLen, labels, lengths, max_len_global) = dem.read_feature_baseline_synthetic(
            start_user,
            end_user,
            ignore_users,
            legitimate_user,
            folder_path,
            n_trials_each_user)

        # pad each trial to max_len_global so that the data is in a form of tensor
        features = neural_network_model_resample.pad_data(np.array(features_varLen), 160)  # max_len_global [dx, dy, vx, vy]
        # print('feature shape: ', features.shape)

        # features = np.array(features_varLen)[:, :, :2]
        # features = features[:, :, :, :4]
        # features = features[:,:,:,:2] # only use dx and dy
        # features = dem.generate_dx_dy_vx_vy_data(features) # use dx, dy, vx, and vy
        # features = dem.generate_vx_vy_dt_data(features)

        # --------------separate the data into two parts: for training and testing------------------------
        features_neg, labels_neg, lengths_neg, features_pos, labels_pos, lengths_pos = dem.separate_data(features,
                                                                                                         labels,
                                                                                                         lengths)
        # features_neg, labels_neg, lengths_neg, features_pos, labels_pos, lengths_pos = dem.separate_data(features,labels,lengths)

        # separate negative data
        num_total_neg = len(features_neg)
        # print('num_total_neg', num_total_neg)
        # the number of legitimate trials used to train for this user
        testing_index_neg = [int(1 / testing_ratio) * n + 2 for n in range(num_total_neg // int(1 / testing_ratio))]
        training_index_neg = [x for x in range(num_total_neg) if x not in testing_index_neg]
        print('    testing index negative:', testing_index_neg[:10])

        features_test_neg = np.array([features_neg[i] for i in testing_index_neg])
        labels_test_neg = np.array([labels_neg[i] for i in testing_index_neg])
        lengths_test_neg = np.array([lengths_neg[i] for i in testing_index_neg])

        features_train_neg = np.array([features_neg[i] for i in training_index_neg])
        labels_train_neg = np.array([labels_neg[i] for i in training_index_neg])
        lengths_train_neg = np.array([lengths_neg[i] for i in training_index_neg])
        print('    #negative trials(train):', len(labels_train_neg), '(test):', len(labels_test_neg))

        # ---------- process positive data - approach 1 ----------------
        # #positive trials(train): 349
        # num_total_pos = len(labels_pos)
        # # num_pos_train = 149 #349 # this is deterimined by the offset experiment, 297 for 3 days data,
        # num_to_sample = 149 + 37 # training data + testing data in offset

        # interval_length = num_total_pos//num_to_sample # 2437/349 = 6.9828 ~~= 6
        # sample_idxs = [interval_length*(n+1)-1 for n in range(num_total_pos//interval_length)]
        # sample_features = np.array([features_pos[i] for i in sample_idxs])
        # sample_labels = np.array([labels_pos[i] for i in sample_idxs])
        # sample_lengths = np.array([lengths_pos[i] for i in sample_idxs])

        # # note: this is the index in sample_features
        # testing_index_pos = [int(1/testing_ratio)*(n+1)-2 for n in range(num_to_sample//int(1/testing_ratio))]
        # training_index_pos = [x for x in range(num_to_sample) if x not in testing_index_pos]

        # # training_index_pos = [interval_length*n+(interval_length-1) for n in range(num_pos_train)]
        # # testing_index_pos = [interval_length*n+0 for n in range(num_pos_train)]
        # # testing_index_pos = testing_index_pos[:round(num_pos_train/4)]

        # features_test_pos = np.array([sample_features[i] for i in testing_index_pos])
        # labels_test_pos = np.array([sample_labels[i] for i in testing_index_pos])
        # lengths_test_pos = np.array([sample_lengths[i] for i in testing_index_pos])

        # features_train_pos = np.array([sample_features[i] for i in training_index_pos])
        # labels_train_pos = np.array([sample_labels[i] for i in training_index_pos])
        # lengths_train_pos = np.array([sample_lengths[i] for i in training_index_pos])
        # print('    #positive trials(train):', len(labels_train_pos), '(test):', len(labels_test_pos))

        # ---------- process positive data - approach 2 --------------
        num_total_pos = len(labels_pos)
        testing_index_pos = [11 * n + 4 for n in range(num_total_pos // 11)] + [11 * n + 5 for n in
                                                                                range(num_total_pos // 11)]
        testing_index_pos.sort()
        print('    testing index positive:', testing_index_pos[:10])
        training_index_pos = [x for x in range(num_total_pos) if x not in testing_index_pos]

        features_test_pos = np.array([features_pos[i] for i in testing_index_pos])
        labels_test_pos = np.array([labels_pos[i] for i in testing_index_pos])
        lengths_test_pos = np.array([lengths_pos[i] for i in testing_index_pos])

        features_train_pos = np.array([features_pos[i] for i in training_index_pos])
        labels_train_pos = np.array([labels_pos[i] for i in training_index_pos])
        lengths_train_pos = np.array([lengths_pos[i] for i in training_index_pos])
        print('    #positive trials(train):', len(labels_train_pos), '(test):', len(labels_test_pos))
        # -----------------------------------------

        # combine negative and positive data
        features_train = np.concatenate((features_train_neg, features_train_pos), axis=0)
        labels_train = np.concatenate((labels_train_neg, labels_train_pos), axis=0)
        lengths_train = np.concatenate((lengths_train_neg, lengths_train_pos), axis=0)

        features_test = np.concatenate((features_test_neg, features_test_pos), axis=0)
        labels_test = np.concatenate((labels_test_neg, labels_test_pos), axis=0)
        lengths_test = np.concatenate((lengths_test_neg, lengths_test_pos), axis=0)

        # print('  testing idxs: ', testing_index[(-5):(-1)])

        # --------------------------------------------------------------------------------------------------

        class MyData(Dataset):
            def __init__(self, features, labels, lengths):
                self.features = features
                self.labels = labels
                self.lengths = lengths

            def __getitem__(self, idx):
                # features_reshape = self.features.reshape(self.features.shape[0],1,-1,self.features.shape[3])
                features_reshape = self.features
                sequence = features_reshape[idx,:,:,:]
                # sequence = features_reshape[idx]
                sequence_tensor = torch.tensor(sequence)
                truth = torch.tensor(self.labels)
                truth = torch.argmax(truth, dim = -1)
                truth = truth[idx]
                lengths = torch.tensor(self.lengths)
                lengths = lengths[idx]
                return sequence_tensor, truth, lengths, idx

            def __len__(self):
                return len(self.labels)

        train_data = MyData(features_train, labels_train, lengths_train)
        valid_data = MyData(features_test, labels_test, lengths_test)
        test_data = MyData(features_test, labels_test, lengths_test)

        # data loader
        def my_collate(batch):
            # sequence_tensor, truth, lengths, idx
            # _, idx_sort = torch.sort(batch[2], dim=0, descending=True)
            # t= [i[2] for i in batch]
            # t = np.array(t)
            # t = np.argsort(t)
            # t = t[::-1]
            # batch_new = []
            inputs = [data[0].tolist() for data in batch]
            inputs = torch.tensor(inputs)
            target = [data[1].tolist() for data in batch]
            target = torch.tensor(target)
            lengths = [data[2].tolist() for data in batch]
            lengths = torch.tensor(lengths)
            _, idx_sort = torch.sort(lengths, dim=0, descending=True)
            # idx = [data[3].tolist() for data in batch]
            # idx = torch.tensor(idx)
            # for i in range(batch.__len__()):
            #     batch_new.append(batch[t[i]])
            order_tensor_in = torch.index_select(inputs, dim=0, index=idx_sort)
            order_labels = torch.index_select(target, dim=0, index=idx_sort)
            order_seq_lengths = torch.index_select(lengths, dim=0, index=idx_sort)
            # idx = torch.index_select(idx, dim=0, index=t)
            # train_data.sort(key=lambda data: len(data), reverse=True)
            # data_length = [len(data) for data in train_data]
            # train_data = rnn_utils.pad_sequence(train_data, batch_first=True, padding_value=0)
            return order_tensor_in, order_labels, order_seq_lengths   # 对train_data增加了一维数据

        num_workers = 0
        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=False,
                                  pin_memory=True,
                                  # collate_fn=my_collate
                                  )

        valid_loader = DataLoader(valid_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=False,
                                  pin_memory=True,
                                  # collate_fn=my_collate
                                  )

        test_loader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 drop_last=False,
                                 pin_memory=True,
                                 # collate_fn=my_collate
                                 )

        data_loader['x_type'] = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        data_loader['y_type'] = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    else: raise UnknownDatasetError()
    
    data_loader['train'] = train_loader
    data_loader['valid'] = valid_loader
    data_loader['test'] = test_loader
    if name in ['cifar10'] or name in ['Imagenet']:
        data_loader['test_origin'] = test_loader_origin
        data_loader['valid_origin'] = test_loader_origin
    return data_loader
