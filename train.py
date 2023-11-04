'''
Final Training script to run traininig for multi-task
'''
import argparse
import random
import numpy as np
import pandas as pd
import logging
import torch
import os
import math
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.task_utils import TasksParam   
from utils.data_utils import METRICS, TaskType
from models.data_manager import allTasksDataset, Batcher, batchUtils
from torch.utils.data import Dataset, DataLoader, BatchSampler
from logger_ import make_logger
from models.model import multiTaskModel
from models.eval import evaluate
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    
def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label,_ in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img

def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD

def train_vaal(models, optimizers, labeled_dataloader, unlabeled_dataloader, cycle, args):
    
    vae = models['vae']
    discriminator = models['discriminator']
    task_model = models['backbone']
    ranker = models['module']
    
    task_model.eval()
    ranker.eval()
    vae.train()
    discriminator.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    vae = vae.to(device)
    discriminator = discriminator.to(device)
    task_model = task_model.to(device)
    ranker = ranker.to(device)

    adversary_param = 1
    beta          = 1
    num_adv_steps = 1
    num_vae_steps = 1

    bce_loss = nn.BCELoss()
    
    labeled_data = read_data(labeled_dataloader)
    unlabeled_data = read_data(unlabeled_dataloader)

    train_iterations = 1250 # int( (args['INCREMENTAL']*cycle+ args['SUBSET']) * args['EPOCHV'] / args['BATCH'] )
    print('Num of Iteration:', str(train_iterations))
    
    for iter_count in range(train_iterations):
        labeled_imgs, labels = next(labeled_data)
        unlabeled_imgs = next(unlabeled_data)[0]
        
        labeled_imgs = labeled_imgs.to(device)
        unlabeled_imgs = unlabeled_imgs.to(device)
        labels = labels.to(device)    
        
        if iter_count == 0 :
            r_l_0 = torch.from_numpy(np.random.uniform(0, 1, size=(labeled_imgs.shape[0],1))).type(torch.FloatTensor).to(device)
            r_u_0 = torch.from_numpy(np.random.uniform(0, 1, size=(unlabeled_imgs.shape[0],1))).type(torch.FloatTensor).to(device)
        else:
            with torch.no_grad():
                _,_,features_l = task_model(labeled_imgs)
                _,_,feature_u = task_model(unlabeled_imgs)
                r_l = ranker(features_l)
                r_u = ranker(feature_u)
        if iter_count == 0:
            r_l = r_l_0.detach()
            r_u = r_u_0.detach()
            r_l_s = r_l_0.detach()
            r_u_s = r_u_0.detach()
        else:
            r_l_s = torch.sigmoid(r_l).detach()
            r_u_s = torch.sigmoid(r_u).detach()   
            
        # VAE step
        for count in range(num_vae_steps): # num_vae_steps
            recon, _, mu, logvar = vae(r_l_s,labeled_imgs)
            unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
            unlab_recon, _, unlab_mu, unlab_logvar = vae(r_u_s,unlabeled_imgs)
            transductive_loss = vae_loss(unlabeled_imgs, 
                    unlab_recon, unlab_mu, unlab_logvar, beta)
        
            labeled_preds = discriminator(r_l,mu)
            unlabeled_preds = discriminator(r_u,unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                
            lab_real_preds = lab_real_preds.to(device)
            unlab_real_preds = unlab_real_preds.to(device)            

            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
            
            optimizers['vae'].zero_grad()
            total_vae_loss.backward()
            optimizers['vae'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_vae_steps - 1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]
                
                labeled_imgs = labeled_imgs.to(device)
                unlabeled_imgs = unlabeled_imgs.to(device)
                labels = labels.to(device)                

        # Discriminator step
        for count in range(num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = vae(r_l_s,labeled_imgs)
                _, _, unlab_mu, _ = vae(r_u_s,unlabeled_imgs)
            
            labeled_preds = discriminator(r_l,mu)
            unlabeled_preds = discriminator(r_u,unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

            lab_real_preds = lab_real_preds.to(device)
            unlab_fake_preds = unlab_fake_preds.to(device)            
            
            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_fake_preds)

            optimizers['discriminator'].zero_grad()
            dsc_loss.backward()
            optimizers['discriminator'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_adv_steps-1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                labeled_imgs = labeled_imgs.to(device)
                unlabeled_imgs = unlabeled_imgs.to(device)
                labels = labels.to(device)                
                
            if iter_count % 50 == 0:
                # print("Iteration: " + str(iter_count) + "  vae_loss: " + str(total_vae_loss.item()) + " dsc_loss: " +str(dsc_loss.item()))
                args['writer-train'].add_scalar(str(cycle) + ' Total VAE Loss ',
                        total_vae_loss.item(), iter_count)
                args['writer-train'].add_scalar(str(cycle) + ' Total DSC Loss ',
                        dsc_loss.item(), iter_count)
                

class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=3, f_filt=4):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.f_filt = f_filt
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, self.f_filt, 2, 1, bias=False),            # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024*6*6)),     #1024*14*12                            # B, 1024*4*4
        )

        self.fc_mu = nn.Linear(1024*6*6, z_dim)                            # B, z_dim
        self.fc_logvar = nn.Linear(1024*6*6, z_dim)                            # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + 1, 1024*6*6),                           # B, 1024*8*8
            View((-1, 1024, 6, 6)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, self.f_filt, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 2, 2),                       # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, r, x):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        z = torch.cat([z,r],1)
        x_recon = self._decode(z)

        return  x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, r,z):  
        z = torch.cat([z, r], 1)
        return self.net(z)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def ResNet18(num_classes = 10, expansion = 1):
    return ResNet(BasicBlock, [2,2,2,2], num_classes, expansion)

class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, expansion, stride=1):
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, expansion=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 1, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 1, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 1, stride=2)
        
        self.linear1 = nn.Linear(512*expansion, 2)
        self.linear2 = nn.Linear(512*expansion, 2)
        self.linear3 = nn.Linear(512*expansion, 2)
        self.linear4 = nn.Linear(512*expansion, 2)
        self.linear5 = nn.Linear(512*expansion, 2)
        self.linear6 = nn.Linear(512*expansion, 2)
        self.linear7 = nn.Linear(512*expansion, 2)
        self.linear8 = nn.Linear(512*expansion, 2)
        self.linear9 = nn.Linear(512*expansion, 2)
        self.linear10 = nn.Linear(512*expansion, 2)
        self.linear11 = nn.Linear(512*expansion, 2)

    def _make_layer(self, block, planes, num_blocks, expansion, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, expansion, stride))
            self.in_planes = planes * expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        outf = out.view(out.size(0), -1)
        outt1 = self.linear1(outf)
        outt2 = self.linear2(outf)
        outt3 = self.linear3(outf)
        outt4 = self.linear4(outf)
        outt5 = self.linear5(outf)
        outt6 = self.linear6(outf)
        outt7 = self.linear7(outf)
        outt8 = self.linear8(outf)
        outt9 = self.linear9(outf)
        outt10 = self.linear10(outf)
        outt11 = self.linear11(outf)

        return [outt1, outt2, outt3, outt4, outt5, outt6, outt7, outt8, outt9, outt10, outt11], outf, [out1, out2, out3, out4]
    
class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[64, 128, 256, 512], interm_dim=128):
        super(LossNet, self).__init__()
        
        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0]*9, interm_dim)
        self.FC2 = nn.Linear(num_channels[1]*9, interm_dim)
        self.FC3 = nn.Linear(num_channels[2]*9, interm_dim)
        self.FC4 = nn.Linear(num_channels[3]*9, interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)
    
    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out

def query_samples(model, method, data_unlabeled, subset, labeled_set, cycle, args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if method == 'TA-VAAL':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=args['BATCH'], 
                                    sampler=SubsetSequentialSampler(subset), 
                                    pin_memory=True)
        labeled_loader = DataLoader(data_unlabeled, batch_size=args['BATCH'], 
                                    sampler=SubsetSequentialSampler(labeled_set), 
                                    pin_memory=True)
        
        vae = VAE()
        discriminator = Discriminator(32)
     
        models      = {'backbone': model['backbone'], 'module': model['module'], 'vae': vae, 'discriminator': discriminator}
        
        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
        optimizers = {'vae': optim_vae, 'discriminator':optim_discriminator}

        train_vaal(models,optimizers, labeled_loader, unlabeled_loader, cycle+1, args)
        task_model = models['backbone']
        ranker = models['module']        
        all_preds, all_indices = [], []

        for images, _, indices in unlabeled_loader:                       
            images = images.to(device)
            with torch.no_grad():
                _,_,features = task_model(images)
                r = ranker(features)
                _, _, mu, _ = vae(torch.sigmoid(r),images)
                preds = discriminator(r,mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1
        # select the points which the discriminator things are the most likely to be unlabeled
        _, arg = torch.sort(all_preds) 
        
        torch.save(vae, 'saved_history/models/vae-' + args['group'] +'cycle-'+str(cycle)+'.pth')
        torch.save(discriminator, 'saved_history/models/discriminator-' + args['group'] +'cycle-'+str(cycle)+'.pth')
        
    return arg

def make_arguments(parser):
    parser.add_argument('--data_dir', type = str, required=True,
                        help='path to directory where prepared data is present')
    parser.add_argument('--task_file', type = str, required = True,
                        help = 'path to the yml task file')
    parser.add_argument('--out_dir', type = str, required=True,
                        help = 'path to save the model')
    parser.add_argument('--epochs', type = int, required=True,
                        help = 'number of epochs to train')
    parser.add_argument('--freeze_shared_model', default=False, action='store_true',
                        help = "True to freeze the loaded pre-trained shared model and only finetune task specific headers")
    parser.add_argument('--train_batch_size', type = int, default=8,
                        help='batch size to use for training')
    parser.add_argument('--eval_batch_size', type = int, default = 32,
                        help = "batch size to use during evaluation")
    parser.add_argument('--grad_accumulation_steps', type =int, default = 1,
                        help = "number of steps to accumulate gradients before update")
    parser.add_argument('--num_of_warmup_steps', type=int, default = 0,
                        help = "warm-up value for scheduler")
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help = "learning rate for optimizer")
    parser.add_argument('--epsilon', type=float, default=1e-8,
                       help="epsilon value for optimizer")
    parser.add_argument('--grad_clip_value', type = float, default=1.0,
                        help = "gradient clipping value to avoid gradient overflowing" )
    parser.add_argument('--log_file', default='multi_task_logs.log', type = str,
                        help = "name of log file to store")
    parser.add_argument('--log_per_updates', default = 10, type = int,
                        help = "number of steps after which to log loss")
    parser.add_argument('--seed', default=42, type = int,
                        help = "seed to set for modules")
    parser.add_argument('--max_seq_len', default=128, type =int,
                        help = "max seq length used for model at time of data preparation")
    parser.add_argument('--save_per_updates', default = 0, type = int,
                        help = "to keep saving model after this number of updates")
    parser.add_argument('--limit_save', default = 10, type = int,
                        help = "max number recent checkpoints to keep saved")
    parser.add_argument('--load_saved_model', type=str, default=None,
                        help="path to the saved model in case of loading from saved")
    parser.add_argument('--eval_while_train', default = False, action = 'store_true',
                        help = "if evaluation on dev set is required during training.")
    parser.add_argument('--test_while_train', default=False, action = 'store_true',
                        help = "if evaluation on test set is required during training.")
    parser.add_argument('--resume_train', default=False, action = 'store_true',
                        help="Set for resuming training from a saved model")
    parser.add_argument('--finetune', default= False, action = 'store_true',
                        help = "If only the shared model is to be loaded with saved pre-trained multi-task model.\
                            In this case, you can specify your own tasks with task file and use the pre-trained shared model\
                            to finetune upon.")
    parser.add_argument('--debug_mode', default = False, action = 'store_true', 
                        help = "record logs for debugging if True")
    parser.add_argument('--silent', default = False, action = 'store_true', 
                        help = "Only write logs to file if True")
    return parser
    
parser = argparse.ArgumentParser()
parser = make_arguments(parser)
args = parser.parse_args()

#setting logging
now = datetime.now()
logDir = now.strftime("%d_%m-%H_%M")
if not os.path.isdir(logDir):
    os.makedirs(logDir)

logger = make_logger(name = "multi_task", debugMode=args.debug_mode,
                    logFile=os.path.join(logDir, args.log_file), silent=args.silent)
logger.info("logger created.")

device = torch.device('cpu')       
#setting seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')

assert os.path.isdir(args.data_dir), "data_dir doesn't exists"
assert os.path.exists(args.task_file), "task_file doesn't exists"
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)


def make_data_handlers(taskParams, mode, isTrain, gpu, labeled_set=None):
    '''
    This function makes the allTaskDataset, Batch Sampler, Collater function
    and DataLoader for train, dev and test files as per mode.
    In order of task file, 
    train file is at 0th index
    dev file is at 1st index
    test file at 2nd index
    '''
    modePosMap = {"train" : 0, "dev" : 1, "test" : 2}
    modeIdx = modePosMap[mode]
    allTaskslist = []
    for taskId, taskName in taskParams.taskIdNameMap.items():
        taskType = taskParams.taskTypeMap[taskName]
        if mode == "test":
            assert len(taskParams.fileNamesMap[taskName])==3, "test file is required along with train, dev"
        #dataFileName =  '{}.json'.format(taskParams.fileNamesMap[taskName][modeIdx].split('.')[0])
        dataFileName = '{}.json'.format(taskParams.fileNamesMap[taskName][modeIdx].lower().replace('.tsv',''))
        taskDataPath = os.path.join(args.data_dir, dataFileName)
        assert os.path.exists(taskDataPath), "{} doesn't exist".format(taskDataPath)
        taskDict = {"data_task_id" : int(taskId),
                    "data_path" : taskDataPath,
                    "data_task_type" : taskType,
                    "data_task_name" : taskName}
        allTaskslist.append(taskDict)

    allData = allTasksDataset(allTaskslist)
    if mode == "train":
        batchSize = args.train_batch_size
    else:
        batchSize = args.eval_batch_size

    if labeled_set is not None:
        batchSampler = Batcher(allData, batchSize=batchSize, seed = args.seed, sampler=SubsetRandomSampler(labeled_set))
        batchSamplerUtils = batchUtils(isTrain = isTrain, modelType= taskParams.modelType,
                                    maxSeqLen = args.max_seq_len, sampler=SubsetRandomSampler(labeled_set))
        multiTaskDataLoader = DataLoader(allData, batch_sampler = batchSampler,
                                    collate_fn=batchSamplerUtils.collate_fn,
                                    pin_memory=gpu, sampler=SubsetRandomSampler(labeled_set))

    else:   
        batchSampler = Batcher(allData, batchSize=batchSize, seed = args.seed)
        batchSamplerUtils = batchUtils(isTrain = isTrain, modelType= taskParams.modelType,
                                    maxSeqLen = args.max_seq_len)
        multiTaskDataLoader = DataLoader(allData, batch_sampler = batchSampler,
                                    collate_fn=batchSamplerUtils.collate_fn,
                                    pin_memory=gpu)

    return allData, batchSampler, multiTaskDataLoader

    

def main():
    allParams = vars(args)
    logger.info('ARGS : {}'.format(allParams))
    # loading if load_saved_model
    if args.load_saved_model is not None:
        assert os.path.exists(args.load_saved_model), "saved model not present at {}".format(args.load_saved_model)
        loadedDict = torch.load(args.load_saved_model, map_location=device)
        logger.info('Saved Model loaded from {}'.format(args.load_saved_model))

        if args.finetune is True:
            '''
            NOTE :- 
            In finetune mode, only the weights from the shared encoder (pre-trained) from the model will be used. The headers
            over the model will be made from the task file. You can further finetune for training the entire model.
            Freezing of the pre-trained moddel is also possible with argument 
            '''
            logger.info('In Finetune model. Only shared Encoder weights will be loaded from {}'.format(args.load_saved_model))
            logger.info('Task specific headers will be made according to task file')
            taskParams = TasksParam(args.task_file)

        else:
            '''
            NOTE : -
            taskParams used with this saved model must also be stored. THE SAVED TASK PARAMS 
            SHALL BE USED HERE TO AVOID ANY DISCREPENCIES/CHANGES IN THE TASK FILE.
            Hence, if changes are made to task file after saving this model, they shall be ignored
            '''
            taskParams = loadedDict['task_params']
            logger.info('Task Params loaded from saved model.')
            logger.info('Any changes made to task file except the data \
                        file paths after saving this model shall be ignored')
            tempTaskParams = TasksParam(args.task_file)
            #transfering the names of file in new task file to loaded task params
            for taskId, taskName in taskParams.taskIdNameMap.items():
                assert taskName in tempTaskParams.taskIdNameMap.values(), "task names changed in task file given.\
                tasks supported for loaded model are {}".format(list(taskParams.taskIdNameMap.values()))

                taskParams.fileNamesMap[taskName] = tempTaskParams.fileNamesMap[taskName]
    else:
        taskParams = TasksParam(args.task_file)
        logger.info("Task params object created from task file...")
        

    allParams['task_params'] = taskParams
    allParams['gpu'] = torch.cuda.is_available()
    logger.info('task parameters:\n {}'.format(taskParams.taskDetails))

    tensorboard = SummaryWriter(log_dir = os.path.join(logDir, 'tb_logs'))
    logger.info("Tensorboard writing at {}".format(os.path.join(logDir, 'tb_logs')))

    # making handlers for train
    logger.info("Creating data handlers for training...")
    #replace with labeled data waala data
####################################################
    ADDENDUM = 100
    NUM_TRAIN = len(allDataTrain)

    indices = list(range(NUM_TRAIN))
    labeled_set = indices[:ADDENDUM]
    unlabeled_set = [x for x in indices if x not in labeled_set]
    data_unlabeled,_,_ = make_data_handlers(taskParams,
                                            "train", isTrain=True,
                                            gpu = allParams['gpu'])
    allDataTrain, BatchSamplerTrain, multiTaskDataLoaderTrain = make_data_handlers(taskParams,
                                                                                "train", isTrain=True,
                                                                                gpu = allParams['gpu'],labeled_set=labeled_set)
    ################################################################
    # if evaluation on dev set is required during training. Labels are required
    # It will occur at the end of each epoch
    if args.eval_while_train:
        logger.info("Creating data handlers for dev...")
        allDataDev, BatchSamplerDev, multiTaskDataLoaderDev = make_data_handlers(taskParams,
                                                                                "dev", isTrain=False,
                                                                                gpu=allParams['gpu'])
    # if evaluation on test set is required during training. Labels are required
    # It will occur at the end of each epoch
    if args.test_while_train:
        logger.info("Creating data handlers for test...")
        allDataTest, BatchSamplerTest, multiTaskDataLoaderTest = make_data_handlers(taskParams,
                                                                                "test", isTrain=False,
                                                                                gpu=allParams['gpu'])
    #######################################    
    random.shuffle(unlabeled_set)
    subset = unlabeled_set[:args['SUBSET']]
    #####################################
    #making multi-task model
    allParams['num_train_steps'] = math.ceil(len(multiTaskDataLoaderTrain)/args.train_batch_size) *args.epochs // args.grad_accumulation_steps
    allParams['warmup_steps'] = args.num_of_warmup_steps
    allParams['learning_rate'] = args.learning_rate
    allParams['epsilon'] = args.epsilon
    
    logger.info("NUM TRAIN STEPS: {}".format(allParams['num_train_steps']))
    logger.info("len of dataloader: {}".format(len(multiTaskDataLoaderTrain)))
    logger.info("Making multi-task model...")
    model = multiTaskModel(allParams)
    #logger.info('################ Network ###################')
    #logger.info('\n{}\n'.format(model.network))

    if args.load_saved_model:
        if args.finetune is True:
            model.load_shared_model(loadedDict, args.freeze_shared_model)
            logger.info('shared model loaded for finetune from {}'.format(args.load_saved_model))
        else:
            model.load_multi_task_model(loadedDict)
            logger.info('saved model loaded with global step {} from {}'.format(model.globalStep,
                                                                            args.load_saved_model))
        if args.resume_train:
            logger.info("Resuming training from global step {}. Steps before it will be skipped".format(model.globalStep))

    # training 
    resCnt = 0
    cycles = 7
    resnet18    = ResNet18(num_classes=11, expansion=9).to(device)
    loss_module = LossNet().to(device)
    models = {'backbone': resnet18, 'module': loss_module}
    method = 'TA-VAAL'

    for cycle in range (cycles):

        for epoch in range(args.epochs):
            logger.info('\n####################### EPOCH {} ###################\n'.format(epoch))
            totalEpochLoss = 0
            text = "Epoch: {}".format(epoch)
            tt = int(allParams['num_train_steps']*args.grad_accumulation_steps/args.epochs)
            with tqdm(total = tt, position=epoch, desc=text) as progress:
                for i, (batchMetaData, batchData) in enumerate(multiTaskDataLoaderTrain):
                    batchMetaData, batchData = BatchSamplerTrain.patch_data(batchMetaData,batchData, gpu = allParams['gpu'])
                    if args.resume_train and args.load_saved_model and resCnt*args.grad_accumulation_steps < model.globalStep:
                        '''
                        NOTE: - Resume function is only to be used in case the training process couldnt
                        complete or you wish to extend the training to some more epochs.
                        Please keep the gradient accumulation step the same for exact resuming.
                        '''
                        resCnt += 1
                        progress.update(1)
                        continue
                    model.update_step(batchMetaData, batchData)
                    totalEpochLoss += model.taskLoss.item()

                    if model.globalStep % args.log_per_updates == 0 and (model.accumulatedStep+1 == args.grad_accumulation_steps):
                        taskId = batchMetaData['task_id']
                        taskName = taskParams.taskIdNameMap[taskId]
                        #avgLoss = totalEpochLoss / ((i+1)*args.train_batch_size)
                        avgLoss = totalEpochLoss / (i+1)
                        logger.info('Steps: {} Task: {} Avg.Loss: {} Task Loss: {}'.format(model.globalStep,
                                                                                        taskName,
                                                                                        avgLoss,
                                                                                        model.taskLoss.item()))
                        
                        tensorboard.add_scalar('train/avg_loss', avgLoss, global_step= model.globalStep)
                        tensorboard.add_scalar('train/{}_loss'.format(taskName),
                                                model.taskLoss.item(),
                                                global_step=model.globalStep)
                    
                    if args.save_per_updates > 0 and  ( (model.globalStep+1) % args.save_per_updates)==0 and (model.accumulatedStep+1==args.grad_accumulation_steps):
                        savePath = os.path.join(args.out_dir, 'multi_task_model_{}_{}.pt'.format(epoch,
                                                                                                model.globalStep))
                        model.save_multi_task_model(savePath)

                        # limiting the checkpoints save, remove checkpoints if beyond limit
                        if args.limit_save > 0:
                            stepCkpMap = {int(ckp.rstrip('.pt').split('_')[-1]) : ckp for ckp in os.listdir(args.out_dir) if ckp.endswith('.pt') }
                            
                            #sorting based on global step
                            stepToDel = sorted(list(stepCkpMap.keys()))[:-args.limit_save]

                            for ckpStep in stepToDel:
                                os.remove(os.path.join(args.out_dir, stepCkpMap[ckpStep]))
                                logger.info('Removing checkpoint {}'.format(stepCkpMap[ckpStep]))

                    progress.update(1)

                #saving model after epoch
                if args.resume_train and args.load_saved_model and resCnt*args.grad_accumulation_steps < model.globalStep:
                    pass
                else:
                    savePath = os.path.join(args.out_dir, 'multi_task_model_{}_{}.pt'.format(epoch, model.globalStep))  
                    model.save_multi_task_model(savePath)

                if args.eval_while_train:
                    logger.info("\nRunning Evaluation on dev...")
                    with torch.no_grad():
                        evaluate(allDataDev, BatchSamplerDev, multiTaskDataLoaderDev, taskParams,
                                model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)

                if args.test_while_train:
                    logger.info("\nRunning Evaluation on test...")
                    wrtPredpath = "test_predictions_{}.tsv".format(epoch)
                    with torch.no_grad():
                        evaluate(allDataTest, BatchSamplerTest, multiTaskDataLoaderTest, taskParams,
                                model, gpu=allParams['gpu'], evalBatchSize = args.eval_batch_size, needMetrics=True, hasTrueLabels=True,
                                wrtDir=args.out_dir, wrtPredPath=wrtPredpath)
        #'INCREMENTAL': 8138
        incremental = 8138
        subset = 20000
        arg = query_samples(models, method, data_unlabeled, subset, labeled_set, cycle, args)
        new_list = list(torch.tensor(subset)[arg][:incremental].numpy())
        labeled_set += list(torch.tensor(subset)[arg][-incremental:].numpy())
        listd = list(torch.tensor(subset)[arg][:-incremental].numpy()) 
        unlabeled_set = listd + unlabeled_set[subset:]

        allDataTrain, BatchSamplerTrain, multiTaskDataLoaderTrain = make_data_handlers(taskParams,
                                                                                "train", isTrain=True,
                                                                                gpu = allParams['gpu'],labeled_set=labeled_set)

if __name__ == "__main__":
    main()