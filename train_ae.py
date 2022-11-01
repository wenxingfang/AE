#import shutil
import numpy as np
#import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.utils.data
#import torch.nn.utils.rnn
import h5py
#from torch.autograd import Variable
import time
from tqdm import tqdm
import sys
from modules import model_ae as model
#import platform
import argparse
import healpy as hp
import ast
#import torch.distributions as td
#from matplotlib.lines import Line2D

try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False

class Dataset(torch.utils.data.Dataset):
    def __init__(self, filenamelist, channel, startIndex, npe_scale, time_scale):
        super(Dataset, self).__init__()
        
        print("Reading Dataset...")
        self.T = None
        for file in filenamelist:
            f = h5py.File(file, 'r')
            df = f['data'][:]
            f.close()
            tmp_index = np.logical_and(df[:,0,0]>0.5, df[:,0,0]<1.5)##edep cut for 0MeV
            tmp_tensor = torch.tensor(df[tmp_index,channel:channel+1,startIndex:].astype('float32'))
            #tmp_tensor = torch.tensor(df[:,channel:channel+1,startIndex:].astype('float32'))
            self.T = tmp_tensor if self.T is None else torch.cat((self.T,tmp_tensor),0)
        if channel == 0:#npe
            self.T = self.T/(1.0*npe_scale)
        if channel == 1:#ftime
            self.T = self.T/(1.0*time_scale)
        
        self.n = self.T.size()[0]        
        #print('Dataset:self.T size=',self.T.size())
                                    
    def __getitem__(self, index):
        T = self.T[index,]
        #print('Dataset:T size=',T.size())
        return T

    def __len__(self):
        return self.n


class DatasetTest(torch.utils.data.Dataset):
    def __init__(self, filenamelist, channel, startIndex, npe_scale, time_scale):
        super(DatasetTest, self).__init__()
        
        print("Reading DatasetTest...")
        self.T0 = None
        self.T1 = None
        for file in filenamelist:
            f = h5py.File(file, 'r')
            df = f['data'][:]
            f.close()
            tmp_tensor0 = torch.tensor(df[:,channel:channel+1,0:startIndex].astype('float32'))
            tmp_tensor1 = torch.tensor(df[:,channel:channel+1,startIndex: ].astype('float32'))
            self.T0 = tmp_tensor0 if self.T0 is None else torch.cat((self.T0,tmp_tensor0),0)
            self.T1 = tmp_tensor1 if self.T1 is None else torch.cat((self.T1,tmp_tensor1),0)
        if channel == 0:#npe
            self.T1 = self.T1/(1.0*npe_scale)
        if channel == 1:#ftime
            self.T1 = self.T1/(1.0*time_scale)
        
        self.n = self.T1.size()[0]        
                                    
    def __getitem__(self, index):
        T0 = self.T0[index,]
        T1 = self.T1[index,]
        return (T0,T1)

    def __len__(self):
        return self.n



def file_block(files_txt,size):
    blocks = {}
    blocks[0]=[]
    index = 0
    with open(files_txt,'r') as f:
        lines = f.readlines()
        for line in lines:
            if '#' in line:continue
            line = line.replace('\n','')
            line = line.replace(' ','')
            if index == size:
                blocks[len(blocks)]=[]
                index = 0
                blocks[int(len(blocks)-1)].append(line)
                index += 1
            else:
                blocks[int(len(blocks)-1)].append(line)
                index += 1
    return blocks


class Autoencoder(object):
    def __init__(self, batch_size=64, gpu=0, smooth=0.05, parsed={}):
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.smooth = smooth
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = batch_size        
        self.parsed = parsed
        self.train_file_block = file_block(parsed['train_file'],parsed['train_file_bsize'])
        self.valid_file_block = file_block(parsed['valid_file'],parsed['valid_file_bsize'])
        self.test_file_block  = file_block(parsed['test_file' ],parsed['test_file_bsize'])
        #print(f'train file blocks={self.train_file_block}')
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        
        #self.dataset = Dataset(filename='/scratch1/aasensio/doppler_imaging/nside16/stars_T_spots.h5')

        hyperparameters = {
            'NSIDE': 16,
            'channels_enc': 32,
            'dim_latent_enc': 64,
            'n_steps_enc': 3,
            'dim_hidden_dec': 128,
            'dim_hidden_mapping': 128,
            'siren_num_layers': 3
        }
                
        self.model = model.Model(hyperparameters).to(self.device)
        self.loss = nn.MSELoss()
        if parsed['loss']=='mae':
            self.loss = nn.MAELoss()
    
        #idx = np.arange(self.dataset.n_training)
        #np.random.shuffle(idx)

        #self.train_index = idx[0:int((1-validation_split)*self.dataset.n_training)]
        #self.validation_index = idx[int((1-validation_split)*self.dataset.n_training):]

         # Define samplers for the training and validation sets
        #self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        #self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)
        #
        #self.train_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)
        #self.validation_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.validation_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)        
        if parsed['Restore']:
            print('restored from ',parsed['restore_file'])
            checkpoint = torch.load(parsed['restore_file'])
            self.model.load_state_dict(checkpoint['state_dict'])
    def optimize(self, epochs, lr=3e-4):        

        print(f'doing optimizing:')
        best_loss = float('inf')

        self.lr = lr
        self.n_epochs = epochs        

        current_time = time.strftime("%Y-%m-%d-%H:%M")
        self.out_name = parsed['out_name']

        print(f'Model: {self.out_name}')
        print(" Number of params : ", sum(x.numel() for x in self.model.parameters()))

        # Copy model
        #shutil.copyfile(model.__file__, '{0}.model.py'.format(self.out_name))


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if parsed['Restore']:
            print('opt. restored from ',parsed['restore_file'])
            checkpoint = torch.load(parsed['restore_file'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.train_loss = []
        self.valid_loss = []
        best_loss = float('inf')

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        if parsed['scheduler']=='Plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7,patience=2,threshold=0.001,threshold_mode='rel')
        self.beta = 0.0
        self.n_warmup_steps = 10000

        for epoch in range(1, epochs + 1):
            train_loss = self.train(epoch)
            valid_loss = self.validate()
            current_lr = 0
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']
            print(f'epoch{epoch},train_loss={train_loss},valid_loss={valid_loss}, lr={current_lr}')
            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)

            if  (valid_loss < best_loss):
                best_loss = valid_loss

                hyperparameters = self.model.hyperparameters

                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'train_loss': self.train_loss,
                    'valid_loss': self.valid_loss,
                    'best_loss': best_loss,
                    'hyperparameters': hyperparameters,
                    'optimizer': self.optimizer.state_dict(),
                }
                
                print("Saving model...")
                torch.save(checkpoint, f'{self.out_name}')

            if parsed['scheduler']=='Plateau':
                self.scheduler.step(train_loss)
            else:
                self.scheduler.step()

    def train(self, epoch):
        self.model.train()
        current_time = time.strftime("%Y-%m-%d-%H:%M")
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
        print(f"training Epoch {epoch}/{self.n_epochs}    - t={current_time}")
        idx = np.arange(len(self.train_file_block))
        np.random.shuffle(idx)
        total_loss = 0
        n_total = 0
        for i in idx:
            dataset = Dataset(filenamelist=self.train_file_block[i], channel=self.parsed['channel'], startIndex=17, npe_scale=self.parsed['npe_scale'], time_scale=self.parsed['time_scale'])
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, **self.kwargs)
            #t = tqdm(train_loader)
            #for batch_idx, T in enumerate(t):
            for batch_idx, T in enumerate(train_loader):
                                        
                T = T.to(self.device)
                            
                self.optimizer.zero_grad()
                #print('T size=',T.size()) 
                out, z = self.model(T)
                
                loss = self.loss(out.squeeze(), T.squeeze())
                                        
                loss.backward()

                if self.parsed['clip_grad'] != 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.parsed['clip_grad'])
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                total_loss += loss.item()
                n_total += T.size(0)
                #if (batch_idx == 0):
                #    loss_avg = loss.item()
                #else:
                #    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                #if (NVIDIA_SMI):
                #    usage = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                #    memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                #    t.set_postfix(loss=loss_avg, lr=current_lr, gpu=usage.gpu, memfree=f'{memory.free/1024**2:5.1f} MB', memused=f'{memory.used/1024**2:5.1f} MB')
                #else:
                #    t.set_postfix(loss=loss_avg, lr=current_lr)
            
        return total_loss/n_total

    def validate(self):
        current_time = time.strftime("%Y-%m-%d-%H:%M")
        print(f"validing - t={current_time}")
        self.model.eval()
        total_loss = 0
        n_total = 0
        for i in self.valid_file_block:
            dataset = Dataset(filenamelist=self.valid_file_block[i], channel=self.parsed['channel'], startIndex=17, npe_scale=self.parsed['npe_scale'], time_scale=self.parsed['time_scale'])
            validation_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, **self.kwargs)
            #t = tqdm(validation_loader)
            with torch.no_grad():
                #for batch_idx, T in enumerate(t):
                for batch_idx, T in enumerate(validation_loader):
                                                    
                    T = T.to(self.device)
                    
                    out, z = self.model(T)
                
                    loss = self.loss(out.squeeze(), T.squeeze())
                    total_loss += loss.item()
                    n_total += T.size(0)

                    #if (batch_idx == 0):
                    #    loss_avg = loss.item()
                    #else:
                    #    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                                    
                    #t.set_postfix(loss=loss_avg)            
   
        return total_loss/n_total


    def test(self):
        current_time = time.strftime("%Y-%m-%d-%H:%M")
        print(f"testing - t={current_time}")
        self.model.eval()
        total_loss = 0
        n_total = 0
        start_index = 17
        NPIX = hp.nside2npix(self.model.hyperparameters['NSIDE'])
        df  = np.full((self.batch_size*100, 1, start_index+2*NPIX), 0, np.float32)
        df_index = 0
        test_i = 0
        for i in self.test_file_block:
            dataset = DatasetTest(filenamelist=self.test_file_block[i], channel=self.parsed['channel'], startIndex=17, npe_scale=self.parsed['npe_scale'], time_scale=self.parsed['time_scale'])
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, **self.kwargs)
            #t = tqdm(test_loader)
            with torch.no_grad():
                #for batch_idx, T in enumerate(t):
                for batch_idx, T in enumerate(test_loader):
                    T0 = T[0]                                
                    T1 = T[1]         
                    assert T0.size(0)==T1.size(0)                       
                    T1 = T1.to(self.device)
                  
                    out, z = self.model(T1)
                
                    loss = self.loss(out.squeeze(), T1.squeeze())
                    total_loss += loss.item()
                    n_total += T1.size(0)
                    y_pred = out.cpu()
                    y_pred = y_pred.detach().numpy()##B,NPIX
                    
                    df[df_index:df_index+T0.size(0),0,0:start_index]= T0.squeeze().numpy()
                    df[df_index:df_index+T0.size(0),0,start_index:start_index+NPIX]= T1.squeeze().cpu().numpy()
                    df[df_index:df_index+T0.size(0),0,start_index+NPIX:]= y_pred
                    df_index += T0.size(0) 
                    if df_index>= df.shape[0]:
                        outFile1 = self.parsed['outFile'].replace('.h5','_%d.h5'%test_i)
                        hf = h5py.File(outFile1, 'w')
                        hf.create_dataset('Pred', data=df)
                        hf.close()
                        print('Saved produced data %s'%outFile1)
                        test_i += 1
                        df  = np.full((self.batch_size*100, 1, start_index+2*NPIX), 0, np.float32)
                        df_index = 0
                if True:
                    tmp_index = []
                    for i in range(df.shape[0]):
                        if df[i,0,0]==0: ##edep
                            tmp_index.append(i)
                    df = np.delete(df, tmp_index, 0)
                if df.shape[0]>0:
                    outFile1 = self.parsed['outFile'].replace('.h5','_%d.h5'%test_i)
                    hf = h5py.File(outFile1, 'w')
                    hf.create_dataset('Pred', data=df)
                    hf.close()
                    print('Saved produced data %s'%outFile1)
                    test_i += 1
                df  = np.full((self.batch_size*100, 1, start_index+2*NPIX), 0, np.float32)
                df_index = 0
        return total_loss/n_total            

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, metavar='LR', help='Learning rate')
    parser.add_argument('--gpu', '--gpu', default=0, type=int, metavar='GPU', help='GPU')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.05, type=float, metavar='SM', help='Smoothing factor for loss')
    parser.add_argument('--epochs', '--epochs', default=200, type=int, metavar='EPOCHS', help='Number of epochs')
    parser.add_argument('--batch', '--batch', default=128, type=int, metavar='BATCH', help='Batch size')
    parser.add_argument('--train_file', default='', type=str, help='')
    parser.add_argument('--valid_file', default='', type=str, help='')
    parser.add_argument('--test_file' , default='', type=str, help='')
    parser.add_argument('--train_file_bsize', default=100, type=int, help='')
    parser.add_argument('--valid_file_bsize', default=100, type=int, help='')
    parser.add_argument('--test_file_bsize' , default=100, type=int, help='')
    parser.add_argument('--out_name' , default='', type=str, help='')
    parser.add_argument('--channel'  , default=0, type=int, help='0 for npe, 1 for first hit time')
    parser.add_argument('--npe_scale', default=1, type=float, help='')
    parser.add_argument('--time_scale', default=1, type=float, help='')
    parser.add_argument('--Restore', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--restore_file' , default='', type=str, help='')
    parser.add_argument('--outFile' , default='', type=str, help='')
    parser.add_argument('--DoTest', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--DoOptimization', action='store', type=ast.literal_eval, default=True, help='')
    parser.add_argument('--clip_grad', default=1, type=float, help='')
    parser.add_argument('--scheduler' , default='', type=str, help='')
    parser.add_argument('--loss' , default='mse', type=str, help='')
    
    parsed = vars(parser.parse_args())

    network = Autoencoder(batch_size=parsed['batch'], gpu=parsed['gpu'], smooth=parsed['smooth'], parsed=parsed)
    if parsed['DoOptimization']:
        network.optimize(parsed['epochs'], lr=parsed['lr'])
    if parsed['DoTest']:
        network.test()
