import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
from utils import *
from model import *
from dataset import *
import copy
import config
import csv

import warnings
warnings.filterwarnings("ignore")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=0.0001, beta_end=0.0200):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(config.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, y):
        with torch.no_grad():
            n = y.shape[0]
            x = torch.randn((n, 1, 40, 48, 40)).to(config.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(config.device)
                predicted_noise = model(x, y, t)
                alpha = self.alpha[t][:, None, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None, None]
                beta = self.beta[t][:, None, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        return x

# First stage
def train_AAE():
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(),lr=config.learning_rate,betas=(0.5, 0.999)) 
    disc = Discriminator().to(config.device)
    opt_disc = optim.Adam(disc.parameters(),lr=config.learning_rate,betas=(0.5, 0.999))

    average = 0

    for epoch in range(config.epochs):
        print("epoch:", epoch)
        lossfile = open("result/"+str(config.exp)+"loss_curve.csv", 'a+',newline = '')
        writer = csv.writer(lossfile)
        if epoch == 0:
            writer.writerow(["Epoch","recon_loss","disc_loss_epoch"])
        
        dataset = OneDataset(root_Abeta=config.whole_Abeta, task = config.train, name = "train")
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)
        length = dataset.length_dataset
        recon_loss_epoch=0
        disc_loss_epoch=0

        for idx, (Abeta, name) in enumerate(loop):
            Abeta = np.expand_dims(Abeta, axis=1)
            Abeta = torch.tensor(Abeta)
            Abeta = Abeta.to(config.device)
            decoded_Abeta = model(Abeta)
            
            disc_real = disc(Abeta)
            disc_fake = disc(decoded_Abeta)

            recon_loss = torch.abs(Abeta - decoded_Abeta).mean()
            g_loss = -torch.mean(disc_fake)
            loss = recon_loss*config.Lambda + g_loss

            d_loss_real = torch.mean(F.relu(1. - disc_real))
            d_loss_fake = torch.mean(F.relu(1. + disc_fake))
            disc_loss = (d_loss_real + d_loss_fake)/2

            opt_model.zero_grad()
            loss.backward(retain_graph=True)

            opt_disc.zero_grad()
            disc_loss.backward()

            opt_model.step()
            opt_disc.step()

            recon_loss_epoch = recon_loss_epoch + recon_loss
            disc_loss_epoch = disc_loss_epoch + disc_loss
        
        writer.writerow([epoch+1, recon_loss_epoch.item()/length, disc_loss_epoch.item()/length])
        lossfile.close()

        #validation part
        dataset = OneDataset(root_Abeta=config.whole_Abeta, task=config.validation, name= "validation")
        loader = DataLoader(dataset,batch_size= 1,shuffle=False,num_workers=config.numworker,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)
        length = dataset.length_dataset
        psnr_0 = 0
        ssim_0 = 0

        csvfile = open("result/"+str(config.exp)+"validation.csv", 'a+',newline = '')
        writer = csv.writer(csvfile)
        if epoch == 0:
            writer.writerow(['Epoch','PSNR','SSIM'])

        for idx, (Abeta, name) in enumerate(loop):
            Abeta = np.expand_dims(Abeta, axis=1)
            Abeta = torch.tensor(Abeta)
            Abeta = Abeta.to(config.device)
            
            decoded_Abeta = model(Abeta)
            decoded_Abeta = torch.clamp(decoded_Abeta,0,1)
            decoded_Abeta = decoded_Abeta.detach().cpu().numpy()
            decoded_Abeta = np.squeeze(decoded_Abeta)
            decoded_Abeta = decoded_Abeta.astype(np.float32)

            Abeta = Abeta.detach().cpu().numpy()
            Abeta = np.squeeze(Abeta)
            Abeta = Abeta.astype(np.float32)
        
            psnr_0 += round(psnr(Abeta,decoded_Abeta),3)
            ssim_0 += round(ssim(Abeta,decoded_Abeta),3)
        
        average_epoch = psnr_0/length + ssim_0 * 10/length
        writer.writerow([epoch+1, psnr_0/length, ssim_0/length])
        csvfile.close()
        
        # test part
        if average_epoch > average:
            average = average_epoch
            save_checkpoint(model, opt_model, filename=config.CHECKPOINT_AAE)

            dataset = OneDataset(root_Abeta=config.whole_Abeta, task=config.test, name= "test")
            loader = DataLoader(dataset,batch_size= 1,shuffle=False,num_workers=config.numworker,pin_memory=True,drop_last=True)
            loop = tqdm(loader, leave=True)
            length = dataset.length_dataset
            psnr_0 = 0
            ssim_0 = 0
            
            csvfile = open("result/"+str(config.exp)+"test.csv", 'a+',newline = '')
            writer = csv.writer(csvfile)
            if epoch == 0:
                writer.writerow(['Epoch','PSNR','SSIM'])

            for idx, (Abeta, name) in enumerate(loop):
                Abeta = np.expand_dims(Abeta, axis=1)
                Abeta = torch.tensor(Abeta)
                Abeta = Abeta.to(config.device)
                
                decoded_Abeta = model(Abeta)
                decoded_Abeta = torch.clamp(decoded_Abeta,0,1)
                decoded_Abeta = decoded_Abeta.detach().cpu().numpy()
                decoded_Abeta = np.squeeze(decoded_Abeta)
                decoded_Abeta = decoded_Abeta.astype(np.float32)

                Abeta = Abeta.detach().cpu().numpy()
                Abeta = np.squeeze(Abeta)
                Abeta = Abeta.astype(np.float32)
            
                psnr_0 += round(psnr(Abeta,decoded_Abeta),3)
                ssim_0 += round(ssim(Abeta,decoded_Abeta),3)

            writer.writerow([epoch+1, psnr_0/length, ssim_0/length])
            csvfile.close()

def encoding():
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(), lr=config.learning_rate,betas=(0.5, 0.9))
    load_checkpoint(config.CHECKPOINT_AAE, model, opt_model, config.learning_rate)
    image = nib.load(config.path)

    dataset = OneDataset(root_Abeta=config.whole_Abeta, task = config.train, name= "Non")
    loader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=config.numworker,pin_memory=True)
    loop = tqdm(loader, leave=True)

    for idx, (Abeta,name) in enumerate(loop):
        Abeta = np.expand_dims(Abeta, axis=1)
        Abeta = torch.tensor(Abeta)
        Abeta = Abeta.to(config.device)
        
        latent_Abeta = model.encoder(Abeta)
        latent_Abeta = latent_Abeta.detach().cpu().numpy()
        latent_Abeta = np.squeeze(latent_Abeta)
        latent_Abeta = latent_Abeta.astype(np.float32)

        latent_Abeta = nib.Nifti1Image(latent_Abeta, image.affine)
        nib.save(latent_Abeta, config.latent_Abeta+str(name[0]))

# Second stage
def train_LDM():
    gpus = config.gpus
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(),lr=config.learning_rate,betas=(0.5, 0.9))
    load_checkpoint(config.CHECKPOINT_AAE, model, opt_model, config.learning_rate)

    Unet = UNet().to(config.device)
    opt_Unet= optim.AdamW(Unet.parameters(), lr=config.learning_rate)
    Unet = nn.DataParallel(Unet,device_ids=gpus,output_device=gpus[0])
    ema = EMA(0.9999)
    ema_Unet = copy.deepcopy(Unet).eval().requires_grad_(False)

    L2 = nn.MSELoss()
    diffusion = Diffusion()
    average = 0

    for epoch in range(config.epochs):
        lossfile = open("result/"+str(config.exp)+"loss_curve.csv", 'a+',newline = '')
        writer = csv.writer(lossfile)
        if epoch == 0:
            writer.writerow(["Epoch","MSE_loss"])
        
        dataset = TwoDataset(root_MRI=config.whole_MRI, root_Abeta=config.latent_Abeta, task = config.train, stage = "train")
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True)
        loop = tqdm(loader, leave=True)
        length = dataset.length_dataset

        MSE_loss_epoch = 0

        for idx, (MRI, latent_Abeta, name, label) in enumerate(loop):
            label = label.to(config.device)
            MRI = np.expand_dims(MRI, axis=1)
            MRI = torch.tensor(MRI)
            MRI = MRI.to(config.device)
            latent_Abeta = np.expand_dims(latent_Abeta, axis=1)
            latent_Abeta = torch.tensor(latent_Abeta)
            latent_Abeta = latent_Abeta.to(config.device)

            t = diffusion.sample_timesteps(latent_Abeta.shape[0]).to(config.device)
            x_t, noise = diffusion.noise_images(latent_Abeta, t)
            predicted_noise = Unet(x_t, MRI, t, label)
            loss = L2(predicted_noise, noise)

            opt_Unet.zero_grad()
            loss.backward()
            opt_Unet.step()
            ema.step_ema(ema_Unet, Unet)

            MSE_loss_epoch += loss

        writer.writerow([epoch+1,MSE_loss_epoch.item()/length])
        lossfile.close()

        #validation part
        dataset = TwoDataset(root_MRI=config.whole_MRI, root_Abeta=config.whole_Abeta, task = config.validation, stage = "validation")
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True)
        loop = tqdm(loader, leave=True)
        length = dataset.length_dataset
        psnr_0 = 0
        ssim_0 = 0

        csvfile = open("result/"+str(config.exp)+"validation.csv", 'a+',newline = '')
        writer = csv.writer(csvfile)
        if epoch == 0:
            writer.writerow(['Epoch','PSNR','SSIM'])

        for idx, (MRI, Abeta, name, label) in enumerate(loop):
            MRI = np.expand_dims(MRI, axis=1)
            MRI = torch.tensor(MRI)
            MRI = MRI.to(config.device)

            sampled_latent = diffusion.sample(ema_Unet, MRI)
            syn_Abeta = model.decoder(sampled_latent)
            syn_Abeta = torch.clamp(syn_Abeta,0,1)
            syn_Abeta = syn_Abeta.detach().cpu().numpy()
            syn_Abeta = np.squeeze(syn_Abeta)
            syn_Abeta = syn_Abeta.astype(np.float32)

            Abeta = np.squeeze(Abeta)
            Abeta = Abeta.astype(np.float32)

            psnr_0 += round(psnr(Abeta,syn_Abeta),3)
            ssim_0 += round(ssim(Abeta,syn_Abeta),3)
        
        average_epoch = psnr_0/length + ssim_0 * 10/length
        writer.writerow([epoch+1, psnr_0/length, ssim_0/length])
        csvfile.close()
        
        # test part
        if average_epoch > average:
            average = average_epoch
            save_checkpoint(ema_Unet, opt_Unet, filename=config.CHECKPOINT_Unet)

            dataset = TwoDataset(root_MRI=config.whole_MRI, root_Abeta=config.whole_Abeta, task = config.test, stage = "test")
            loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True)
            loop = tqdm(loader, leave=True)
            length = dataset.length_dataset
            psnr_0 = 0
            ssim_0 = 0

            csvfile = open("result/"+str(config.exp)+"test.csv", 'a+',newline = '')
            writer = csv.writer(csvfile)
            if epoch == 0:
                writer.writerow(['Epoch','PSNR','SSIM'])

            for idx, (MRI, Abeta, name, label) in enumerate(loop):
                MRI = np.expand_dims(MRI, axis=1)
                MRI = torch.tensor(MRI)
                MRI = MRI.to(config.device)

                sampled_latent = diffusion.sample(ema_Unet, MRI)
                syn_Abeta = model.decoder(sampled_latent)
                syn_Abeta = torch.clamp(syn_Abeta,0,1)
                syn_Abeta = syn_Abeta.detach().cpu().numpy()
                syn_Abeta = np.squeeze(syn_Abeta)
                syn_Abeta = syn_Abeta.astype(np.float32)

                Abeta = np.squeeze(Abeta)
                Abeta = Abeta.astype(np.float32)

                psnr_0 += round(psnr(Abeta,syn_Abeta),3)
                ssim_0 += round(ssim(Abeta,syn_Abeta),3)

            writer.writerow([epoch+1, psnr_0/length, ssim_0/length])
            csvfile.close()

if __name__ == '__main__':
    seed_torch()
    train_AAE()
    encoding()
    train_LDM()
