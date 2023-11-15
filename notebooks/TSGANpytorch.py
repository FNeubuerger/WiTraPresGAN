from typing import Concatenate
import torch
from torch import nn
from torch.utils.data import Dataset
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
from glob import glob

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        return output

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),            
        )

    def forward(self, x):
        output = self.model(x)
        return output   

class TorchDS(Dataset):

  def __init__(self,file_dir, labels):
    nodelabels=['timestamp', 'Stempel_innen_mitte', 'Stempel_aussen', 'Matrize_zarge_oben', 'Matrize_zarge_mitte','Matrize_zarge_unten', 'Werkstueck_boden', 'Werkstueck_zarge_unten' , 'Werkstueck_zarge_mitte', 'Werkstueck_zarge_oben']

    filenames = glob(os.path.append(file_dir,"*.csv")
    print(len(filenames))
    train_df = []
    for filename in filenames:
        df = pd.read_csv(filename, names=nodelabels, skiprows=1, index_col=False).drop_duplicates()
        train_df.append(df)
   
    #train_data = tf.keras.preprocessing.sequence.pad_sequences(train_df)
    df = pd.concat(train_df)
    y = df[labels].values
    x = df.values

    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.float32)

  def __len__(self):
    return len(self.y_train)
  
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]

if __name__=='__main__':
    torch.manual_seed(111)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lr = 0.001
    num_epochs = 300
    batch_size = 128
    loss_function = nn.BCELoss()

    discriminator = nn.DataParallel(Discriminator())
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    discriminator.to(device=device)
    
    generator=nn.DataParallel(Generator())
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
    generator.to(device=device)
    
    if os.path.isfile('discriminator.pt') and os.path.isfile('generator.pt'):
        discriminator.load_state_dict(torch.load('./discriminator.pt'))
        generator.load_state_dict(torch.load('./generator.pt'))   
    else:
        for epoch in range(num_epochs):
            for n, (real_samples, mnist_labels) in enumerate(train_loader):
                # Data for training the discriminator
                real_samples = real_samples.to(device=device)
                real_samples_labels = torch.ones((batch_size, 1)).to(device=device)
                latent_space_samples = torch.randn((batch_size, 100)).to(device=device)
                generated_samples = generator(latent_space_samples)
                generated_samples_labels = torch.zeros(
                    (batch_size, 1)).to(device=device)
                all_samples = torch.cat((real_samples, generated_samples))
                all_samples_labels = torch.cat(
                    (real_samples_labels, generated_samples_labels))

                # Training the discriminator
                discriminator.zero_grad()
                output_discriminator = discriminator(all_samples)
                loss_discriminator = loss_function(
                    output_discriminator, all_samples_labels)
                loss_discriminator.backward()
                optimizer_discriminator.step()

                # Data for training the generator
                latent_space_samples = torch.randn((batch_size, 100)).to(device=device)

                # Training the generator            
                generator.zero_grad()
                generated_samples = generator(latent_space_samples)
                output_discriminator_generated = discriminator(generated_samples)
                loss_generator = loss_function(
                    output_discriminator_generated, real_samples_labels)
                loss_generator.backward()
                optimizer_generator.step()

                # Show loss
                if n == batch_size - 1:
                    print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                    print(f"Epoch: {epoch} Loss G.: {loss_generator}")

    latent_space_samples = torch.randn(batch_size, 100).to(device=device)

    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.cpu().detach()
    

    # Save trained NN parameters
    torch.save(generator.state_dict(), 'generator.pt')
    torch.save(discriminator.state_dict(), 'discriminator.pt')


