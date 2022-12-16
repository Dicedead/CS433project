import time
import torch

import pandas as pd

from torch import nn, optim, autograd

import matplotlib.pyplot as plt

from dataclasses import dataclass
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


@dataclass
class DistanceHyperparameters:
    batchsize: int = 128
    num_epochs: int = 1
    noise_size: int = 2
    n_critic: int = 5
    gp_lambda: float = 10.


class DistanceGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(2 + 1, 64), # input: (noise , en_p)  # TODO hp in class params
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1), # output: (dist_p)
            nn.Tanh()
        )

    def forward(self, noise, labels):
        out = self.model(torch.cat([noise, labels], dim=-1))
        return out


class DistanceCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 32), # input: (dist_p, en_p)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, event, labels):
        out = self.model(torch.cat([event, labels], dim=-1))
        return out.squeeze()


class DistanceDataset(Dataset):

    def __init__(self, path="../pickled_data/water_dataset.pkl", test_size=0.3):
        dataset = pd.read_pickle(path)
        dataset = dataset[dataset["emission"] == 1]

        x = dataset[["dist_p"]]
        y = dataset[["en_p"]]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

        self.x_train = torch.from_numpy(x_train.values).float()
        self.y_train = torch.from_numpy(y_train.values).float()
        self.x_test = x_test
        self.y_test = y_test

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.manual_seed(1)

    hp = DistanceHyperparameters()

    dataset = DistanceDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=hp.batchsize, num_workers=1, shuffle=True,
                                             drop_last=True, pin_memory=True)
    critic, generator = DistanceCritic().to("cuda"), DistanceGenerator().to("cuda")

    critic_optimizer = optim.AdamW(critic.parameters(), lr=1e-4, betas=(0., 0.9))
    generator_optimizer = optim.AdamW(generator.parameters(), lr=1e-4, betas=(0., 0.9))

    img_list, generator_losses, critic_losses = [], [], []
    iters = 0
    grad_tensor = torch.ones((hp.batchsize, 1), device="cuda")

    start_time = time.time()

    for epoch in range(hp.num_epochs):
        for batch_idx, data in enumerate(dataloader):
            real_images, real_class_labels = data[0].to("cuda"), data[1].to("cuda")

            # Update critic
            critic_optimizer.zero_grad()

            critic_output_real = critic(real_images, real_class_labels)
            critic_loss_real = critic_output_real.mean()

            noise = torch.randn((hp.batchsize, hp.noise_size), device="cuda")
            with torch.no_grad():
                fake_image = generator(noise, real_class_labels)
            critic_output_fake = critic(fake_image, real_class_labels)
            critic_loss_fake = critic_output_fake.mean()

            alpha = torch.rand(1, device="cuda")
            interpolates = (alpha * real_images + (1. - alpha) * fake_image).requires_grad_(True)
            d_interpolates = critic(interpolates, real_class_labels).reshape(-1, 1)
            gradients = autograd.grad(d_interpolates, interpolates, grad_tensor, create_graph=True, only_inputs=True)[0]
            gradient_penalty = hp.gp_lambda * ((gradients.view(hp.batchsize, -1).norm(dim=1) - 1.) ** 2).mean()

            critic_loss = -critic_loss_real + critic_loss_fake + gradient_penalty

            critic_loss.backward()
            critic_optimizer.step()

            if batch_idx % hp.n_critic == 0:
                # Update Generator
                generator_optimizer.zero_grad()

                fake_class_labels = dataset.y_train[torch.randint(high=len(dataset), size=[hp.batchsize])].to(
                    device="cuda")
                noise = torch.randn((hp.batchsize, hp.noise_size), device="cuda")
                with torch.no_grad(): fake_image = generator(noise, fake_class_labels)
                critic_output_fake = critic(fake_image, fake_class_labels)
                generator_loss = -critic_output_fake.mean()

                generator_loss.backward()
                generator_optimizer.step()

            # Output training stats
            if batch_idx % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"[{epoch:>2}/{hp.num_epochs}][{iters:>7}][{elapsed_time:8.2f}s]\t"
                      f"d_loss/g_loss: {critic_loss.item():4.2}/{generator_loss.item():4.2}\t")

            # Save Losses for plotting later
            generator_losses.append(generator_loss.item())
            critic_losses.append(critic_loss.item())

            iters += 1

    torch.save(generator.state_dict(), "../model_parameters/water/distance_prediction.sav")

    plt.title("Generator and critic losses during training")
    plt.plot(generator_losses, label="G")
    plt.plot(critic_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
