import time

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from abc import abstractmethod
from dataclasses import dataclass
from data_types import Particle
from torch.utils.data import Dataset
from torch import nn, optim, autograd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class ParticlesDataset(Dataset):

    def __init__(self,
                 columns_x: list[str],
                 columns_y: list[str],
                 emission_only: bool,
                 path="../pickled_data/water_dataset.pkl",
                 validation_size=0.3
                 ):
        dataset = pd.read_pickle(path)
        if emission_only:
            dataset = dataset[dataset["emission"] == 1]

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        x = dataset[columns_x]
        y = dataset[columns_y]

        self.mean_x = x.mean()
        self.std_x = x.std()

        self.mean_y = y.mean()
        self.std_y = y.std()

        x = x_scaler.fit_transform(x)
        y = y_scaler.fit_transform(y)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_size)

        self.x_train = torch.from_numpy(x_train).float()
        self.y_train = torch.from_numpy(y_train).float()
        self.x_val = x_val
        self.y_val = y_val

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


@dataclass
class CGANHyperparameters:
    batchsize: int
    num_epochs: int
    noise_size: int
    n_critic: int
    gp_lambda: float

    opti_lr_gen: float = 1e-4
    opti_betas_gen: tuple[float, float] = (0, 0.9)

    opti_lr_cri: float = 1e-4
    opti_betas_cri: tuple[float, float] = (0, 0.9)

    def generate_noise(self, n, device="cuda"):
        return torch.randn((n, self.noise_size), device=device)


class CGANGenerator(nn.Module):
    def __init__(self, hp: CGANHyperparameters, mean_x: float, std_x: float, mean_y: float, std_y: float):
        super().__init__()
        self.__noise_size = hp.noise_size
        self.__hp = hp
        self.model = self.define_model()
        self.__mean_x = mean_x
        self.__mean_y = mean_y
        self.__std_x = std_x
        self.__std_y = std_y

    def forward(self, noise, labels):
        out = self.model(torch.cat([noise, labels], dim=-1))
        return out

    def generate_from_particle(self, p: Particle, *args):
        with torch.no_grad():
            pred = self.__unstandardize_x(
                self(
                    self.__hp.generate_noise(1, device="cpu"),
                    torch.from_numpy(
                        self.__standardize_y(np.array(self._extract_relevant_info(p, args)))
                    ).float()
                )
            )
        return pred.detach().cpu().numpy()

    def __standardize_y(self, arr: np.ndarray):
        return (arr - self.__mean_y) / self.__std_y

    def __unstandardize_x(self, arr: np.ndarray):
        return self.__std_x * arr + self.__mean_x

    def _get_noise_size(self):
        return self.__noise_size

    @abstractmethod
    def define_model(self):
        pass

    @abstractmethod
    def _extract_relevant_info(self, p: Particle, *args) -> list[float]:
        pass


class CGANCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self.define_model()

    def forward(self, event, labels):
        out = self.model(torch.cat([event, labels], dim=-1))
        return out.squeeze()

    @abstractmethod
    def define_model(self):
        pass


def dataset_to_stats(data: ParticlesDataset):
    return data.mean_x, data.std_x, data.mean_y, data.std_y


def train(
        generator: CGANGenerator,
        critic: CGANCritic,
        hp: CGANHyperparameters,
        dataset: ParticlesDataset,
        num_threads=2,
        seed=1,
        verbose=True
):
    torch.set_num_threads(num_threads)
    torch.manual_seed(seed)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=hp.batchsize, num_workers=1, shuffle=True,
                                             drop_last=True, pin_memory=True)
    critic, generator = critic.to("cuda"), generator.to("cuda")

    critic_optimizer = optim.AdamW(critic.parameters(), lr=hp.opti_lr_cri, betas=hp.opti_betas_cri)
    generator_optimizer = optim.AdamW(generator.parameters(), lr=hp.opti_lr_gen, betas=hp.opti_betas_gen)

    generator_losses, critic_losses = [], []
    iters = 0
    grad_tensor = torch.ones((hp.batchsize, 1), device="cuda")

    start_time = time.time()

    for epoch in range(hp.num_epochs):
        for batch_idx, data in enumerate(dataloader):
            real_samples, real_labels = data[0].to("cuda"), data[1].to("cuda")

            # Update critic
            critic_optimizer.zero_grad()

            critic_output_real = critic(real_samples, real_labels)
            critic_loss_real = critic_output_real.mean()

            noise = hp.generate_noise(hp.batchsize)
            with torch.no_grad():
                fake_sample = generator(noise, real_labels)
            critic_output_fake = critic(fake_sample, real_labels)
            critic_loss_fake = critic_output_fake.mean()

            alpha = torch.rand((hp.batchsize, 1), device="cuda")
            interpolates = (alpha * real_samples + (1. - alpha) * fake_sample).requires_grad_(True)
            d_interpolates = critic(interpolates, real_labels).reshape(-1, 1)
            gradients = autograd.grad(d_interpolates, interpolates, grad_tensor, create_graph=True, only_inputs=True)[0]
            gradient_penalty = hp.gp_lambda * ((gradients.view(hp.batchsize, -1).norm(dim=1) - 1.) ** 2).mean()

            critic_loss = -critic_loss_real + critic_loss_fake + gradient_penalty

            critic_loss.backward()
            critic_optimizer.step()

            if batch_idx % hp.n_critic == 0:
                # Update Generator
                generator_optimizer.zero_grad()

                fake_labels = dataset.y_train[torch.randint(high=len(dataset), size=[hp.batchsize])].to(
                    device="cuda")
                noise = hp.generate_noise(hp.batchsize)
                with torch.no_grad(): fake_sample = generator(noise, fake_labels)
                critic_output_fake = critic(fake_sample, fake_labels)
                generator_loss = -critic_output_fake.mean()

                generator_loss.backward()
                generator_optimizer.step()

            # Output training stats
            if verbose and batch_idx % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"[{epoch:>2}/{hp.num_epochs}][{iters:>7}][{elapsed_time:8.2f}s]\t"
                      f"critic loss / generator loss: {critic_loss.item():4.2}/{generator_loss.item():4.2}\t")

            # Save Losses for plotting later
            generator_losses.append(generator_loss.item())
            critic_losses.append(critic_loss.item())

            iters += 1

    return generator_losses, critic_losses


def save(generator: CGANGenerator, data: ParticlesDataset, model_path: str, data_stats_path: str) -> None:
    torch.save(generator.state_dict(), model_path)
    filenames = ["mean_x", "std_x", "mean_y", "std_y"]
    for idx, series in enumerate(list(dataset_to_stats(data))):
        series.to_pickle(f"{data_stats_path}/{filenames[idx]}.pkl")


def load(model_class: type, hp: CGANHyperparameters, model_path: str, data_stats_path: str) -> CGANGenerator:
    ls = []
    for fn in ["mean_x", "std_x", "mean_y", "std_y"]:
        ls.append(pd.read_pickle(f"{data_stats_path}/{fn}.pkl").values[0])
    distance_model = model_class(hp, *ls)
    distance_model.load_state_dict(torch.load(model_path))
    distance_model.eval()
    return distance_model


def plot_training_losses(generator_losses: list, critic_losses: list) -> None:
    plt.title("Generator and critic losses during training")
    plt.plot(generator_losses, label="G")
    plt.plot(critic_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
