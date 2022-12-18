import torch.distributions

from cgan import *


@dataclass
class EnergyChildHyperparameters(CGANHyperparameters):
    batchsize: int = 128
    num_epochs: int = 1
    noise_size: int = 1
    n_critic: int = 5
    gp_lambda: float = 10.

    en_c_noise = torch.distributions.Exponential(1.)

    def generate_noise(self, n, device="cuda"):
        return self.en_c_noise.sample((n, self.noise_size)).to(device=device)


class EnergyChildGenerator(CGANGenerator):
    def __init__(self, hp: EnergyChildHyperparameters, mean_x: float, std_x: float, mean_y: float, std_y: float):
        super().__init__(hp, mean_x, std_x, mean_y, std_y)

    def define_model(self):
        return nn.Sequential(
            nn.Linear(self._get_noise_size() + 2, 128),  # input: (noise, dist_p, en_p)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),  # output: (en_c)
            nn.Tanh()
        )

    @staticmethod
    def load(model_path: str, data_stats_path: str) -> CGANGenerator:
        return load(EnergyChildGenerator, EnergyChildHyperparameters(), model_path, data_stats_path)

    def _extract_relevant_info(self, p: Particle, *args) -> list[float]:
        dist = args[0][0]
        return [dist, p.ene]


class EnergyChildCritic(CGANCritic):
    def __init__(self):
        super().__init__()

    def define_model(self):
        return nn.Sequential(
            nn.Linear(3, 128),  # input: (en_c, dist_p, en_p)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )


if __name__ == "__main__":
    dataset = ParticlesDataset(columns_x=["en_c"],
                               columns_y=["dist_p", "en_p"],
                               emission_only=True)
    hp = EnergyChildHyperparameters()
    gen = EnergyChildGenerator(hp, *dataset_to_stats(dataset))
    cri = EnergyChildCritic()
    gen_losses, cri_losses = train(gen, cri, hp, dataset)
    save(gen, dataset, "../model_parameters/water/event_prediction.sav",
         "../model_parameters/water/cos_p_prediction_dataset_stats")
    plot_training_losses(gen_losses, cri_losses)
