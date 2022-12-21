import torch.distributions

from cgan import *


@dataclass
class EnergyChildHyperparameters(CGANHyperparameters):
    batchsize: int = 64
    num_epochs: int = 1
    noise_size: int = 1
    n_critic: int = 5
    gp_lambda: float = 10.

    en_c_noise = torch.distributions.Exponential(1.)

    scaling_mean = 1.515
    scaling_std = 5.75 / 0.45
    max_iters = 15

    def generate_noise(self, n, device="cuda"):
        return self.en_c_noise.sample((n, self.noise_size)).to(device=device)


class EnergyChildGenerator(CGANGenerator):
    def __init__(self, hp: EnergyChildHyperparameters, mean_x: float, std_x: float, mean_y: float, std_y: float):
        super().__init__(hp, mean_x, std_x, mean_y, std_y)
        self.__hp = hp

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

    def predict(
            self,
            p: Particle,
            distance: float,
            device="cpu"
    ):
        it = 0
        while it < self.__hp.max_iters:
            raw = self.generate_from_particle(p, distance, device=device)[0, 0]
            pred = self.__hp.scaling_std * (raw - self.__hp.scaling_mean)
            it += 1
            if pred >= 0:
                return pred
        return 0.

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
    save(gen, dataset, "../model_parameters/water/ene_c_prediction.sav",
         "../model_parameters/water/ene_c_prediction_dataset_stats")
    plot_training_losses(gen_losses, cri_losses)
