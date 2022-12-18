from cgan import *


@dataclass
class DistanceHyperparameters(CGANHyperparameters):
    batchsize: int = 64
    num_epochs: int = 1
    noise_size: int = 1
    n_critic: int = 5
    gp_lambda: float = 10.

    noise_distrib = torch.distributions.Exponential(1)

    def generate_noise(self, n, device="cuda"):
        return self.noise_distrib.sample((n, self.noise_size)).to(device=device)


class DistanceGenerator(CGANGenerator):
    def __init__(self, hp: DistanceHyperparameters, mean_x: float, std_x: float, mean_y: float, std_y: float):
        super().__init__(hp, mean_x, std_x, mean_y, std_y)
        self.__hp = hp

    def define_model(self):
        return nn.Sequential(
            nn.Linear(self._get_noise_size() + 1, 64), # input: (noise , en_p)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 128), # output: (dist_p)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    @staticmethod
    def load(model_path: str, data_stats_path: str) -> CGANGenerator:
        return load(DistanceGenerator, DistanceHyperparameters(), model_path, data_stats_path)

    def _extract_relevant_info(self, p: Particle, *args) -> list[float]:
        return [p.ene]


class DistanceCritic(CGANCritic):
    def __init__(self):
        super().__init__()

    def define_model(self):
        return nn.Sequential(
            nn.Linear(2, 32), # input: (dist_p, en_p)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )


if __name__ == "__main__":
    dataset = ParticlesDataset(columns_x=["dist_p"],
                               columns_y=["en_p"],
                               emission_only=False)
    hp = DistanceHyperparameters()
    gen = DistanceGenerator(hp, *dataset_to_stats(dataset))
    cri = DistanceCritic()
    gen_losses, cri_losses = train(gen, cri, hp, dataset)
    save(gen, dataset, "../model_parameters/water/distance_prediction.sav",
         "../model_parameters/water/distance_prediction_dataset_stats")
    plot_training_losses(gen_losses, cri_losses)
