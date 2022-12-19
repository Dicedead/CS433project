from cgan import *


@dataclass
class NoEmissionCosParentHyperparameters(CGANHyperparameters):
    batchsize: int = 64
    num_epochs: int = 1
    noise_size: int = 1
    n_critic: int = 5
    gp_lambda: float = 10.

    no_em_cos_p_noise = torch.distributions.Pareto(1., 3.)

    def generate_noise(self, n, device="cuda"):
        return self.no_em_cos_p_noise.sample((n, self.noise_size)).to(device=device)


class NoEmissionCosParentGenerator(CGANGenerator):
    def __init__(self, hp: NoEmissionCosParentHyperparameters, mean_x: float, std_x: float, mean_y: float, std_y: float):
        super().__init__(hp, mean_x, std_x, mean_y, std_y)

    def define_model(self):
        return nn.Sequential(
            nn.Linear(self._get_noise_size() + 2, 128),  # input: (noise, dist_p, en_p)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),  # output: (cos_p)
            nn.Tanh()
        )

    @staticmethod
    def load(model_path: str, data_stats_path: str) -> CGANGenerator:
        return load(NoEmissionCosParentGenerator, NoEmissionCosParentHyperparameters(), model_path, data_stats_path)

    def _extract_relevant_info(self, p: Particle, *args) -> list[float]:
        dist = args[0][0]
        return [dist, p.ene]


class NoEmissionCosParentCritic(CGANCritic):
    def __init__(self):
        super().__init__()

    def define_model(self):
        return nn.Sequential(
            nn.Linear(3, 128),  # input: (cos_p, dist_p, en_p)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )


if __name__ == "__main__":
    dataset = ParticlesDataset(columns_x=["cos_p"],
                               columns_y=["dist_p", "en_p"],
                               emission_only=True)
    hp = NoEmissionCosParentHyperparameters()
    gen = NoEmissionCosParentGenerator(hp, *dataset_to_stats(dataset))
    cri = NoEmissionCosParentCritic()
    gen_losses, cri_losses = train(gen, cri, hp, dataset)
    save(gen, dataset, "../model_parameters/water/no_em_cos_p_prediction.sav",
         "../model_parameters/water/no_em_cos_p_prediction_dataset_stats")
    plot_training_losses(gen_losses, cri_losses)
