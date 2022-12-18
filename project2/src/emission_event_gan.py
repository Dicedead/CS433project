import torch.distributions

from cgan import *


@dataclass
class EmissionEventHyperparameters(CGANHyperparameters):
    batchsize: int = 128
    num_epochs: int = 1
    noise_size: int = 3
    n_critic: int = 5
    gp_lambda: float = 10.

    cos_p_noise = torch.distributions.Kumaraswamy(0.5, 0.5)
    en_c_noise = torch.distributions.Exponential(1.)
    cos_c_noise = torch.distributions.Exponential(0.5)

    def generate_noise(self, n, device="cuda"):
        shape = (n, 1)
        return torch.cat([
            self.cos_p_noise.sample(shape),
            self.en_c_noise.sample(shape),
            self.cos_c_noise.sample(shape)
        ], dim=-1).to(device=device)


class EmissionEventGenerator(CGANGenerator):
    def __init__(self, hp: EmissionEventHyperparameters, mean_x: float, std_x: float, mean_y: float, std_y: float):
        super().__init__(hp, mean_x, std_x, mean_y, std_y)

    def define_model(self):
        return nn.Sequential(
            nn.Linear(self._get_noise_size() + 2, 128),  # input: (noise, dist_p, en_p)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 3),  # output: (cos_p, cos_c, en_c)
            nn.Tanh()
        )

    @staticmethod
    def load(model_path: str, data_stats_path: str) -> CGANGenerator:
        return load(EmissionEventGenerator, EmissionEventHyperparameters(), model_path, data_stats_path)

    def _extract_relevant_info(self, p: Particle, *args) -> list[float]:
        return [args[0][0], p.ene]


class EmissionEventCritic(CGANCritic):
    def __init__(self):
        super().__init__()

    def define_model(self):
        return nn.Sequential(
            nn.Linear(5, 128),  # input: (cos_p, cos_c, en_c, dist_p, en_p)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )


if __name__ == "__main__":
    dataset = ParticlesDataset(columns_x=["cos_p", "en_c", "cos_c"],
                               columns_y=["dist_p", "en_p"],
                               emission_only=True)
    hp = EmissionEventHyperparameters()
    gen = EmissionEventGenerator(hp, *dataset_to_stats(dataset))
    cri = EmissionEventCritic()
    gen_losses, cri_losses = train(gen, cri, hp, dataset)
    save(gen, dataset, "../model_parameters/water/event_prediction.sav",
         "../model_parameters/water/event_prediction_dataset_stats")
    plot_training_losses(gen_losses, cri_losses)
