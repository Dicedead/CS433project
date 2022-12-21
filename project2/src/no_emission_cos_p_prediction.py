import numpy as np
import torch.distributions

from data_types import Particle


def predict_no_emission_cos_p(
        p : Particle,
        distance: float,
        scale=1/80
):
    return np.clip(1-np.abs(np.random.laplace(0, np.minimum(scale, 1/(1 + p.ene * distance)))), -1, 1)

