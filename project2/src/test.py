import sys
import math
import pickle
import queue

import pandas as pd
import matplotlib.pyplot as plt

import emission_prediction
import no_emission_cos_p_prediction

from data_types import *
from de_p_generation import generate_de_p
from distance_gan import DistanceGenerator
from cos_parent_gan import CosParentGenerator
from ene_child_gan import EnergyChildGenerator
from cos_child_gan import CosChildGenerator


def CoreEvent(max_dist: float, max_dele: float, max_cos: float, prob: float):
    s = random.random()  # simple event generator for testing
    distance = max_dist * s
    delta_e = max_dele * s
    cos_theta = max_cos * (random.random() - 0.5)
    if s > prob:
        delta_e *= 0.5
        Q = Particle(P.pos[0], P.pos[1], P.pos[2], P.dir[0], P.dir[1], P.dir[2], delta_e, Type.electron)
        Q.Rotate(0.5)
        return distance, delta_e, cos_theta, Q
    else:
        return distance, delta_e, cos_theta, None


def GetEventTest(P: Particle):
    """this is the function you are responsible for creating, given a particle it returns:
    distance:	distance the particle travels before having this event
    delta_e:	amount of energy that the particle loses during this event
    cos_theta:	cosine of the angle that the particle rotates by
    Q:		a particle generated during this event, or None
    """
    if P.type == Type.photon:
        return CoreEvent(5.0, 0.5, 0.05, 0.99)
    elif P.type == Type.electron:
        return CoreEvent(1.0, 0.2, 0.1, 0.75)


def GetEvent(P: Particle, device="cpu"):
    for m in [distance_model, en_c_model, cos_p_model, cos_c_model]:
        m.to(device)

    distance = distance_model.predict(P, device=device)
    emission = emission_prediction.predict_emission(clf_logreg, P, distance)

    if emission:
        en_c = en_c_model.predict(P, distance, device=device)
        cos_p = cos_p_model.predict(P, distance, en_c, device=device)
        cos_c = cos_c_model.predict(P, distance, en_c, cos_p, device=device)
        de_p = generate_de_p(de_p_mean, de_p_std)
        child_particle = P.create_child(distance, en_c, cos_c)

    else:
        de_p = 0.
        cos_p = no_emission_cos_p_prediction.predict_no_emission_cos_p(P, distance)
        child_particle = None

    return distance, de_p, cos_p, child_particle


if __name__ == "__main__":

    with open('../model_parameters/water/emission_prediction.sav', "rb") as f:
        clf_logreg = pickle.load(f)
    water_dataset = pd.read_pickle("../pickled_data/water_dataset.pkl")
    cos_p_model = CosParentGenerator.load("../model_parameters/water/cos_p_prediction.sav",
                                          "../model_parameters/water/cos_p_prediction_dataset_stats")
    en_c_model = EnergyChildGenerator.load("../model_parameters/water/ene_c_prediction.sav",
                                           "../model_parameters/water/ene_c_prediction_dataset_stats")
    cos_c_model = CosChildGenerator.load("../model_parameters/water/cos_c_prediction.sav",
                                         "../model_parameters/water/cos_c_prediction_dataset_stats")
    distance_model = DistanceGenerator.load("../model_parameters/water/distance_prediction.sav",
                                            "../model_parameters/water/distance_prediction_dataset_stats")
    de_p_mean, de_p_std = pd.read_pickle("../model_parameters/water/de_p_generation_dataset_stats/mean.pkl")["de_p"], \
        pd.read_pickle("../model_parameters/water/de_p_generation_dataset_stats/std.pkl")["de_p"]

    WX, WY, WZ = 300, 200, 200  # 300x200x200 mm
    NX, NY, NZ = 150, 100, 100  # for 2x2x2 mm voxels
    A = Arena(WX, WY, WZ, NX, NY, NZ)
    NMC = 100_000  # number of particles we want to simulate
    EMAX = 6.0  # maximum energy of particles

    ToSimulate = queue.Queue()  # the queue of particles
    for i in range(NMC):
        # create NMC particles at x=0, y=WY/2, z=WZ/2, going in the X direction (1,0,0)
        # and with a gamma distributed energy
        s = random.gauss(0.0, 0.1)
        ypos = (1 + s) * WY / 2
        ydir = 0.2 * s
        s = random.gauss(0.0, 0.1)
        zpos = (1 + s) * WZ / 2
        zdir = 0.2 * s
        xdir = math.sqrt(1 - ydir ** 2 - zdir ** 2)
        e = EMAX * random.random()
        P = Particle(0.0, ypos, zpos, xdir, ydir, zdir, e, Type.photon, True)
        ToSimulate.put(P)

    DONE = 0  # count number of primary particles
    NALL = 0  # count total number of particles
    while not ToSimulate.empty() > 0:
        P = ToSimulate.get()
        if P.is_primary:
            DONE += 1
        NALL += 1
        while P.ene > 0.0:
            distance, delta_e, cos_theta, generated_particle = GetEvent(P)
            P.Move(distance)
            P.Lose(delta_e, A)
            P.Rotate(cos_theta)
            if generated_particle is not None:
                pass
                # Drop generated electrons
                # ToSimulate.put(generated_particle)
        if DONE % 1000:
            sys.stdout.write(f"Finished {DONE:8} out of {NMC:8} {(100.0 * DONE) / NMC:.2f} %\r")
            sys.stdout.flush()
    sys.stdout.write(f"\nFinished with {NMC:8} primaries and a total of {NALL:8} particles\n")
    sys.stdout.flush()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    CNTR = axes[0].contourf(np.log10(A.M.sum(axis=1) + 0.001), cmap="inferno")
    axes[0].axvline(NY / 2, ls=":", color="white")
    axes[0].axhline(5, ls=":", color="green")
    plt.colorbar(CNTR, aspect=60)
    axes[0].set_title("log(dose) in X/Z plane")
    axes[1].plot(A.M[:, int(NY / 2), int(NZ / 2)])
    axes[1].set_title("dose along white line")
    axes[2].plot(A.M[5, :, int(NZ / 2)])
    axes[2].set_title("dose across green line")
    plt.show()
