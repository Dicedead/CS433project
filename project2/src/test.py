import sys
import math
import pickle
import queue

import matplotlib.pyplot as plt
import torch

from sklearn.preprocessing import PolynomialFeatures

from data_types import *
from emission_event_gan import *


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


def GetEvent(P: Particle):
    distance = predict_distance(P)
    emission = predict_emission(P, distance)

    if emission:
        de_p, cos_p, en_c, cos_c = emission_event_prediction(P, distance)
        child_particle = Particle.create_child(P, distance, en_c, cos_c)

    else:
        de_p = 0.
        cos_p = no_emission_event_prediction(P, distance)
        child_particle = None

    return distance, de_p, cos_p, child_particle


def predict_distance(p: Particle):
    pass


def predict_emission(p: Particle, distance: float):
    def log_reg_features(dist_p, en_p):
        polyfeat = PolynomialFeatures(degree=4)
        x = np.array([dist_p, en_p])
        return np.concatenate([x, polyfeat.fit_transform(x), np.exp(-en_p), np.exp(-dist_p)])

    return clf_logreg.predict(log_reg_features(distance, p.ene))


def emission_event_prediction(p: Particle, distance: float):
    with torch.no_grad():
        pred = event_emission_model(torch.randn(4), torch.from_numpy(np.array([distance, p.ene])))  # TODO put noise size in model attribute

    pred = pred.detach().cpu().numpy()
    return pred


with open('../model_parameters/water/emission_prediction.sav', "lb") as f:
    clf_logreg = pickle.load(f)

event_emission_model = EmissionEventGenerator()
event_emission_model.load_state_dict(torch.load('../model_parameters/water/event_prediction.sav'))
event_emission_model.eval()

WX, WY, WZ = 300, 200, 200  # 300x200x200 mm
NX, NY, NZ = 150, 100, 100  # for 2x2x2 mm voxels
A = Arena(WX, WY, WZ, NX, NY, NZ)
NMC = 100000  # number of particles we want to simulate
EMAX = 20.0  # maximum energy of particles

ToSimulate = queue.Queue()  # the queue of particles
for i in range(NMC):
    # create NMC particles at x=0, y=WY/2, z=WZ/2, going in the X direction (1,0,0) and with a gamma distributed energy
    s = random.gauss(0.0, 0.1)
    ypos = (1 + s) * WY / 2
    ydir = 0.2 * s
    s = random.gauss(0.0, 0.1)
    zpos = (1 + s) * WZ / 2
    zdir = 0.2 * s
    xdir = math.sqrt(1 - ydir ** 2 - zdir ** 2)
    e = EMAX * random.gammavariate(2.0, 0.667)
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
        distance, delta_e, cos_theta, generated_particle = GetEventTest(P)
        P.Move(distance)
        P.Lose(delta_e, A)
        P.Rotate(cos_theta)
        if generated_particle is not None:
            ToSimulate.put(generated_particle)
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
axes[0].set_title("log(Dose) in X/Z plane")
axes[1].plot(A.M[:, int(NY / 2), int(NZ / 2)])
axes[1].set_title("dose along white line")
axes[2].plot(A.M[5, :, int(NZ / 2)])
axes[2].set_title("dose across green line")
plt.show()
