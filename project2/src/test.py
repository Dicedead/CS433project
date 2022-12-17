import sys
import math
import pickle
import queue

from sklearn.preprocessing import PolynomialFeatures

from data_types import *
from distance_gan import DistanceGenerator
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
    distance = predict_distance(P)[0]
    emission = predict_emission(P, distance)[0]

    if emission:
        de_p, cos_p, en_c, cos_c = emission_event_prediction(P, distance)
        child_particle = P.create_child(distance, en_c, cos_c)

    else:
        de_p = 0.
        cos_p = 1 # TODO no_emission_event_prediction(P, distance)
        child_particle = None

    return distance, de_p, cos_p, child_particle


def predict_distance(p: Particle):
    return distance_model.generate_from_particle(p)


def predict_emission(p: Particle, distance: float):
    def log_reg_features(dist_p, en_p):
        polyfeat = PolynomialFeatures(degree=3)  # TODO put degree in log reg class attribute
        x = np.array([[dist_p, en_p]])
        y = polyfeat.fit_transform(x)
        z = np.exp(-np.array([[en_p, dist_p]]))
        return np.concatenate([x, y, z], axis=1)

    z = log_reg_features(distance, p.ene)
    return clf_logreg.predict(z)


def emission_event_prediction(p: Particle, distance: float):
    return event_emission_model.generate_from_particle(p, distance)


with open('../model_parameters/water/emission_prediction.sav', "rb") as f:
    clf_logreg = pickle.load(f)

water_dataset = pd.read_pickle("../pickled_data/water_dataset.pkl")
energy_levels = [i/10 for i in range(1, 61)]

event_emission_model = EmissionEventGenerator()
event_emission_model.load_state_dict(torch.load('../model_parameters/water/event_prediction.sav'))
event_emission_model.eval()

distance_model = DistanceGenerator()
distance_model.load_state_dict(torch.load('../model_parameters/water/distance_prediction.sav'))
distance_model.eval()

WX, WY, WZ = 300, 200, 200  # 300x200x200 mm
NX, NY, NZ = 150, 100, 100  # for 2x2x2 mm voxels
A = Arena(WX, WY, WZ, NX, NY, NZ)
NMC = 100000  # number of particles we want to simulate
EMAX = 6.0  # maximum energy of particles

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
        print(NALL)
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
axes[0].set_title("log(Dose) in X/Z plane")
axes[1].plot(A.M[:, int(NY / 2), int(NZ / 2)])
axes[1].set_title("dose along white line")
axes[2].plot(A.M[5, :, int(NZ / 2)])
axes[2].set_title("dose across green line")
plt.show()
