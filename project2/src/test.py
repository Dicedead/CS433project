import heapq
import matplotlib.pyplot as plt
import sys
from data_types import *


def GetEvent(P: Particle):
    """this is the function you are responsible for creating, given a particle, it returns:
    distance:	distance the particle travels before having this event
    delta_e:	amount of energy that the particle loses during this event
    cos_theta:	cosine of the angle that the particle rotates by
    Q:		a particle generated during this event, or None
    """
    s = 5.0 * random.random()
    distance = s
    delta_e = 0.1 * s
    cos_theta = 0.1 * (random.random() - 0.5)
    if s > 4.5:
        delta_e *= 0.5
        Q = Particle(P.pos[0], P.pos[1], P.pos[2], P.dir[0], P.dir[1], P.dir[2], delta_e)
        Q.Rotate(0.5)
        return distance, delta_e, cos_theta, Q
    else:
        return distance, delta_e, cos_theta, None


WX, WY, WZ = 200, 200, 200  # 200x200x200 mm
NX, NY, NZ = 200, 200, 200  # 200x200x200 voxels
A = Arena(WX, WY, WZ, NX, NY, NZ)  # 1x1x1 mm voxels
NMC = 100000  # number of particles we want to simulate
EMAX = 10.0  # maximum energy of particles

ToSimulate = list()  # the list/heap of particles
for i in range(NMC):
    # create NMC particles at x=0, y=WY/2, z=WZ/2, going in the X direction (1,0,0) and with a random energy
    P = Particle(0.0, WY / 2, WZ / 2, 1, 0, 0, EMAX * random.random(), True)
    heapq.heappush(ToSimulate, P)

DONE = 0  # count number of primary particles
NALL = 0  # count total number of particles
while len(ToSimulate) > 0:
    P = heapq.heappop(ToSimulate)
    if P.is_primary:
        DONE += 1
    NALL += 1
    while P.ene > 0.0:
        distance, delta_e, cos_theta, generated_particle = GetEvent(P)
        P.Lose(delta_e, A)
        P.Move(distance)
        P.Rotate(cos_theta)
        if generated_particle is not None:
            heapq.heappush(ToSimulate, generated_particle)
    if DONE % 1000:
        sys.stdout.write(f"Finished {DONE:8} out of {NMC:8} {(100.0 * DONE) / NMC:.2f} %\r")
        sys.stdout.flush()
sys.stdout.write(f"\nFinished with {NMC:8} primaries and a total of {NALL:8} particles")
sys.stdout.flush()
plt.contourf(np.log10(A.M.sum(axis=1) + 0.001), cmap="inferno")
plt.colorbar()
plt.show()
