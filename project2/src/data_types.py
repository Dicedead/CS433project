import numpy as np
import random


class Arena:
    def __init__(self, wx: float, wy: float, wz: float, nx: int, ny: int, nz: int):
        self.M = np.zeros((nx, ny, nz))
        self.voxel = np.array((wx / (nx + 1), wy / (ny + 1), wz / (nz + 1)))

    def Deposit(self, pos, ene: float):
        idx = (pos / self.voxel).astype(int)
        try:
            self.M[idx[0], idx[1], idx[2]] += ene
        except:
            return False  # if a deposit fails, it means we're out of the arena
        return True


class Particle:
    def __init__(self, x: float, y: float, z: float, dx: float, dy: float, dz: float, e: float,
                 is_primary: bool = False):
        self.pos = np.array((x, y, z), dtype=float)
        self.dir = np.array((dx, dy, dz), dtype=float)
        self.ene = e
        self.is_primary = is_primary

    def __lt__(self, Q):
        return P.ene < Q.ene

    def Lose(self, energy: float, phantom: Arena):
        energy = min(energy, self.ene)  # lose this much energy and deposit it in the arena
        self.ene -= energy
        if not phantom.Deposit(self.pos, energy):
            self.ene = 0.0  # if a deposit fails, it means we're out of the arena, so kill the particle

    def Move(self, distance: float):
        self.pos += distance * self.dir

    def Rotate(self, cos_angle: float):
        s = cos_angle * random.random()  # approximate version
        self.dir[1] += s
        self.dir[2] += (cos_angle - s)
        self.dir /= np.linalg.norm(self.dir)
