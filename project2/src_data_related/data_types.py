import enum

import numpy as np
import random
from dataclasses import dataclass


class Type(enum.Enum):
    photon = 0
    electron = 1
    positron = 2
    proton = 4
    neutron = 5


class Arena:
    def __init__(self, wx: float, wy: float, wz: float, nx: int, ny: int, nz: int):
        self.M = np.zeros((nx, ny, nz))
        self.voxel = np.array((wx / nx, wy / ny, wz / nz))

    def Deposit(self, pos, ene: float):
        idx = (pos / self.voxel).astype(int)
        try:
            self.M[idx[0], idx[1], idx[2]] += ene
        except:
            return False  # if a deposit fails, it means we're out of the arena
        return True


class Particle:
    def __init__(self, x: float, y: float, z: float, dx: float, dy: float, dz: float, e: float, t: Type,
                 is_primary: bool = False):
        self.pos = np.array((x, y, z), dtype=float)
        self.dir = np.array((dx, dy, dz), dtype=float)
        self.ene = e
        self.type = t
        self.is_primary = is_primary

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


@dataclass
class Event:
    parent_particle: Particle
    distance: float
    delta_e: float
    cos_theta: float
    child_particle: Particle | None

    def to_entry(self) -> dict:
        c = self.child_particle
        p = self.parent_particle  # shorthand


        entry = {
            'x_p': p.pos[0], 'y_p': p.pos[1], 'z_p': p.pos[2],
            'dx_p': p.dir[0], 'dy_p': p.dir[1], 'dz_p': p.dir[2],
            'en_p': p.ene,

            'dist_p': self.distance,
            'de_p': self.delta_e,
            'cos_p': self.cos_theta,
        }

        if c is None:
            augmented_entry = {
                'emission': 0,
                'x_c': np.NaN, 'y_c': np.NaN, 'z_c': np.NaN,
                'dx_c': np.NaN, 'dy_c': np.NaN, 'dz_c': np.NaN,
                'en_c': np.NaN, 'cos_c': np.NaN
            }
        else:
            augmented_entry = {
                'emission': 1,
                'x_c': c.pos[0], 'y_c': c.pos[1], 'z_c': c.pos[2],
                'dx_c': c.dir[0], 'dy_c': c.dir[1], 'dz_c': c.dir[2],
                'en_c': c.ene, 'cos_c': (p.dir @ c.dir) / (np.linalg.norm(p.dir) * np.linalg.norm(c.dir))
            }

        entry.update(augmented_entry)
        return entry

    @staticmethod
    def child_columns():
        return ['emission', 'x_c', 'y_c', 'z_c', 'dx_c', 'dy_c', 'dz_c', 'en_c', 'cos_c']
