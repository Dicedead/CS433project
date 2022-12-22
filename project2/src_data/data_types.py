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
        self.dir = self.__gen_dir(cos_angle)

    def __gen_dir(self, cos_angle):
        uniform_val = np.random.uniform(-1, 1)
        bound = np.sqrt(1 - uniform_val ** 2)
        remaining_val = np.random.uniform(-bound, bound)

        while True:
            special_coord = np.random.randint(0, 3)
            if self.dir[special_coord] != 0:
                coords_without_special = [0, 1, 2]
                coords_without_special.remove(special_coord)
                uniform_coord = np.random.choice(coords_without_special)
                coords_without_special.remove(uniform_coord)
                remaining_coord = coords_without_special[0]
                break

        special_value = cos_angle - self.dir[uniform_coord] * uniform_val - self.dir[remaining_coord] * remaining_val
        new_dir = [-1, -1, -1]
        new_dir[uniform_coord] = uniform_val
        new_dir[remaining_coord] = remaining_val
        new_dir[special_coord] = special_value

        new_dir = np.array(new_dir)
        new_dir = new_dir / np.linalg.norm(new_dir)

        return new_dir

    def create_child(self, dist_p: float, en_p: float, cos_c: float):
        c_dir = self.__gen_dir(cos_c)
        new_pos = self.pos + dist_p * self.dir
        return Particle(
            new_pos[0], new_pos[1], new_pos[2],
            c_dir[0], c_dir[1], c_dir[2],
            en_p, Type.electron
        )


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
