import jittor

from typing import NewType, Union, Optional
from dataclasses import dataclass, asdict, fields
from jittor import Var

import numpy as np
from numpy import ndarray as Array


@dataclass
class ModelOutput:
    vertices: Optional[Var] = None
    joints: Optional[Var] = None
    full_pose: Optional[Var] = None
    global_orient: Optional[Var] = None
    transl: Optional[Var] = None
    v_shaped: Optional[Var] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


@dataclass
class SMPLOutput(ModelOutput):
    betas: Optional[Var] = None
    body_pose: Optional[Var] = None


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def find_joint_kin_chain(joint_id, kinematic_tree):
    kin_chain = []
    curr_idx = joint_id
    while curr_idx != -1:
        kin_chain.append(curr_idx)
        curr_idx = kinematic_tree[curr_idx]
    return kin_chain


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def to_var(array: Union[Array, Var], dtype=jittor.float32) -> Var:
    if jittor.is_var(array):
        return array
    else:
        return jittor.array(array, dtype=dtype)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful with extreme cases of euler angles like [0.0, pi, 0.0]

    sy = jittor.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                     rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return jittor.atan2(-rot_mats[:, 2, 0], sy)
