import jittor
import numpy as np
from .smplx.lbs import vertices2joints
from .smplx import SMPL as _SMPL
from .smplx.body_models import SMPLOutput
import config
import constants


class SMPL(_SMPL):
    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        self.J_regressor_extra = jittor.array(np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA))
        self.joint_map = jittor.int32(joints)  # default is int64

    def execute(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).execute(*args, **kwargs)
        # 9 extra joints
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = jittor.concat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = SMPLOutput(vertices=smpl_output.vertices,
                            global_orient=smpl_output.global_orient,
                            body_pose=smpl_output.body_pose,
                            joints=joints,
                            betas=smpl_output.betas,
                            full_pose=smpl_output.full_pose)
        return output




