import jittor
import jittor.nn as nn
import numpy as np
from jittor import Var
from typing import Optional, Dict, Union
from .utils import Struct, to_np, to_var, SMPLOutput
from .utils import Array
from .utils import find_joint_kin_chain
from .vertex_joint_selector import VertexJointSelector
from .vertex_ids import vertex_ids as VERTEX_IDS
from .lbs import lbs, blend_shapes
import os
import pickle


class SMPL(nn.Module):

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    SHAPE_SPACE_DIM = 300

    def __init__(
            self, model_path: str,
            kid_template_path: str = '',
            data_struct: Optional[Struct] = None,
            create_betas: bool = True,
            betas: Optional[Var] = None,
            num_betas: int = 10,
            create_global_orient: bool = True,
            global_orient: Optional[Var] = None,
            create_body_pose: bool = True,
            body_pose: Optional[Var] = None,
            create_transl: bool = True,
            transl: Optional[Var] = None,
            dtype=jittor.float32,
            batch_size: int = 1,
            joint_mapper=None,
            gender: str = 'neutral',
            age: str = 'adult',
            vertex_ids: Dict[str, int] = None,
            v_template: Optional[Union[Var, Array]] = None,
            **kwargs
    ) -> None:
        """
        Args:
            model_path:str
                The path to the folder or to the file where the model parameters are stored
            kid_template_path:
                The path to the folder
            data_struct:str
                A struct object. something like the follwing
                    the struct of XXX.pkl
                        shapedir:[6890,3,10]
                        posedirs:[6890,3,207]
                        J:[24,3]
                        f:[13776,3]
                        kintree_table:[2,24]
                        ...
                load pkl file to get more

            create_betas: bool, optional, (default = True)
                Flag for creating a member variable for the shape space
            betas: jittor.Var, optional, [B,10], (default = None)
                The default value for the shape member variable.
            num_betas: int, optional, (default = 10).
                Number of shape components to use.

            create_global_orient: bool, optional, (default = True)
                Flag for creating a member variable for the global orientation of the body.
            global_orient: jittor.Var, optional, [B,3]
                The default value for the global orientation variable.

            create_body_pose: bool, optional, (default = True)
                Flag for creating a member variable for the pose of the body.
            body_pose: jittor.Var, optional, [B,(J-1)*3]
                       J is the number of all joints
                       J-1 is the number of body joints
                The default value for the body pose variable.

            create_transl: bool, optional, (default = True)
                Flag for creating a member variable for the translation of the body.
            transl: jittor.Var, optional, [B,3]
                The default value for the transl variable.

            dtype: jittor.float32
            batch_size: int, optional
            joint_mapper: object, optional
            gender: str, optional
            age: str, optional
            vertex_ids:
            v_template:
            **kwargs:
        """
        self.gender = gender
        self.age = age
        if data_struct is None:

            if os.path.isdir(model_path):
                model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
                smpl_path = os.path.join(model_path, model_fn)
            else:
                smpl_path = model_path
            assert os.path.exists(smpl_path), 'Path {} does not exist!'.format(
                smpl_path)

            with open(smpl_path, 'rb') as smpl_file:
                data_struct = Struct(**pickle.load(smpl_file, encoding='latin1'))

        super(SMPL, self).__init__()
        self.batch_size = batch_size
        shapedirs = data_struct.shapedirs
        if shapedirs.shape[-1] < self.SHAPE_SPACE_DIM:
            print(f'WARNING: You are using a {self.name()} model, with only'
                  f'{shapedirs.shape[-1]} shape coefficients.\n'
                  f'num_betas = {num_betas}')
            num_betas = min(num_betas, 10)
        else:
            num_betas = min(num_betas, self.SHAPE_SPACE_DIM)

        self._num_betas = num_betas
        self.shapedirs = jittor.array(to_np(shapedirs[:, :, :num_betas]))
        if vertex_ids is None:
            vertex_ids = VERTEX_IDS['smplh']

        self.dtype = dtype
        self.joint_mapper = joint_mapper
        self.vertex_joint_selector = VertexJointSelector(vertex_ids=vertex_ids, **kwargs)
        self.faces = data_struct.f

        if create_betas:
            if betas is None:
                _default_betas = jittor.zeros(
                    [batch_size, self.num_betas], dtype=dtype)  # [B,num_betas=10]
            else:
                if jittor.is_var(betas):
                    _default_betas = betas.clone().detach()
                else:
                    _default_betas = jittor.array(betas, dtype=dtype)
            self.betas = nn.Parameter(_default_betas, requires_grad=True)

        if create_global_orient:
            if global_orient is None:
                _default_global_orient = jittor.zeros([batch_size, 3], dtype=dtype)
            else:
                if jittor.is_var(global_orient):
                    _default_global_orient = global_orient.clone().detach()
                else:
                    _default_global_orient = jittor.array(global_orient, dtype=dtype)
            # self.global_orient = nn.Parameter(_default_global_orient, requires_grad=True)
            self.global_orient = _default_global_orient

        if create_body_pose:
            if body_pose is None:
                _default_body_pose = jittor.zeros([batch_size, self.NUM_BODY_JOINTS * 3], dtype=dtype)
            else:
                if jittor.is_var(body_pose):
                    _default_body_pose = body_pose.clone().detach()
                else:
                    _default_body_pose = jittor.array(body_pose, dtype=dtype)
            # self.body_pose = nn.Parameter(_default_body_pose, requires_grad=True)
            self.body_pose = _default_body_pose

        if create_transl:
            if transl is None:
                _default_transl = jittor.zeros([batch_size, 3], dtype=dtype)
            else:
                _default_transl = jittor.array(transl, dtype=dtype)
            # self.transl = nn.Parameter(_default_body_pose, requires_grad=True)
            # self.transl = _default_transl

        if v_template is None:
            self.v_template = data_struct.v_template
        if not jittor.is_var(v_template):
            self.v_template = to_var(self.v_template)

        self.J_regressor = to_var(to_np(data_struct.J_regressor), dtype=dtype)

        num_pose_basis = data_struct.posedirs.shape[-1]
        self.posedirs = to_var(to_np(np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T))

        self.parents = to_var(to_np(data_struct.kintree_table[0])).long()
        self.parents[0] = -1

        self.lbs_weights = to_var(to_np(data_struct.weights), dtype=dtype)

    @property
    def num_betas(self):
        return self._num_betas

    @property
    def num_expression_coeffs(self):
        return 0

    def create_mean_pose(self, data_struct) -> Var:
        pass

    def name(self) -> str:
        return 'SMPL'

    def forward_shape(
            self,
            betas: Optional[Var] = None,
    ) -> SMPLOutput:
        betas = betas if betas is not None else self.betas
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        return SMPLOutput(vertices=v_shaped, betas=betas, v_shaped=v_shaped)

    def execute(
            self,
            betas: Optional[Var] = None,
            body_pose: Optional[Var] = None,
            global_orient: Optional[Var] = None,
            transl: Optional[Var] = None,
            return_verts=True,
            return_full_pose: bool = False,
            pose2rot: bool = True,
            **kwargs
    ) -> SMPLOutput:
        """
        Args:
            betas: jittor.Var, Optional, [B,num_betas]
            body_pose: jittor.Var, Optional, [B,(J-1)*3]
            global_orient: jittor.Var, Optional, [B,3]
            transl:jittor.Var, Optional, [B,3]
            return_verts: bool, Optional, default=True
            return_full_pose: bool, Optional, default=True
            pose2rot: bool
            **kwargs:

        Returns:

        """
        global_orient = (global_orient if global_orient is not None else self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas
        full_pose = jittor.concat([global_orient, body_pose], dim=1)

        apply_tranl = transl is not None or hasattr(self, 'transl')
        if transl is None and hasattr(self, 'tranl'):
            transl = self.transl

        batch_size = max(betas.shape[0], global_orient.shape[0], body_pose.shape[0])

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.expand(num_repeats, -1)

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot)

        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        output = SMPLOutput(vertices=vertices if return_verts else None,
                            global_orient=global_orient,
                            body_pose=body_pose,
                            joints=joints,
                            betas=betas,
                            full_pose=full_pose if return_full_pose else None)

        return output


def create(
    model_path: str,
    model_type: str = 'smpl',
    **kwargs
) -> Union[SMPL]:

    if os.path.join.isdir(model_path):
        model_path = os.path.join(model_path, model_type)
    else:
        model_type = os.path.join.basename(model_path).split('_')[0].lower()

    if model_type.lower() == 'smpl':
        return SMPL(model_path, **kwargs)

    else:
        raise ValueError(f'Unknown model type {model_type}, exiting!')