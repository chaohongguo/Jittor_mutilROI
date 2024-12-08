from typing import Tuple, List
from jittor import Var
import jittor
import jittor.nn as nn


def find_dynamic_lmk_idx_and_bcoords(
        vertices: Var,
        pose: Var,
        dynamic_lmk_faces_idx: Var,
        dynamic_lmk_b_coords: Var,
        neck_kin_chain: List[int],
        pose2rot: bool = True,
) -> Tuple[Var, Var]:
    pass


def vertices2joints(J_regressor: Var, vertices: Var) -> Var:
    """
       Calculates the 3D joint locations from the vertices
    :param J_regressor:[J,V]
    :param vertices:[B,V,3]
    :return: The location of the joints: [B,J,3]
    """
    return jittor.linalg.einsum('bik,ji->bjk', vertices, J_regressor)


def blend_shapes(betas: Var, shape_disps: Var) -> Var:
    """
        Calculates the per vertex displacement due to the blend shapes
    :param betas: [B,num_betas]
    :param shape_disps: [V,3,num_betas]
    :return:
        The per-vertex displacement due to shape deformation
        blend_shape:[B,V,3]
    """
    blend_shape = jittor.linalg.einsum('bl,mkl->bmk', betas, shape_disps)
    return blend_shape


def batch_rodrigues(rot_vecs: Var, epsilon: float = 1e-8) -> Var:
    """
        Calculates the rotation matrices for a batch of rotation vectors
        axis-angle vectors ===> rotate matrix
    :param rot_vecs:[N,3]
        N is the num of all joints,such as N=B*J
        B is model batch_size
        J is the number of joints of one people
    :param epsilon
    :return:R:[N,3,3]
    """

    batch_size = rot_vecs.shape[0]
    dtype = rot_vecs.dtype

    angle = jittor.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = jittor.unsqueeze(jittor.cos(angle), dim=1)
    sin = jittor.unsqueeze(jittor.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = jittor.split(rot_dir, 1, dim=1)
    K = jittor.zeros((batch_size, 3, 3), dtype=dtype)

    zeros = jittor.zeros((batch_size, 1), dtype=dtype)
    K = jittor.concat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = jittor.init.eye(3, dtype=dtype).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * jittor.bmm(K, K)
    return rot_mat


def transform_mat(R: Var, t: Var) -> Var:
    """ Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    """
    # No padding left or right, only add an extra row
    return jittor.concat([nn.pad(R, [0, 0, 0, 1]),
                          nn.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def lbs(
        betas: Var,
        pose: Var,
        v_template: Var,
        shapedirs: Var,
        posedirs: Var,
        J_regressor: Var,
        parents: Var,
        lbs_weights: Var,
        pose2rot: bool = True,
) -> Tuple[Var, Var]:
    """
        Performs Linear Blend Skinning with the given shape and pose parameters
    Args:
        betas:
        pose:
        v_template:
        shapedirs:
        posedirs:
        J_regressor:
        parents:
        lbs_weights:
        pose2rot:

    Returns:

    """
    batch_size = max(betas.shape[0], pose.shape[0])
    # device = betas.dtype
    dtype = betas.dtype
    # 2.add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    # joints
    J = vertices2joints(J_regressor, v_shaped)  # [B,J,3]
    # 3.add pose blend shapes
    ident = jittor.init.eye(3, dtype=dtype)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)) \
            .view([batch_size, -1, 3, 3])  # [B,J+1,3,3]
        pose_feature = (rot_mats[:, 1:, :, :] - ident) \
            .view([batch_size, -1])  # [B,J*3*3]
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = jittor.matmul(pose_feature, posedirs).view(batch_size, -1, 3)  # [B,V,3]
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = jittor.matmul(pose_feature.view(batch_size, -1), posedirs) \
            .view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 4. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = jittor.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = jittor.ones([batch_size, v_posed.shape[1], 1], dtype=dtype)
    v_posed_homo = jittor.concat([v_posed, homogen_coord], dim=2)
    v_homo = jittor.matmul(T, jittor.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


def batch_rigid_transform(
        rot_mats: Var,
        joints: Var,
        parents: Var,
        dtype=jittor.float32
) -> Var:
    """

    :param rot_mats: [B,J,3,3]
    :param joints: [B,J,3]
        Locations of joints
    :param parents:
    :param dtype:
    :return:
    """
    joints = jittor.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)
    # transforms_mat = jittor.float32(transforms_mat)
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = jittor.matmul(transform_chain[parents[i]], transforms_mat[:, i])

        transform_chain.append(curr_res)
    transforms = jittor.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = nn.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - nn.pad(
        jittor.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms
