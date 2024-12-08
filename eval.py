import jittor
import numpy as np
import cv2
import os
import argparse
import config
from models import SMPL, hmr, build_model
from models.cliff.cliff_resnet50 import CLIFF_resnet50
from utils import CheckpointDataLoader
from datasets import BaseDataset_MutilROI as BaseDataset
from tqdm import tqdm
import constants
from utils.pose_utils import reconstruction_error

jittor.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',
                    default="data/jittor_best.pkl",
                    help='Path to network checkpoint')
parser.add_argument('--dataset', default='3dpw', choices=['h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp'],
                    help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=16, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=32, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')
parser.add_argument('--model_name', default="mutilROI", help='If set, save detections to a .npz file')
parser.add_argument('--n_views', type=int, default=5, help='Views to use')
parser.add_argument('--is_pos_enc', default=True, action="store_true", help='using relative encodings')
parser.add_argument('--is_fuse', default=True, action="store_true", help='using fusion module')
parser.add_argument('--use_extraviews', default=True, action='store_true',
                    help='Use parallel stages for regression')
parser.add_argument('--shift_center', default=True, action='store_true', help='Shuffle data')
parser.add_argument('--rescale_bbx', default=True, action='store_true', help='Shuffle data')
parser.add_argument('--noise_factor', type=float, default=0.4,
                    help='Randomly multiply pixel values with factor in the range [1-noise_factor, 1+noise_factor]')
parser.add_argument('--rot_factor', type=float, default=30,
                    help='Random rotation in the range [-rot_factor, rot_factor]')
parser.add_argument('--scale_factor', type=float, default=0.25,
                    help='Rescale bounding boxes by a factor of [1-scale_factor,1+scale_factor]')
parser.add_argument('--bbox_type', default='rect',
                    help='Use square bounding boxes in 224x224 or rectangle ones in 256x192')


def run_evaluation(model, model_name, dataset_name, dataset, result_file,
                   batch_size=32, img_res=224,
                   num_workers=32, shuffle=False, log_freq=8):
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False)
    smpl_male = SMPL(config.SMPL_MODEL_DIR, gender='male', create_transl=False)
    smpl_female = SMPL(config.SMPL_MODEL_DIR, gender='female', create_transl=False)
    save_results = result_file is not None

    J_regressor = jittor.array(np.load(config.JOINT_REGRESSOR_H36M), dtype=jittor.float32)
    data_loader = CheckpointDataLoader(dataset=dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers)

    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    pve = np.zeros(len(dataset))

    eval_pose = False
    eval_masks = False
    eval_parts = False

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp':
        eval_pose = True
    elif dataset_name == 'lsp':
        eval_masks = True
        eval_parts = True
        annot_path = config.DATASET_FOLDERS['upi-s1h']

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14

    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader) // batch_size)):
        # Get ground truth annotations from the batch
        gt_pose = batch['pose']
        gt_betas = batch['betas']
        gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
        images = batch['img']
        gender = batch['gender']
        bbox = batch['bbox_info']
        curr_batch_size = images.shape[0]

        with jittor.no_grad():
            if model_name == 'cliff':
                pred_rotmat, pred_betas, pred_camera = model(images, bbox)
            elif model_name == 'hmr':
                pred_rotmat, pred_betas, pred_camera = model(images)
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                       global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

        if eval_pose:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1)
            # Get 14 ground truth joints
            if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
                gt_keypoints_3d = batch['pose_3d']
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            # For 3DPW get the 14 common joints from the rendered shape
            else:
                gt_vertices = smpl_male(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:], betas=gt_betas).vertices
                gt_vertices_female = smpl_female(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:],
                                                 betas=gt_betas).vertices
                gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]
                gt_keypoints_3d = jittor.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = jittor.matmul(J_regressor_batch, pred_vertices)
            if save_results:
                pred_joints[step * batch_size:step * batch_size + curr_batch_size, :,
                :] = pred_keypoints_3d.cpu().numpy()
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

            # Absolute error (MPJPE)
            error = jittor.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).numpy()
            # 32 * step: 32 * step + 32
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.numpy(),
                                           reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

            # pve
            per_vertex_error = jittor.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(
                dim=-1).numpy()
            pve[step * batch_size:step * batch_size + curr_batch_size] = per_vertex_error

            # Print intermediate results during evaluation
            if step % log_freq == log_freq - 1:
                if eval_pose:
                    print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                    print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
                    print('PVE: ' + str(1000 * pve[:step * batch_size].mean()))
                    print()

    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
        # Print final results during evaluation
    print('*** Final Results ***')
    print()
    if eval_pose:
        print('MPJPE: ' + str(1000 * mpjpe.mean()))
        print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
        print('PVE: ' + str(1000 * pve.mean()))
        print()
    return 1000 * mpjpe.mean(), 1000 * recon_err.mean(), 1000 * pve.mean()


def run_evaluation_mutilROI(model, model_name, dataset_name, dataset, result_file,
                            batch_size=32, img_res=224,
                            num_workers=32, shuffle=False, log_freq=8,
                            use_extra=True, use_fuse=True, n_views=5, ):
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False)
    smpl_male = SMPL(config.SMPL_MODEL_DIR, gender='male', create_transl=False)
    smpl_female = SMPL(config.SMPL_MODEL_DIR, gender='female', create_transl=False)
    save_results = result_file is not None

    J_regressor = jittor.array(np.load(config.JOINT_REGRESSOR_H36M), dtype=jittor.float32)
    data_loader = CheckpointDataLoader(dataset=dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers)

    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    pve = np.zeros(len(dataset))

    eval_pose = False
    eval_masks = False
    eval_parts = False

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp':
        eval_pose = True
    elif dataset_name == 'lsp':
        eval_masks = True
        eval_parts = True
        annot_path = config.DATASET_FOLDERS['upi-s1h']

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14

    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader) // batch_size)):
        # Get ground truth annotations from the batch
        gt_pose = batch['pose']
        gt_betas = batch['betas']
        gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
        images = batch['img']  # [b,3,h,w]
        gender = batch['gender']
        bbox_info = batch['bbox_info']  # [b,3]
        curr_batch_size = images.shape[0]
        if use_extra:
            bboxs_extras = batch['bbox_extras']  # [b,(v-1)*3]
            img_extras = batch['img_extras']  # [b,(v-1)*3,h,w]
            images = jittor.concat([images, img_extras], 1)  # [b,v*3,h,w]
            bbox_info = jittor.concat([bbox_info, bboxs_extras], 1)  # [b,v*3]

        index = 0
        with jittor.no_grad():
            if model_name == 'cliff':
                pred_rotmat, pred_betas, pred_camera = model(images, bbox_info)
            elif model_name == 'hmr':
                pred_rotmat, pred_betas, pred_camera = model(images)
            elif model_name == 'mutilROI':
                pred_rotmat, pred_betas, pred_camera, _1 = model(images, bbox_info)
                if use_fuse:
                    pred_rotmat = pred_rotmat.view(curr_batch_size, n_views, 24, 3, 3)[:, index]
                    pred_betas = pred_betas.view(curr_batch_size, n_views, -1)[:, index]
                    pred_camera = pred_camera.view(curr_batch_size, n_views, -1)[:, index]
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                       global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

        if eval_pose:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1)
            # Get 14 ground truth joints
            if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
                gt_keypoints_3d = batch['pose_3d']
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            # For 3DPW get the 14 common joints from the rendered shape
            else:
                gt_vertices = smpl_male(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:], betas=gt_betas).vertices
                gt_vertices_female = smpl_female(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:],
                                                 betas=gt_betas).vertices
                gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]
                gt_keypoints_3d = jittor.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = jittor.matmul(J_regressor_batch, pred_vertices)
            if save_results:
                pred_joints[step * batch_size:step * batch_size + curr_batch_size, :,
                :] = pred_keypoints_3d.cpu().numpy()
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

            # Absolute error (MPJPE)
            error = jittor.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).numpy()
            # 32 * step: 32 * step + 32
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.numpy(),
                                           reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

            # pve
            per_vertex_error = jittor.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(
                dim=-1).numpy()
            pve[step * batch_size:step * batch_size + curr_batch_size] = per_vertex_error

            # Print intermediate results during evaluation
            if step % log_freq == log_freq - 1:
                if eval_pose:
                    print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                    print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
                    print('PVE: ' + str(1000 * pve[:step * batch_size].mean()))
                    print()

    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
        # Print final results during evaluation
    print('*** Final Results ***')
    print()
    if eval_pose:
        print('MPJPE: ' + str(1000 * mpjpe.mean()))
        print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
        print('PVE: ' + str(1000 * pve.mean()))
        print()
    return 1000 * mpjpe.mean(), 1000 * recon_err.mean(), 1000 * pve.mean()


if __name__ == '__main__':
    args = parser.parse_args()
    model = build_model(config.SMPL_MEAN_PARAMS, model_name=args.model_name, option=args)
    checkpoint = jittor.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Setup evaluation dataset
    dataset = BaseDataset(args, args.dataset, is_train=False)
    if args.model_name == 'mutilROI':
        run_evaluation_mutilROI(model, args.model_name, args.dataset, dataset,
                                result_file=None,
                                batch_size=40,
                                shuffle=False,
                                log_freq=8)
    else:
        # Run evaluation
        run_evaluation(model, args.model_name, args.dataset, dataset, args.result_file,
                       batch_size=args.batch_size,
                       shuffle=args.shuffle,
                       log_freq=args.log_freq)
