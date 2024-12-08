import os
import time

from models import hmr, SMPL, build_model
from utils import BaseTrainer
import config
import constants
from datasets import BaseDataset_MutilROI as BaseDataset, MixedDataset
import jittor
from jittor import nn
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation, cam_crop2full
from eval import run_evaluation, run_evaluation_mutilROI
from jittor import Var
import numpy as np
from .loss import SetCriterion
import cv2

jittor.flags.use_cuda = 1


class Trainer(BaseTrainer):
    def init_fn(self):
        """
        init dataset,model,
        :return:
        """
        if self.options.train_dataset is not None:
            self.train_ds = BaseDataset(self.options, self.options.train_dataset, is_train=True,
                                        bbox_type=self.options.bbox_type)
        else:
            self.train_ds = MixedDataset(self.options, is_train=True)

        self.eval_ds = BaseDataset(self.options, self.options.eval_dataset, is_train=False,
                                   bbox_type=self.options.bbox_type)

        self.model = build_model(config.SMPL_MEAN_PARAMS, pretrained=True,
                                 backbone=self.options.backbone,
                                 model_name=self.options.model_name,
                                 option=self.options)
        # print(self.model)
        self.optimizer = jittor.optim.Adam(params=self.model.parameters(),
                                           lr=self.options.lr)
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.options.batch_size,
                         create_transl=False)

        self.joints_idx = 25
        self.joints_num = 49
        self.n_views = self.options.n_views

        weight_dict = {
            "loss_keypoint_2d": self.options.keypoint_loss_weight,
            "loss_keypoint_3d": self.options.keypoint_loss_weight,
            "loss_shape": self.options.shape_loss_weight,
            "loss_pose": self.options.pose_loss_weight,
            "loss_beta": self.options.beta_loss_weight,
            "loss_con": self.options.con_loss_weight,
            "loss_camera": self.options.cam_loss_weight
        }
        self.criterion = SetCriterion(weight_dict=weight_dict, model_name="mutil_roi")
        # self.criterion_keypoints = nn.MSELoss(reduction='none')
        # self.criterion_shape = nn.L1Loss()
        # self.criterion_regr = nn.MSELoss()
        # self.focal_length = constants.FOCAL_LENGTH
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}

        self.bbox_type = self.options.bbox_type
        if self.bbox_type == 'square':
            print(">>>>>>>>>>training using square bbox")
            self.crop_w = constants.IMG_RES
            self.crop_h = constants.IMG_RES
            self.bbox_size = constants.IMG_RES
        elif self.bbox_type == 'rect':
            print(">>>>>>>>>>training rect bbox")
            self.crop_w = constants.IMG_W
            self.crop_h = constants.IMG_H
            self.bbox_size = constants.IMG_H
        if self.options.use_extraviews:
            print(">>>>>>>>>>training using extra view")

        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

    def train_step(self, input_batch):
        self.model.train()
        n_views = self.n_views
        # img base info
        img_names = input_batch['img_name']  # [b,]
        dataset = input_batch['dataset_name']  # [b,]
        batch_size = len(img_names)

        images = input_batch['img']  # [b,3,h,w]
        image_extras = input_batch['img_extras']  # [b,(v-1)*3,h,w]
        images = jittor.concat([images, image_extras], 1)  # [b,v,h,w]

        # crop bbox 2D gt keypoint
        gt_keypoints_2d = input_batch['keypoints'][:self.joints_num]  # [B,49,3]
        gt_keypoints_2d_extra = input_batch['keypoint_2d_extras'][:, :, :self.joints_num]  # [b,v-1,49,3]
        gt_keypoints_2d = jittor.concat([gt_keypoints_2d.unsqueeze(1), gt_keypoints_2d_extra], 1).view(
            batch_size * n_views, self.joints_num, 3)  # [b,v,49,3]

        # origin full 2d gt keypoint
        gt_keypoints_2d_full = input_batch['keypoints_full']  # [B,49,3]
        gt_joints = input_batch['pose_3d'].unsqueeze(1).repeat(1, n_views, 1, 1) \
            .view(batch_size * n_views, self.joints_num - self.joints_idx, 4)  # [b*v,24,4]
        # [B,72]=>[B,5,72]
        gt_pose = input_batch['pose'].unsqueeze(1).repeat(1, n_views, 1) \
            .view(batch_size * n_views, -1)  # [B*n_views,72]
        gt_betas = input_batch['betas'].unsqueeze(1).repeat(1, n_views, 1) \
                       .view(batch_size * n_views, -1)[:, :10]  # [B*n_views,10]
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        # 3d flag
        has_smpl = input_batch['has_smpl'].unsqueeze(1).repeat(1, n_views) \
            .view(batch_size * n_views, )  # [B*n_views,]
        has_pose_3d = input_batch['has_pose_3d'].unsqueeze(1).repeat(1, n_views) \
            .view(batch_size * n_views, )  # [B*n_views,]

        # data augmentation info
        is_flipped = input_batch['is_flipped']  # flag that indicates whether image was flipped during data augmentation
        crop_trans = input_batch['crop_trans']  # [b,2,3]
        crop_trans_extras = input_batch['crop_trans_extras']  # [b,v-1,2,3]
        crop_trans = jittor.concat([crop_trans.unsqueeze(1), crop_trans_extras], 1).view(batch_size * n_views, 2,
                                                                                         3)  # [b*v,2,3]
        rot_angle = input_batch['rot_angle']  # rotation angle used for data augmentation

        bbox_info = input_batch['bbox_info']  # []
        bbox_info_extras = input_batch['bbox_extras']  # []
        bbox_info = jittor.concat([bbox_info, bbox_info_extras], 1)

        center, scale, = input_batch['center'], input_batch['scale'].squeeze(),
        center_extras = input_batch['center_extras']
        scale_extras = input_batch['scale_extras']
        temp = jittor.unsqueeze(center, 1)
        center = jittor.concat([temp, center_extras], 1) \
            .view(batch_size * n_views, 2)  # [B*n_views,2]
        temp = jittor.unsqueeze(scale, 1)
        scale = jittor.concat([temp, scale_extras], 1) \
            .view(batch_size * n_views, )  # [B*n_views,2]
        focal_length = input_batch['focal_length'].repeat(1, n_views) \
            .view(batch_size * n_views, ).float()  # [B*n_views,]

        shifts_extra = input_batch['shift_extras']
        rescales_extra = input_batch['rescale_extras']

        # De-normalize 2D keypoints from [-1,1] to crop pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, 0] = 0.5 * self.crop_w * (gt_keypoints_2d_orig[:, :, 0] + 1)
        gt_keypoints_2d_orig[:, :, 1] = 0.5 * self.crop_h * (gt_keypoints_2d_orig[:, :, 1] + 1)

        pred_rotmat, pred_betas, pred_camera, xf_g = self.model(images, bbox_info)

        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints[:, :self.joints_num]

        # print(temp.shape)
        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_crop = jittor.stack([pred_camera[:, 1],
                                      pred_camera[:, 2],
                                      2 * 5000. / (self.bbox_size * pred_camera[:, 0] + 1e-9)], dim=-1)  # [B,3]
        img_w = input_batch['img_w'].unsqueeze(1).repeat(1, n_views).view(-1, 1)  # [B*n_views,1]
        img_h = input_batch['img_h'].unsqueeze(1).repeat(1, n_views).view(-1, 1)  # [B*n_views,1]
        full_img_shape = jittor.concat((img_h, img_w), dim=1)  # [B*n_views,2]

        pred_cam_full = cam_crop2full(pred_camera, center, scale,
                                      full_img_shape=full_img_shape, focal_length=focal_length)


        camera_center = jittor.concat((img_w, img_h), dim=1) / 2  # [b*v,2]
        camera_center = jittor.float32(camera_center)

        pred_keypoint_2d_full = perspective_projection(pred_joints,
                                                       rotation=jittor.float32(
                                                           jittor.init.eye(3).unsqueeze(0).expand(batch_size * n_views,
                                                                                                  -1, -1)),
                                                       translation=pred_cam_full,
                                                       focal_length=focal_length,
                                                       camera_center=camera_center)  # [B*n_views,49,2]
        # print(pred_keypoint_2d_full)
        pred_keypoint_2d_full_with_conf = jittor.concat(
            (pred_keypoint_2d_full, jittor.ones(batch_size * n_views, 49, 1)), dim=2)  # [B*n_views,49,3]
        # trans @ pred_keypoint_2d_full_homo
        # [b*v,2,3] [b*v,49,3]
        pred_keypoint_2d_bbox = jittor.linalg.einsum('bij,bkj->bki', crop_trans,
                                                     pred_keypoint_2d_full_with_conf)  # [b*v,49,2]

        # [b,b] => [b,b,1] => [b,b*v,v] => [b*v,b*v]
        label_mask = jittor.init.eye(batch_size).unsqueeze(-1) \
            .repeat(1, self.n_views, self.n_views).view(batch_size * self.n_views, batch_size * self.n_views)

        predict = {
            # "pred_keypoint_2d": pred_keypoints_2d_crop,
            "pred_rotmat": pred_rotmat,
            "pred_betas": pred_betas,
            "pred_camera": pred_camera,
            "pred_output": pred_output,
            "pred_keypoint_2d_bbox": pred_keypoint_2d_bbox,
            "global_feature": xf_g,
            "label_mask": label_mask
        }
        gt = {
            "gt_keypoint_2d": gt_keypoints_2d,
            "gt_pose": gt_pose,
            "gt_betas": gt_betas,
            "gt_joints": gt_joints,
            # "gt_camera": gt_camera,
            "gt_output": gt_out,
        }
        const = {
            "openpose_train_weight": self.options.openpose_train_weight,
            "gt_train_weight": self.options.gt_train_weight,
            "has_pose_3d": has_pose_3d,
            "has_smpl": has_smpl,
            "crop_w": self.crop_w,
            "crop_h": self.crop_h,
            "center": center,
            "scale": scale,
            "full_img_shape": full_img_shape
        }

        loss, losses = self.criterion(predict, gt, const)

        loss *= 60

        start = time.time()
        self.optimizer.zero_grad()
        # self.optimizer.backward(loss)
        self.optimizer.step(loss)
        end = time.time()
        print(f'optimizer time {end - start:.4f}s')
        losses = {'loss': losses['loss'].detach().item(),
                  'loss_keypoints_2d': losses['loss_keypoints_2d'].detach().item(),
                  'loss_keypoints_3d': losses['loss_keypoints_3d'].detach().item(),
                  'loss_regr_pose': losses['loss_regr_pose'].detach().item(),
                  'loss_regr_betas': losses['loss_regr_betas'].detach().item(),
                  'loss_shape': losses['loss_shape'].detach().item(),
                  'loss_con': losses['loss_con'].detach().item(),
                  'loss_camera': losses['loss_camera'].detach().item()}
        print(losses)

        output = {
            'pred_vertices': pred_vertices.detach(),
            'pred_cam_t': pred_cam_crop.detach()
        }

        return output, losses

    def test(self):
        self.model.eval()

        mpjpe_3dpw, pa_mpjpe_3dpw, pve_3dpw = run_evaluation_mutilROI(self.model, "mutilROI", '3dpw', self.eval_ds,
                                                                      result_file=None,
                                                                      batch_size=48,
                                                                      shuffle=False,
                                                                      log_freq=32)
        results = {
            'mpjpe': mpjpe_3dpw,
            'pa_mpjpe': pa_mpjpe_3dpw,
            'pve': pve_3dpw
        }
        return results

    def train_summaries(self, input_batch, output, losses):

        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)
