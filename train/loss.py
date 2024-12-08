import jittor.nn as nn
import jittor
from utils.geometry import batch_rodrigues


class SetCriterion(nn.Module):
    def __init__(self, weight_dict, model_name=None):
        super(SetCriterion).__init__()
        self.weight = weight_dict
        self.model_name = model_name

    def loss_keypoint_2d(self, pred_keypoint_2d, gt_keypoint_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
         The loss is weighted by the confidence.
         The available keypoints are different for each dataset.
         """
        conf = gt_keypoint_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        _mse = (pred_keypoint_2d - gt_keypoint_2d[:, :, :-1]) ** 2
        loss = conf * _mse
        loss = loss.mean()
        return loss

    def loss_keypoint_3d(self, pred_keypoint_3d, gt_keypoint_3d, has_pose_3d):
        pred_keypoint_3d = pred_keypoint_3d[:, 25:, :]  # [B,24,4]
        conf = gt_keypoint_3d[:, :, -1].unsqueeze(-1).clone()  # [B,24,1]
        gt_keypoint_3d = gt_keypoint_3d[:, :, :-1].clone()  # [B,24,3]
        gt_keypoint_3d = gt_keypoint_3d[has_pose_3d == 1]  # [B,valid,3]
        conf = conf[has_pose_3d == 1]  # [B,valid,1]
        pred_keypoint_3d = pred_keypoint_3d[has_pose_3d == 1]
        if len(gt_keypoint_3d) > 0:
            gt_pelvis = (gt_keypoint_3d[:, 2, :] + gt_keypoint_3d[:, 3, :]) / 2
            gt_keypoint_3d = gt_keypoint_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoint_3d[:, 2, :] + pred_keypoint_3d[:, 3, :]) / 2
            pred_keypoint_3d = pred_keypoint_3d - pred_pelvis[:, None, :]
            _mse = ((pred_keypoint_3d - gt_keypoint_3d) ** 2)
            return (conf * _mse).mean()
        else:
            return jittor.array(0, dtype=jittor.float32)

    def loss_smpl(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        mse_loss_fn = nn.MSELoss()
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = mse_loss_fn(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = mse_loss_fn(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = jittor.array(0, dtype=jittor.float32)
            loss_regr_betas = jittor.array(0, dtype=jittor.float32)
        return loss_regr_pose, loss_regr_betas

    def loss_shape(self, pred_vertices, gt_vertices, has_smpl):
        """
           Compute per-vertex loss on the shape for the examples that SMPL annotations are available.
        """
        l1_loss_fn = nn.L1Loss()
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return l1_loss_fn(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return jittor.array(0, dtype=jittor.float32)

    def loss_keypoint_2d_cliff(self, pred_keypoint_2d_bbox, gt_keypoint_2d, crop_w, crop_h,
                               openpose_train_weight, gt_train_weight):
        """

        Args:
            self:
            pred_keypoint_2d_bbox:
            gt_keypoint_2d:
            crop_w:
            crop_h:
            openpose_train_weight:
            gt_train_weight:
        Returns:

        """
        pred_keypoint_2d_bbox[:, :, 0] = 2. * pred_keypoint_2d_bbox[:, :, 0] / crop_w - 1.
        gt_keypoint_2d[:, :, 0] = 2. * gt_keypoint_2d[:, :, 0] / crop_w - 1.
        pred_keypoint_2d_bbox[:, :, 1] = 2. * pred_keypoint_2d_bbox[:, :, 1] / crop_h - 1.
        gt_keypoint_2d[:, :, 1] = 2. * gt_keypoint_2d[:, :, 1] / crop_h - 1.

        loss = self.loss_keypoint_2d(pred_keypoint_2d_bbox, gt_keypoint_2d,
                                     openpose_train_weight,
                                     gt_train_weight)

        return loss

    def loss_supervised_contrastive(self, features, mask, temperature=0.5, scale_by_temperature=True):
        """
        Args:
            features:[b*v,2048]
            mask:[b*v,b*v]
            temperature:
            scale_by_temperature:

        Returns:

        """
        features = jittor.misc.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        temperature = jittor.array(temperature, dtype=jittor.float32)
        # compute logits
        T_features = jittor.transpose(features, (1, 0))
        anchor_dot_contrast = jittor.divide(jittor.matmul(features, T_features), temperature)  # [b*v,b*v]
        logits_max = jittor.max(anchor_dot_contrast, dim=1, keepdims=True)  # [b*v,1]
        logits = anchor_dot_contrast - logits_max.detach()  # [b*v,b*v]
        exp_logits = jittor.exp(logits)  # [b*v,b*v]
        logits_mask = jittor.ones_like(mask) - jittor.init.eye(batch_size)  # [b*v,b*v]
        positives_mask = mask * logits_mask  # []
        negatives_mask = 1. - mask
        num_positives_per_row = jittor.sum(positives_mask, dim=1)
        multi_pos = exp_logits * positives_mask
        mutil_neg = exp_logits * negatives_mask
        denominator = jittor.sum(multi_pos, dim=1, keepdims=True) + jittor.sum(mutil_neg, dim=1, keepdims=True)
        log_probs = logits - jittor.log(denominator)
        if jittor.any(jittor.misc.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        log_probs = jittor.sum(log_probs * positives_mask, dim=1)[num_positives_per_row > 0] / num_positives_per_row[
            num_positives_per_row > 0]
        loss = -log_probs
        if scale_by_temperature:
            loss *= temperature
        loss = loss.mean()
        return loss

    def loss_camera(self, pred_camera, center, scale, full_img_shape, n_views=5):
        """

        Args:

            pred_camera: [b*v,3]
            center: [b*v,2]
            scale: [b*v,]
            full_img_shape:[b*v,2]
            n_views:
        Returns:

        """
        _mse = nn.MSELoss()
        ori_batch = center.shape[0] // n_views
        img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]  # []
        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        w_2, h_2 = img_w / 2., img_h / 2.
        bs = b * pred_camera[:, 0] + 1e-9
        # [b*v,] + [b*v,]
        # [0,0,0....] + [0,0,0,....,5,5,5,....]  v*0 b*v
        main_ind = jittor.full((center.shape[0],), 0) + \
                   n_views * jittor.arange(ori_batch).unsqueeze(1).repeat(1, n_views).view(-1, )
        # [0,1,3,4,5] * b + [0*v,...b]
        views_ind = jittor.arange(n_views).unsqueeze(0).repeat(ori_batch, 1).view(-1, ) + \
                    n_views * jittor.arange(ori_batch).unsqueeze(1).repeat(1, n_views).view(-1, )
        reg_x = pred_camera[main_ind, 1] - pred_camera[views_ind, 1] - (2 * (
                (cx[views_ind] - w_2[views_ind]) / bs[views_ind] - (cx[main_ind] - w_2[main_ind]) / bs[main_ind]))
        reg_y = pred_camera[main_ind, 2] - pred_camera[views_ind, 2] - (2 * (
                (cy[views_ind] - h_2[views_ind]) / bs[views_ind] - (cy[main_ind] - h_2[main_ind]) / bs[main_ind]))
        reg_s = (bs[views_ind] - bs[main_ind]) * 1e-4
        vis_w = jittor.ones([center.shape[0]])
        loss_1 = _mse(vis_w * (reg_x + reg_y), jittor.zeros_like(reg_x))
        loss_2 = _mse(vis_w * reg_s, jittor.zeros_like(reg_s))
        return loss_2 + loss_1

    def execute(self, predict, gt, const, **kw):

        loss_keypoint_3d = self.loss_keypoint_3d(pred_keypoint_3d=predict['pred_output'].joints,
                                                 gt_keypoint_3d=gt['gt_joints'],
                                                 has_pose_3d=const['has_pose_3d'])
        # loss_keypoint_2d = self.loss_keypoint_2d(pred_keypoint_2d=predict['pred_keypoint_2d'],
        #                                          gt_keypoint_2d=gt['gt_keypoint_2d'],
        #                                          openpose_weight=const['openpose_train_weight'],
        #                                          gt_weight=const['gt_train_weight'])
        loss_shape = self.loss_shape(pred_vertices=predict['pred_output'].vertices,
                                     gt_vertices=gt['gt_output'].vertices,
                                     has_smpl=const['has_smpl'])
        loss_regr_pose, loss_regr_betas = self.loss_smpl(pred_rotmat=predict['pred_rotmat'],
                                                         pred_betas=predict['pred_betas'],
                                                         gt_pose=gt['gt_pose'], gt_betas=gt['gt_betas'],
                                                         has_smpl=const['has_smpl'])
        loss_keypoint_2d_cliff = self.loss_keypoint_2d_cliff(pred_keypoint_2d_bbox=predict['pred_keypoint_2d_bbox'],
                                                             gt_keypoint_2d=gt['gt_keypoint_2d'],
                                                             crop_w=const['crop_w'],
                                                             crop_h=const['crop_h'],
                                                             openpose_train_weight=const['openpose_train_weight'],
                                                             gt_train_weight=const['gt_train_weight'])

        if self.model_name == 'mutil_roi':
            loss_supervised_contrastive = self.loss_supervised_contrastive(features=predict['global_feature'],
                                                                           mask=predict['label_mask'])
            loss_camera = self.loss_camera(pred_camera=predict['pred_camera'],
                                           center=const['center'],
                                           scale=const['scale'],
                                           full_img_shape=const['full_img_shape'])

            loss = self.weight['loss_shape'] * loss_shape + \
                   self.weight['loss_keypoint_3d'] * loss_keypoint_3d + \
                   self.weight['loss_keypoint_2d'] * loss_keypoint_2d_cliff + \
                   self.weight['loss_pose'] * loss_regr_pose + \
                   self.weight['loss_beta'] * loss_regr_betas + \
                   self.weight['loss_con'] * loss_supervised_contrastive + \
                   self.weight['loss_camera'] * loss_camera
            # ((jittor.exp(-predict['pred_camera'][:, 0] * 10)) ** 2).mean()
            # losses = None
            # losses = {'loss': (loss*60).detach().item(),
            #           'loss_keypoints_2d': loss_keypoint_2d_cliff.detach().item(),
            #           'loss_keypoints_3d': loss_keypoint_3d.detach().item(),
            #           'loss_regr_pose': loss_regr_pose.detach().item(),
            #           'loss_regr_betas': loss_regr_betas.detach().item(),
            #           'loss_shape': loss_shape.detach().item(),
            #           'loss_con': loss_supervised_contrastive.detach().item(),
            #           'loss_camera': loss_camera.detach().item()}

            losses = {'loss': (loss * 60),
                      'loss_keypoints_2d': loss_keypoint_2d_cliff,
                      'loss_keypoints_3d': loss_keypoint_3d,
                      'loss_regr_pose': loss_regr_pose,
                      'loss_regr_betas': loss_regr_betas,
                      'loss_shape': loss_shape,
                      'loss_con': loss_supervised_contrastive,
                      'loss_camera': loss_camera}
            return loss, losses

        elif self.model_name == 'cliff':
            loss = self.weight['loss_shape'] * loss_shape + \
                   self.weight['loss_keypoint_3d'] * loss_keypoint_3d + \
                   self.weight['loss_keypoint_2d'] * loss_keypoint_2d_cliff + \
                   self.weight['loss_pose'] * loss_regr_pose + \
                   self.weight['loss_beta'] * loss_regr_betas
            # ((jittor.exp(-predict['pred_camera'][:, 0] * 10)) ** 2).mean()
            losses = None
            # losses = {'loss': loss.detach().item(),
            #           'loss_keypoints_2d': loss_keypoint_2d_cliff.detach().item(),
            #           'loss_keypoints_3d': loss_keypoint_3d.detach().item(),
            #           'loss_regr_pose': loss_regr_pose.detach().item(),
            #           'loss_regr_betas': loss_regr_betas.detach().item(),
            #           'loss_shape': loss_shape.detach().item()}
            return loss, losses
