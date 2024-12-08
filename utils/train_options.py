import os
import json
import argparse
import numpy as np
from collections import namedtuple
from datetime import datetime


class TrainOptions:

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', help='Name of the experiment')
        req.add_argument('--fixname', default=False, action='store_true',
                           help='Ignore GT 3D data (for unpaired experiments')
        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=np.inf,
                         help='Total time to run in seconds. Used for training in environments with timing constraints')
        gen.add_argument('--resume', dest='resume', default=False, action='store_true',
                         help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=0, help='Number of processes used for data loading')

        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false')
        gen.set_defaults(pin_memory=True)

        io = self.parser.add_argument_group('io')
        io.add_argument('--log_dir', default='logs', help='Directory to store logs')
        io.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        io.add_argument('--from_json', default=None, help='Load options from json file instead of the command line')
        io.add_argument('--pretrained_checkpoint', default=None,
                        help='Load a pretrained checkpoint at the beginning training')

        train = self.parser.add_argument_group('Training Options')

        train.add_argument('--backbone', type=str, default="resnet", help='', choices=['resnet', 'hrnet', ])
        train.add_argument('--model_name', type=str, default="cliff", help='', choices=['hmr', 'cliff', 'mutilROI'])
        train.add_argument('--train_dataset', type=str, help='')
        train.add_argument('--eval_dataset', type=str, default='3dpw', help='')
        train.add_argument('--num_epochs', type=int, default=250, help='Total number of training epochs')
        train.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
        train.add_argument('--batch_size', type=int, default=16, help='Batch size')
        train.add_argument('--summary_steps', type=int, default=1, help='Summary saving frequency')
        train.add_argument('--test_steps', type=int, default=420, help='Testing frequency during training')
        train.add_argument('--checkpoint_steps', type=int, default=500, help='Checkpoint saving frequency')
        train.add_argument('--save_epochs', type=int, default=1, help='Checkpoint saving frequency')
        train.add_argument('--test_epochs', type=int, default=1, help='Testing frequency during training')
        train.add_argument('--start_test_epoch', type=int, default=5, help='Time to start test during training')
        train.add_argument('--img_res', type=int, default=224,
                           help='Rescale bounding boxes to size [img_res, img_res] before feeding them in the network')
        # mutilROI
        train.add_argument('--n_views', type=int, default=5, help='Views to use')
        train.add_argument('--use_extraviews', default=True, action='store_true',
                           help='Use parallel stages for regression')
        train.add_argument('--rescale_bbx', default=True, action='store_true',
                           help='Use rescaled bbox for consistency data aug and loss')
        train.add_argument('--shift_center', default=True, action='store_true',
                           help='Use shifted center for consistency data aug and loss')

        train.add_argument('--rot_factor', type=float, default=30,
                           help='Random rotation in the range [-rot_factor, rot_factor]')
        train.add_argument('--noise_factor', type=float, default=0.4,
                           help='Randomly multiply pixel values with factor in the range [1-noise_factor, 1+noise_factor]')
        train.add_argument('--scale_factor', type=float, default=0.25,
                           help='Rescale bounding boxes by a factor of [1-scale_factor,1+scale_factor]')
        train.add_argument('--ignore_3d', default=False, action='store_true',
                           help='Ignore GT 3D data (for unpaired experiments')
        train.add_argument('--viz_debug', default=False, action='store_true', help='Use visualization for debugging')
        # loss weight
        train.add_argument('--shape_loss_weight', default=1, type=float, help='Weight of per-vertex loss')
        train.add_argument('--keypoint_loss_weight', default=5., type=float, help='Weight of 2D and 3D keypoint loss')
        train.add_argument('--pose_loss_weight', default=1., type=float, help='Weight of SMPL pose loss')
        train.add_argument('--beta_loss_weight', default=0.001, type=float, help='Weight of SMPL betas loss')
        train.add_argument('--openpose_train_weight', default=0., help='Weight for OpenPose keypoints during training')
        train.add_argument('--gt_train_weight', default=1., help='Weight for GT keypoints during training')
        train.add_argument('--cam_loss_weight', type=float, default=1.,
                           help='Weight for loss cam-consistency during training')
        train.add_argument('--con_loss_weight', type=float, default=0.1,
                           help='Weight for loss_contrast during training')
        # optimization
        train.add_argument('--run_smplify', default=False, action='store_true', help='Run SMPLify during training')
        train.add_argument('--smplify_threshold', type=float, default=100.,
                           help='Threshold for ignoring SMPLify fits during training')
        train.add_argument('--num_smplify_iters', default=100, type=int, help='Number of SMPLify iterations')

        train.add_argument('--bbox_type', default='square',
                           help='Use square bounding boxes in 224x224 or rectangle ones in 256x192')
        train.add_argument('--is_pos_enc', default=True, action="store_true", help='using relative encodings')
        train.add_argument('--is_fuse', default=True, action="store_true", help='using fusion module')

        train.add_argument('--use_aug_trans', default=False, action="store_true", help='flag for using augment')
        train.add_argument('--use_aug_img', default=False, action="store_true", help='flag for using augment')
        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true',
                                   help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false',
                                   help='Don\'t shuffle training data')
        shuffle_train.set_defaults(shuffle_train=False)

        return

    def parse_args(self):
        """Parse input arguments."""
        self.args = self.parser.parse_args()
        # If config file is passed, override all arguments with the values from the config file
        now = datetime.now()
        if not self.args.fixname:
            if self.args.train_dataset is not None:
                self.args.name = f'{self.args.model_name}_{self.args.train_dataset}_{now.month}{now.day}_{now.hour}_{now.minute}'
            else:
                self.args.name = f'{self.args.model_name}_mixDataset_{now.month}{now.day}_{now.hour}_{now.minute}'
        if self.args.from_json is not None:
            path_to_json = os.path.abspath(self.args.from_json)
            with open(path_to_json, "r") as f:
                json_args = json.load(f)
                json_args = namedtuple("json_args", json_args.keys())(**json_args)
                return json_args
        else:
            self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
            self.args.summary_dir = os.path.join(self.args.log_dir, 'tensorboard')
            if not os.path.exists(self.args.log_dir):
                os.makedirs(self.args.log_dir)
            self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
            if not os.path.exists(self.args.checkpoint_dir):
                os.makedirs(self.args.checkpoint_dir)
            self.save_dump()
            return self.args

    def save_dump(self):
        """
        Store all argument values to a json file.
        The default location is logs/expname/config.json.
        """
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        return
