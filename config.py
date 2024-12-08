from os.path import join

# some npz
DATASET_NPZ_PATH = 'data/dataset_extras'
DATASET_NPZ_PATH_ = 'data/dataset_extras'
DATASET_NPZ_PATH__ = 'data/dataset_extras'

DATASET_FILES = [
    # test.npz
    {
        '3dpw': join(DATASET_NPZ_PATH__, '3dpw_test_w2d_smpl3d_gender.npz'),
    },

    # train.npz
    {
        # 'h36m': join(DATASET_NPZ_PATH, 'h36m_train.npz'),
        'h36m': join(DATASET_NPZ_PATH__, 'h36m_mosh_train_fixname.npz'),
        'lsp-orig': join(DATASET_NPZ_PATH_, 'lsp_dataset_original_train.npz'),
        'mpii': join(DATASET_NPZ_PATH_, 'mpii_train.npz'),
        'coco': join(DATASET_NPZ_PATH__, 'coco_2014_smpl_train.npz'),
        'lspet': join(DATASET_NPZ_PATH_, 'hr-lspet_train.npz'),
        'mpi-inf-3dhp': join(DATASET_NPZ_PATH__, 'mpi_inf_3dhp_train_name_revised.npz'),
        '3dpw': join(DATASET_NPZ_PATH__, '3dpw_train_w2d_smpl3d_gender.npz'),
    }
]

# the position of raw data
PW3D_ROOT = '{your_data_path}/3DPW'
H36M_ROOT = '{your_data_path}/h36m'
LSP_ROOT = '{your_data_path}/lsp_dataset_original'
LSP_ORIGINAL_ROOT = '{your_data_path}/lsp_dataset'
LSPET_ROOT = '{your_data_path}/hr_lspet'
MPII_ROOT = '{your_data_path}/mpii_human_pose_v1'
COCO_ROOT = '{your_data_path}/coco'
MPI_INF_3DHP_ROOT = '{your_data_path}/mpii3d'
AGORA_ROOT = '{your_data_path}/agora'
H36M_ROOT = '{your_data_path}/h36m'

DATASET_FOLDERS = {
    '3dpw': PW3D_ROOT,
    'mpii': MPII_ROOT,
    'coco': COCO_ROOT,
    'lsp-orig': LSP_ORIGINAL_ROOT,
    'lspet': LSPET_ROOT,
    'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,

    'h36m': H36M_ROOT,
    'h36m-p1': H36M_ROOT,
    'h36m-p2': H36M_ROOT,
}

#
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'
