# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import numpy as np
import h5py

import sys
sys.path.append('../')
from common.mpi_inf_3dhp_dataset import MpiInf3dhpDataset
from common.camera import project_to_2d, image_coordinates
from common.utils import wrap

output_filename = 'data_3d_mpi_inf_3dhp'
output_filename_2d = 'data_2d_mpi_inf_3dhp_gt'
output_filename_2d2 = 'data_2d_mpi_inf_3dhp_computed_gt'
subjects_train = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
subjects_test = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']
joint_idx_train_matlab = [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7]    # notice: it is in matlab index
joint_idx_train = [i-1 for i in joint_idx_train_matlab]

if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)
        
    parser = argparse.ArgumentParser(description='MPI_INF_3DHP dataset downloader/converter')
    
    # Convert dataset from original source, using files converted to .mat (the Human3.6M dataset path must be specified manually)
    # This option requires MATLAB to convert files using the provided script
    parser.add_argument('--from-source', default='', type=str, metavar='PATH', help='convert original dataset')
    
    args = parser.parse_args()
    
    if os.path.exists(output_filename + '.npz'):
        print('The dataset already exists at', output_filename + '.npz')
        exit(0)
                
    if args.from_source:
        print('Converting original MPI_INF_3DHP dataset from', args.from_source)
        output = {}
        output_2d_poses = {}
        from scipy.io import loadmat
        
        for subject in subjects_train:
            output[subject] = {}
            output_2d_poses[subject] = {}
            file_1 = args.from_source + '/' + subject + '/Seq1/annot.mat'
            file_2 = args.from_source + '/' + subject + '/Seq2/annot.mat'
            hf = loadmat(file_1)
            positions_3d_temp = []
            positions_2d_temp = []
            
            for index in range(14):
            	positions = hf['annot3'][index, 0].reshape(-1, 28, 3)
            	positions /= 1000 # Meters instead of millimeters
            	positions_17 = positions[:,joint_idx_train,:]
            	positions_17[:, 1:] -= positions_17[:, :1] # Remove global offset, but keep trajectory in first position
            	positions_3d_temp.append(positions_17.astype('float32'))
            	positions_2d = hf['annot2'][index, 0].reshape(-1, 28, 2)
            	positions_2d_temp.append(positions_2d[:,joint_idx_train,:].astype('float32'))
            	
            output[subject]['Seq1'] = positions_3d_temp
            output_2d_poses[subject]['Seq1'] = positions_2d_temp
            
            positions_3d_temp = []
            positions_2d_temp = []
            hf = loadmat(file_2)
            for index in range(14):
            	positions = hf['annot3'][index, 0].reshape(-1, 28, 3)
            	positions /= 1000 # Meters instead of millimeters
            	positions_17 = positions[:,joint_idx_train,:]
            	positions_17[:, 1:] -= positions_17[:, :1] # Remove global offset, but keep trajectory in first position
            	positions_3d_temp.append(positions_17.astype('float32'))
            	positions_2d = hf['annot2'][index, 0].reshape(-1, 28, 2)
            	positions_2d_temp.append(positions_2d[:,joint_idx_train,:].astype('float32'))
            output[subject]['Seq2'] = positions_3d_temp
            output_2d_poses[subject]['Seq2'] = positions_2d_temp
	    
        for subject in subjects_test:
            output[subject] = {}
            output_2d_poses[subject] = {}
            file_1 = args.from_source + '/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set/' + subject + '/annot_data.mat'
            hf = {}
            f = h5py.File(file_1)
            for k, v in f.items():
            	hf[k] = np.array(v)
            positions = hf['annot3'].reshape(-1, 17, 3)
            positions /= 1000 # Meters instead of millimeters
            positions_17 = positions
            positions_17[:, 1:] -= positions_17[:, :1] # Remove global offset, but keep trajectory in first position
            output[subject]['Test'] = [positions_17.astype('float32')]
            positions_2d = hf['annot2'].reshape(-1, 17, 2)
            output_2d_poses[subject]['Test'] = [positions_2d.astype('float32')]
        
        print('Saving...')
        np.savez_compressed(output_filename, positions_3d=output)
        print('')
        print('Getting 2D poses...')
        dataset = MpiInf3dhpDataset(output_filename + '.npz')
        metadata = {
        	'num_joints': dataset.skeleton().num_joints(),
        	'keypoints_symmetry': [dataset.skeleton().joints_left(), dataset.skeleton().joints_right()]
        }
        print('Saving...')
        np.savez_compressed(output_filename_2d, positions_2d=output_2d_poses, metadata=metadata)
        
        print('Done.')
    else:
        print('Please specify the dataset source')
        exit(0)
'''
    # Create 2D pose file
    print('')
    print('Computing ground-truth 2D poses...')
    dataset = MpiInf3dhpDataset(output_filename + '.npz')
    output_2d_poses = {}
    for subject in dataset.subjects():
        output_2d_poses[subject] = {}
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            
            positions_2d = []
            for i,cam in enumerate(anim['cameras']):
                pos_3d = anim['positions'][i]
                pos_2d = wrap(project_to_2d, pos_3d, cam['intrinsic'], unsqueeze=True)
                pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
                positions_2d.append(pos_2d_pixel_space.astype('float32'))
            output_2d_poses[subject][action] = positions_2d
            
    print('Saving...')
    metadata = {
        'num_joints': dataset.skeleton().num_joints(),
        'keypoints_symmetry': [dataset.skeleton().joints_left(), dataset.skeleton().joints_right()]
    }
    np.savez_compressed(output_filename_2d2, positions_2d=output_2d_poses, metadata=metadata)
'''   

