# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import argparse
import subprocess
 
parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str, required=True)
args = parser.parse_args()

# run mtcnn needed for Deep3DFaceRecon
command_list = ['python', 'batch_mtcnn.py', '--in_root', args.indir]
print(" ".join(s for s in command_list))
# subprocess.run(command_list)
command = "python batch_mtcnn.py --in_root " + args.indir
subprocess.run(command, shell=True)
# print(command)
# os.system(command)

v_id = args.indir.split("/")[-2] if args.indir.endswith("/") else args.indir.split("/")[-1]
out_folder = f"{args.indir}/results"

# run Deep3DFaceRecon
os.chdir('Deep3DFaceRecon_pytorch')
# command_list = ['python', 'test.py', f'--img_folder={args.indir}', '--gpu_ids=0', '--name=pretrained', '--epoch=20', '--use_opengl', 'False']
# print(" ".join(s for s in command_list))
# subprocess.run(command_list)
command = "python test.py --img_folder=" + args.indir + " --gpu_ids=0 --name=pretrained --epoch=20 --use_opengl False"
print(command)
subprocess.run(command, shell=True)
# os.system(command)
# os.chdir('/home/jio/workspace/3DInversion/data_preprocessing/eg3d/dataset_preprocessing/ffhq')
os.chdir('..')

# crop out the input image
# command_list = ['python', 'crop_images_in_the_wild.py ', f'--indir={args.indir}']
# print(" ".join(s for s in command_list))
# subprocess.run(command_list)
command = "python crop_images_in_the_wild.py --indir=" + args.indir
print(command)
subprocess.run(command, shell=True)
# os.system(command)

# convert the pose to our format
# command_list = ['python', '3dface2idr_mat.py ', '--in_root', f'Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/{v_id}/epoch_20_000000', 
#                 '--out_path', os.path.join(args.indir, 'crop', 'cameras.json')]
# print(" ".join(s for s in command_list))
# subprocess.run(command_list)
command = f"python 3dface2idr_mat.py --in_root Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/{v_id}/epoch_20_000000 --out_path {os.path.join(args.indir, 'crop', 'cameras.json')}"
print(command)
subprocess.run(command, shell=True)
# os.system(command)

# additional correction to match the submission version
# command_list = ['python', 'preprocess_face_cameras.py ', '--source', os.path.join(args.indir, 'crop'), '--dest', v_id, '--mode', 'orig']
# print(" ".join(s for s in command_list))
# subprocess.run(command_list)
command = f"python preprocess_face_cameras.py --source {os.path.join(args.indir, 'crop')} --dest {out_folder} --mode orig"
print(command)
subprocess.run(command, shell=True)
# os.system(command)