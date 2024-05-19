import argparse

def str2bool(v):
  return v.lower() in ('true', '1')

base_dir='.'
score_base_dir='./res_scorer'
parser = argparse.ArgumentParser()
arg_lists = []
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg
Dirs = add_argument_group('Dirs')
Dataset_args = add_argument_group('Dataset')
Test_Args=add_argument_group("Test_Args")
############################################# Base ###################################################
#Dirs
Dirs.add_argument('--base_dir',type=str,default=base_dir,help="base dir containing the whole project")
Dirs.add_argument('--origin_data_dir',type=str,default=f'{base_dir}/data',help="base dir containing the whole project")
Dirs.add_argument('--kps_dir',type=str,default=f'{score_base_dir}/data/KPts',help="base dir containing the whole project")
Dirs.add_argument('--fcgf_reg_dir',type=str,default=f'{score_base_dir}/data/FCGF_Reg',help='FCGF feature of trainset & matched transformations and labels for training dir')
Dirs.add_argument('--save_dir',type=str,default=f'{score_base_dir}/pre')
Dirs.add_argument('--model_fn',type=str,default=f'./model',help='well trained model path')

############################################# Test ###################################################
# detection
Test_Args.add_argument('--testset',default='3dmatch',type=str,help='dataset name')
Test_Args.add_argument('--keynum',default=1024,type=int,help='number of key points')

# extaction
Test_Args.add_argument('--backbone', default='./res_scorer/backbone/checkpoints/best_val_checkpoint.pth', type=str, help='path to latest backbone checkpoint (default: None)')
Test_Args.add_argument('--voxel_size',type=float,default=0.025, help='voxel size for FCGF feature')
Test_Args.add_argument('--dsp_voxel_size',type=float,default=0.03, help='downsample voxel size for pc')
Test_Args.add_argument('--curvature_radius',type=float,default=0.1, help='radius for surface curvature calculation')

#eistimation
Test_Args.add_argument('--best_model_fn',type=str,default='./res_scorer/model/model_best.pth')
Test_Args.add_argument('--max_iter',default=500,type=int,help='calculate transformation iterations')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config,unparsed

def print_usage():
    parser.print_usage()
