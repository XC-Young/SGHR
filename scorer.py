import numpy as np
import argparse
from tqdm import tqdm
import os
import torch
from utils.utils import transform_points,make_non_exists_dir
from dataops.dataset import get_dataset_name
from utils.r_eval import compute_R_diff
from scorer.ptv3.dataops.scorer_transform import TEST_TRANSFORM
from scorer.ptv3.Classifier import Classifier

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes',type=int,default=1)
parser.add_argument('--backbone_embed_dim',type=int,default=512)
parser.add_argument('--testset',type=str,default='3dmatch',help='dataset name')
parser.add_argument('--best_model_fn',type=str,default='./scorer/model_best.pth')
parser.add_argument('--origin_data_dir',type=str,default='./data')
parser.add_argument('--pre_dir',type=str,default='./pre/cycle_results/cycle_prelogs')
parser.add_argument('--save_dir',type=str,default='./scorer/result')
parser.add_argument('--dsp_voxel_size',type=float,default=0.03)
config = parser.parse_args()

transformer = TEST_TRANSFORM()
# load model
network = Classifier(config).cuda()
def load_model(config,network):
    best_para = 0
    if os.path.exists(config.best_model_fn):
        checkpoint=torch.load(config.best_model_fn)
        best_para = checkpoint['best_para']
        network.load_state_dict(checkpoint['network_state_dict'])
        print(f'\n ==> resuming best para {best_para}')
    else:
        raise ValueError("No model exists")
load_model(config, network)
network.eval()

datasets = get_dataset_name(config.testset,config.origin_data_dir)
datasetname=datasets['wholesetname']
for scene,dataset in tqdm(datasets.items()):
    if scene=='wholesetname':continue
    if scene=='valscenes':continue
    if dataset.name[0:4]=='3dLo':
        datasetname=f'3d{dataset.name[4:]}'
    else:
        datasetname=dataset.name
    
    # load pre transformation
    pre_fn = f'{config.pre_dir}/{datasetname}/pre.log'
    pair_trans = {}
    with open(pre_fn,'r') as f:
        lines = f.readlines()
        pair_num = len(lines)//5
        for k in range(pair_num):
            id0,id1=np.fromstring(lines[k*5],dtype=np.float32,sep=' ')[0:2]
            id0=int(id0)
            id1=int(id1)
            row0=np.fromstring(lines[k*5+1],dtype=np.float32,sep=' ')
            row1=np.fromstring(lines[k*5+2],dtype=np.float32,sep=' ')
            row2=np.fromstring(lines[k*5+3],dtype=np.float32,sep=' ')
            row3=np.fromstring(lines[k*5+4],dtype=np.float32,sep=' ')
            transform=np.stack([row0,row1,row2,row3])
            pair_trans[f'{id0}-{id1}'] = transform
    
    score_dir = (f'{config.save_dir}/{datasetname}')
    make_non_exists_dir(score_dir)
    writer=open(f'{score_dir}/pre.log','w')
    correct_num = 0
    for pair in tqdm(dataset.pair_ids):
        id0,id1 = pair
        pcd0 = dataset.get_pc_o3d(id0)
        pcd1 = dataset.get_pc_o3d(id1)
        pcd0 = pcd0.voxel_down_sample(config.dsp_voxel_size)
        pcd1 = pcd1.voxel_down_sample(config.dsp_voxel_size)
        pc0 = np.asarray(pcd0.points)
        pc1 = np.asarray(pcd1.points)
        trans = pair_trans[f'{id0}-{id1}']
        pc1 = transform_points(pc1,trans)

        xyzp0 = np.c_[pc0,np.zeros(pc0.shape[0]).T]
        xyzp1 = np.c_[pc1,np.ones(pc1.shape[0]).T]
        xyzp = np.concatenate((xyzp0,xyzp1),axis=0)
        coord = xyzp[:,0:3].astype(np.float32)
        pc_idx = xyzp[:,3].astype(np.int32).reshape(-1,1)
        data_dict = transformer(coord,pc_idx)
        for key,val in data_dict.items():
            data_dict[key] = data_dict[key].cuda()
        cls_logits = network(data_dict)
        prediction = torch.sigmoid(cls_logits).detach().cpu().numpy()
        pre_int = int(np.round(prediction))

        gt = dataset.get_transform(id0,id1)
        tdiff = np.linalg.norm(trans[0:3,-1]-gt[0:3,-1])
        Rdiff=compute_R_diff(gt[0:3,0:3],trans[0:3,0:3])
        right = 0
        if Rdiff<15 and tdiff<0.3 :
            right = 1
        if right == pre_int:
            correct_num += 1
        writer.write(f'{id0}-{id1}: pre:{prediction}  pre:{pre_int}  true:{right}   RRE:{Rdiff}  RTE:{tdiff}\n')
    recall = correct_num/len(dataset.pair_ids)
    writer.write(f'recall:{recall}')
    writer.close()
