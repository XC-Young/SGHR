import numpy as np
import argparse
from tqdm import tqdm
import time
import os
import torch
from utils.utils import transform_points,make_non_exists_dir
from dataops.dataset import get_dataset_name
from utils.r_eval import compute_R_diff
import res_scorer.parses as parses
from res_scorer.detector import detector
from res_scorer.extractor import extractor
from res_scorer.network import Scorer

start = time.perf_counter()
parser = argparse.ArgumentParser()
config,nouse = parses.get_config()

datasets = get_dataset_name(config.testset,config.origin_data_dir)
datasetname=datasets['wholesetname']

detctor = detector(config)
extor = extractor(config)

if not os.path.exists(f'{config.kps_dir}/{config.testset}'):
    print('---> generate keypoints')
    detctor.curvature_kps(datasets)

if not os.path.exists(f'{config.fcgf_reg_dir}/{config.testset}'):
    print('---> extract FCGF feature')
    extor.batch_feature_extraction()

# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = Scorer(config).to(device)
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

pre_dir = f'./pre/cycle_results/cycle_prelogs'
for scene,dataset in tqdm(datasets.items()):
    if scene=='wholesetname':continue
    if scene=='valscenes':continue
    if dataset.name[0:4]=='3dLo':
        datasetname=f'3d{dataset.name[4:]}'
    else:
        datasetname=dataset.name
    Keys_dir=f'{config.kps_dir}/{datasetname}/Keypoints_PC'
    feats_dir=f'{config.fcgf_reg_dir}/{datasetname}/FCGF_feature'
    
    # load pre transformation
    pre_fn = f'{pre_dir}/{datasetname}/pre.log'
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
        #Keypoints
        Keys0=np.load(f'{Keys_dir}/cloud_bin_{id0}Keypoints.npy').astype(np.float32)
        Keys1=np.load(f'{Keys_dir}/cloud_bin_{id1}Keypoints.npy').astype(np.float32)
        trans = pair_trans[f'{id0}-{id1}']
        Keys1 = transform_points(Keys1,trans)
        #features
        feats0 = np.load(f'{feats_dir}/{id0}.npy').astype(np.float32)
        feats1 = np.load(f'{feats_dir}/{id1}.npy').astype(np.float32)
        inkeys0 = torch.from_numpy(Keys0.reshape(1,-1,Keys0.shape[1]).astype(np.float32)).to(device)
        inkeys1 = torch.from_numpy(Keys1.reshape(1,-1,Keys1.shape[1]).astype(np.float32)).to(device)
        infeats0 = torch.from_numpy(feats0.reshape(1,-1,feats0.shape[1]).astype(np.float32)).to(device) 
        infeats1 = torch.from_numpy(feats1.reshape(1,-1,feats1.shape[1]).astype(np.float32)).to(device)
        score = network(inkeys0, inkeys1, infeats0, infeats1)
        prediction = torch.round(torch.sigmoid(score)).int().cpu().numpy()

        gt = dataset.get_transform(id0,id1)
        tdiff = np.linalg.norm(trans[0:3,-1]-gt[0:3,-1])
        Rdiff=compute_R_diff(gt[0:3,0:3],trans[0:3,0:3])
        right = 0
        if Rdiff<3 and tdiff<0.2 :
            right = 1
        if right == int(prediction):
            correct_num += 1
        writer.write(f'{id0}-{id1}: pre:{int(prediction)} true:{right}  RRE:{Rdiff}   RTE:{tdiff}\n')
    recall = correct_num/len(dataset.pair_ids)
    writer.write(f'recall:{recall}')
    writer.close()