import torch
import numpy as np
from tqdm import tqdm
import open3d as o3d
import MinkowskiEngine as ME

from dataops.dataset import get_dataset_name
from res_scorer.backbone.model import load_model
from utils.utils import make_non_exists_dir
from utils.knn_search import knn_module

class FCGFDataset():
    def __init__(self,datasets,config):
        self.points={}
        self.pointlist=[]
        self.voxel_size = config.voxel_size
        self.datasets=datasets
        for scene,dataset in self.datasets.items():
            if scene=='wholesetname':continue
            if scene=='valscenes':continue
            for pc_id in dataset.pc_ids:
                self.pointlist.append((scene,pc_id))
                pts = self.datasets[scene].get_pc_o3d(pc_id)
                pts = pts.voxel_down_sample(config.voxel_size*0.4)
                pts = np.array(pts.points)
                self.points[f'{scene}_{pc_id}']=pts

    def __getitem__(self, idx):
        scene,pc_id=self.pointlist[idx]
        xyz0 = self.points[f'{scene}_{pc_id}']
        # Voxelization
        _, sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
        # Make point clouds using voxelized points
        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(xyz0)
        # Select features and points using the returned voxelized indices
        pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
        # Get coords
        xyz0 = np.array(pcd0.points)
        feats=np.ones((xyz0.shape[0], 1))
        coords0 = np.floor(xyz0 / self.voxel_size)
        return (xyz0, coords0, feats ,self.pointlist[idx])
    
    def __len__(self):
        return len(self.pointlist)
    
class extractor():
    def __init__(self,config):
        self.cfg = config
        self.dataset_name = self.cfg.testset
        self.output_dir = self.cfg.fcgf_reg_dir
        self.datasets = get_dataset_name(self.dataset_name,self.cfg.origin_data_dir)
        self.knn=knn_module.KNN(1)

    # extract batch feature of keypoints
    def collate_fn(self,list_data):
        xyz0, coords0, feats0, scenepc = list(
            zip(*list_data))
        xyz_batch0 = []
        dsxyz_batch0=[]
        batch_id = 0
        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            elif isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            else:
                raise ValueError(f'Can not convert to torch tensor, {x}')
        
        for batch_id, _ in enumerate(coords0):
            xyz_batch0.append(to_tensor(xyz0[batch_id]))
            _, inds = ME.utils.sparse_quantize(coords0[batch_id], return_index=True)
            dsxyz_batch0.append(to_tensor(xyz0[batch_id][inds]))

        coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)

        # Concatenate all lists
        xyz_batch0 = torch.cat(xyz_batch0, 0).float()
        dsxyz_batch0=torch.cat(dsxyz_batch0, 0).float()
        cuts_node=0
        cuts=[0]
        for batch_id, _ in enumerate(coords0):
            cuts_node+=coords0[batch_id].shape[0]
            cuts.append(cuts_node)

        return {
            'pcd0': xyz_batch0,
            'dspcd0':dsxyz_batch0,
            'scenepc':scenepc,
            'cuts':cuts,
            'sinput0_C': coords_batch0,
            'sinput0_F': feats_batch0.float(),
        }

    def Feature_extracting(self, data_loader):
        # load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(self.cfg.backbone)
        config = checkpoint['config']
        num_feats = 1
        Model = load_model(config.model)
        model = Model(
            num_feats,
            config.model_n_out,
            bn_momentum=0.05,
            normalize_feature=config.normalize_feature,
            conv1_kernel_size=config.conv1_kernel_size,
            D=3)    
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
         
        features={}
        for scene,dataset in self.datasets.items():
            if scene=='wholesetname':continue
            if scene=='valscenes':continue
            for pc_id in dataset.pc_ids:
                features[f'{scene}_{pc_id}']=[]

        with torch.no_grad():
            for i, input_dict in enumerate(tqdm(data_loader)):
                sinput0 = ME.SparseTensor(
                        input_dict['sinput0_F'].to(device),
                        coordinates=input_dict['sinput0_C'].to(device))
                torch.cuda.synchronize()
                F0 = model(sinput0).F
                
                cuts=input_dict['cuts']
                scene_pc=input_dict['scenepc']
                for inb in range(len(scene_pc)):
                    scene,pc_id=scene_pc[inb]
                    make_non_exists_dir(f'{self.output_dir}/{self.dataset_name}/{scene}/FCGF_feature')
                    feature=F0[cuts[inb]:cuts[inb+1]]
                    pts=input_dict['dspcd0'][cuts[inb]:cuts[inb+1]]#*config.voxel_size

                    Keys_i=self.kps[f'{scene}_{pc_id}']
                    xyz_down=pts.T[None,:,:].cuda() #1,3,n
                    d,nnindex=self.knn(xyz_down,Keys_i)
                    nnindex=nnindex[0,0]
                    one_R_output=feature[nnindex,:].cpu().numpy()#keynum*32
                                        
                    np.save(f'{self.output_dir}/{self.dataset_name}/{scene}/FCGF_feature/{pc_id}.npy',one_R_output)

    def batch_feature_extraction(self):
        #preload kps
        self.kps={}
        for scene,dataset in self.datasets.items():
            if scene=='wholesetname':continue
            if scene=='valscenes':continue
            if dataset.name[0:4]=='3dLo':
                datasetname=f'3d{dataset.name[4:]}'
            else:
                datasetname=dataset.name
            for pc_id in dataset.pc_ids:
                kps = np.load(f'{self.cfg.kps_dir}/{datasetname}/Keypoints_PC/cloud_bin_{pc_id}Keypoints.npy')
                self.kps[f'{scene}_{pc_id}']=torch.from_numpy(kps.T[None,:,:].astype(np.float32)).cuda()
        dset=FCGFDataset(self.datasets,self.cfg)
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=4, # if out of memory change the batch_size to 1
            shuffle=False,
            num_workers=16,
            collate_fn=self.collate_fn,
            pin_memory=False,
            drop_last=False)
        self.Feature_extracting(loader)