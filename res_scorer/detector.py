import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from utils.utils import make_non_exists_dir

class detector():
    def __init__(self,cfg):
        self.cfg = cfg

    def pca_compute(self, data, sort=True):
        average_data = np.mean(data, axis=0)  # calculate the mean
        decentration_matrix = data - average_data  
        H = np.dot(decentration_matrix.T, decentration_matrix)  # solve for the covariance matrix H
        eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(H)  # use SVD to solve eigenvalues and eigenvectors
        if sort:
            sort = eigenvalues.argsort()[::-1]  # descending sort
            eigenvalues = eigenvalues[sort]  # index
        return eigenvalues
    
    def calcuate_surface_curvature(self, cloud, radius=0.1):
        points = np.asarray(cloud.points)
        kdtree = o3d.geometry.KDTreeFlann(cloud)
        num_points = len(cloud.points)
        curvature = []  
        for i in range(num_points):
            k, idx, _ = kdtree.search_radius_vector_3d(cloud.points[i], radius)
            neighbors = points[idx, :]
            w = self.pca_compute(neighbors)  # w is the eigenvalue
            delt = np.divide(w[2], np.sum(w), out=np.zeros_like(w[2]), where=np.sum(w) != 0)
            curvature.append(delt)
        curvature = np.array(curvature, dtype=np.float64)
        return curvature
    
    def curvature_kps(self,datasets):  
        for scene,dataset in tqdm(datasets.items()):
            if scene=='wholesetname':continue
            if scene=='valscenes':continue
            if dataset.name[0:4]=='3dLo':
                datasetname=f'3d{dataset.name[4:]}'
            else:
                datasetname=dataset.name
            # Downsampling keypoints based on curvature values
            kps_idx_dir = f'{self.cfg.kps_dir}/{datasetname}/Keypoints'
            kps_dir = f'{self.cfg.kps_dir}/{datasetname}/Keypoints_PC'
            make_non_exists_dir(kps_idx_dir)
            make_non_exists_dir(kps_dir)
            for pc_id in tqdm(dataset.pc_ids):
                if os.path.exists(f'{kps_dir}/cloud_bin_{pc_id}Keypoints.npy'):continue
                pcd = dataset.get_pc_o3d(pc_id)
                pcd_down = pcd.voxel_down_sample(voxel_size = self.cfg.dsp_voxel_size)
                pc_xyz = np.asarray(pcd_down.points)
                surface_curvature = self.calcuate_surface_curvature(pcd_down, radius=self.cfg.curvature_radius)
                weight = surface_curvature/np.sum(surface_curvature)
                kps_idx = np.random.choice(pc_xyz.shape[0],self.cfg.keynum,p=weight)
                kps = pc_xyz[kps_idx]
                np.savetxt(f'{kps_idx_dir}/cloud_bin_{pc_id}Keypoints.txt',kps_idx)
                np.save(f'{kps_dir}/cloud_bin_{pc_id}Keypoints.npy',kps)
