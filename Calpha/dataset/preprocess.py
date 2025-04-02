import numpy as np
import warnings
from Bio.PDB import MMCIFParser
from Bio import BiopythonWarning 
warnings.simplefilter('ignore', BiopythonWarning)
from scipy.spatial import KDTree
from .cube_rotation import sampling_cube_rotations

# Class to preprocess Cryo-EM data
class CryoEMPreprocessor:
    def __init__(self, config):
        self.config = config
        self.voxel_size = config['data']['voxel_size']
        
    def load_npz(self, path):
        try:
            data = np.load(path, mmap_mode='r')
            grid = data['grid'].astype(np.float32)
            voxel_size = data['voxel_size']
            global_origin = data['global_origin']
            return {
                'grid': grid,
                'voxel_size': voxel_size,
                'global_origin': global_origin
            }
        except KeyError as e:
            raise ValueError(f"Missing key in NPZ file: {e}")

    def parse_cif(self, cif_path):
        parser = MMCIFParser()
        structure = parser.get_structure('protein', cif_path)
        
        ca_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.is_disordered():
                        atoms = residue.child_list
                    else:
                        atoms = list(residue.get_atoms())
                        
                    for atom in atoms:
                        if atom.get_name().strip().upper() == 'CA':
                            ca_coords.append(atom.get_coord())
        
        if len(ca_coords) == 0:
            raise ValueError(f"No C-alpha atoms found in {cif_path}")
        
        return np.array(ca_coords)

    def create_labels(self, grid_data, ca_coords):
        hard_label = np.zeros_like(grid_data['grid'])
        soft_label = np.zeros_like(grid_data['grid'])
        
        grid_indices = ((ca_coords - grid_data['global_origin']) / 
                        grid_data['voxel_size']).astype(int)
        
        valid_indices = np.all((grid_indices >= 0) & 
                               (grid_indices < grid_data['grid'].shape), axis=1)
        for idx in grid_indices[valid_indices]:
            hard_label[tuple(idx)] = 1.0
            
        kdtree = KDTree(ca_coords)
        grid_points = np.stack(np.indices(grid_data['grid'].shape), axis=-1) * \
                      grid_data['voxel_size'] + grid_data['global_origin'] 
        dists, _ = kdtree.query(grid_points.reshape(-1, 3))
        soft_label = np.exp(-(dists**2)/(self.config['data']['d0']**2))
        soft_label = soft_label.reshape(grid_data['grid'].shape)
        
        return hard_label, soft_label  

# Class to apply augmentations to Cryo-EM volumes and labels
class VolumeAugmentor:
    def __init__(self, config):
        aug_config = config.get('augmentation', {})
        seed_config = config.get('seed', None)
        if seed_config is not None:
            np.random.seed(seed_config)

        rotation_config = aug_config.get('rotation', {})
        self.rotation_prob = rotation_config.get('prob', 0)
        self.rotation_samples = rotation_config.get('rotation_samples', 2)
        
        resolution_config = aug_config.get('resolution', {})
        self.resolution_prob = resolution_config.get('prob', 0)
        self.resolution_sigma_range = resolution_config.get('sigma_range', [0.6, 1.0])
        
        noise_config = aug_config.get('noise', {})
        self.noise_prob = noise_config.get('prob', 0)
        self.noise_ratio_range = noise_config.get('ratio_range', [0.1, 0.3])
        self.noise_std = noise_config.get('std', 0.5)
        
        intensity_inversion_config = aug_config.get('intensity_inversion', {})
        self.inversion_prob = intensity_inversion_config.get('prob', 0)
        
        gamma_config = aug_config.get('gamma_correction', {})
        self.gamma_correction_prob = gamma_config.get('prob', 0)
        self.gamma_range = gamma_config.get('gamma_range', [0.8, 1.2])
    
    def __call__(self, volume, labels):
        hard_label, soft_label = labels

        if np.random.rand() < self.rotation_prob:
            from .cube_rotation import sample_rotation_params, apply_rotation
            rotations = sample_rotation_params(self.rotation_samples)
            rotation_param = rotations[np.random.randint(0, len(rotations))]
            volume = apply_rotation(volume, rotation_param).copy()
            hard_label = apply_rotation(hard_label, rotation_param).copy()
            soft_label = apply_rotation(soft_label, rotation_param).copy()

        if np.random.rand() < self.resolution_prob:
            from scipy.ndimage import gaussian_filter
            sigma = np.random.uniform(self.resolution_sigma_range[0], 
                                    self.resolution_sigma_range[1])
            volume = gaussian_filter(volume, sigma=sigma)

        if np.random.rand() < self.noise_prob:
            noise_ratio = np.random.uniform(self.noise_ratio_range[0], 
                                          self.noise_ratio_range[1])
            noise_mask = np.random.rand(*volume.shape) < noise_ratio
            noise = np.random.normal(0, self.noise_std, volume.shape)
            volume[noise_mask] += noise[noise_mask]

        if np.random.rand() < self.inversion_prob:
            volume = -volume

        if np.random.rand() < self.gamma_correction_prob:
            gamma_val = np.random.uniform(self.gamma_range[0], 
                                        self.gamma_range[1])
            vol_min = volume.min()
            volume_shifted = volume - vol_min
            vol_max = volume_shifted.max()
            if vol_max > 0:
                volume_norm = volume_shifted / vol_max
                volume_norm = volume_norm ** gamma_val
                volume = volume_norm * vol_max + vol_min
        
        return volume, (hard_label, soft_label)
