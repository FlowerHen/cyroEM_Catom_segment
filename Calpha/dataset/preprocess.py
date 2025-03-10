import numpy as np
import warnings
from Bio.PDB import MMCIFParser
from Bio import BiopythonWarning 
import warnings
warnings.simplefilter('ignore', BiopythonWarning)
from scipy.spatial import KDTree
from .cube_rotation import sampling_cube_rotations

class CryoEMPreprocessor:
    def __init__(self, config):
        self.config = config
        self.voxel_size = config['data']['voxel_size']
        
    def load_npz(self, path):
        """Load NPZ file and validate contents"""
        data = np.load(path, mmap_mode='r')
        return {
            'grid': data['grid'].astype(np.float32),
            'voxel_size': data['voxel_size'],
            'global_origin': data['global_origin']
        }

    def parse_cif(self, cif_path):
        """Extract C-alpha coordinates from CIF file with validation"""
        parser = MMCIFParser()
        structure = parser.get_structure('protein', cif_path)
        
        ca_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Handle disordered residues by taking the first conformation
                    if residue.is_disordered():
                        atoms = residue.child_list
                    else:
                        atoms = list(residue.get_atoms())
                        
                    for atom in atoms:
                        # Only append if the atom name is 'CA'
                        if atom.get_name().strip() == 'CA':
                            ca_coords.append(atom.get_coord())
        
        if len(ca_coords) == 0:
            raise ValueError(f"No C-alpha atoms found in {cif_path}")
        
        return np.array(ca_coords)

    def create_labels(self, grid_data, ca_coords):
        """Generate hard and soft labels from coordinates"""
        hard_label = np.zeros_like(grid_data['grid'])
        soft_label = np.zeros_like(grid_data['grid'])
        
        # Convert coordinates to grid indices
        grid_indices = ((ca_coords - grid_data['global_origin']) / 
                       grid_data['voxel_size']).astype(int)
        
        # Create hard labels
        valid_indices = np.all((grid_indices >= 0) & 
                             (grid_indices < grid_data['grid'].shape), axis=1)
        for idx in grid_indices[valid_indices]:
            hard_label[tuple(idx)] = 1.0
            
        # Create soft labels using KDTree
        kdtree = KDTree(ca_coords)
        grid_points = np.stack(np.indices(grid_data['grid'].shape), axis=-1) * \
                    grid_data['voxel_size'] + grid_data['global_origin']
        dists, _ = kdtree.query(grid_points.reshape(-1, 3))
        soft_label = np.exp(-(dists**2)/(self.config['data']['d0']**2))
        soft_label = soft_label.reshape(grid_data['grid'].shape)
        
        return hard_label, soft_label

class VolumeAugmentor:
    def __init__(self, config):
        self.rotation_samples = config['augmentation'].get('rotation_samples', 0)
        self.noise_prob = config['augmentation'].get('noise_prob', 0.5)
        self.noise_std = config['augmentation'].get('noise_std', 0.05)
        self.noise_multiplier = config['augmentation'].get('noise_multiplier', 1.0)
        
    def __call__(self, volume, labels):
        hard_label, soft_label = labels
        
        # Apply rotation augmentation
        if self.rotation_samples > 0:
            from .cube_rotation import sample_rotation_params, apply_rotation
            rotations = sample_rotation_params(self.rotation_samples)
            # For online augmentation, select one random rotation from the list
            rotation_param = rotations[np.random.randint(0, len(rotations))]
            volume = apply_rotation(volume, rotation_param).copy()
            hard_label = apply_rotation(hard_label, rotation_param).copy()
            soft_label = apply_rotation(soft_label, rotation_param).copy()
        
        # Apply noise augmentation
        if np.random.rand() < self.noise_prob:
            noise = np.random.normal(0, self.noise_std * self.noise_multiplier, volume.shape)
            volume = np.clip(volume + noise, 0, 1).astype(np.float32)
            
        return volume, (hard_label, soft_label)
