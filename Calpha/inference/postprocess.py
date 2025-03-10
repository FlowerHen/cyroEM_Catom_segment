import numpy as np
from sklearn.cluster import MeanShift

class CryoPostprocessor:
    """Postprocessing pipeline for Cryo-EM predictions"""
    def __init__(self, voxel_size, origin):
        self.voxel_size = np.array(voxel_size)
        self.origin = np.array(origin)
        
    def refine_coordinates(self, pred_map, bandwidth=1.7):
        """Refine coordinates using mean shift clustering"""
        # Threshold prediction map
        mask = pred_map > 0.6
        if not np.any(mask):
            return np.empty((0, 3))
            
        # Convert grid indices to world coordinates
        indices = np.argwhere(mask)
        world_coords = self._indices_to_world(indices)
        
        # Mean shift clustering
        clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        clusterer.fit(world_coords)
        return clusterer.cluster_centers_
    
    def _indices_to_world(self, indices):
        return self.origin + indices * self.voxel_size
