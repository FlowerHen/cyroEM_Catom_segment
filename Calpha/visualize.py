import pyvista as pv
import numpy as np

def visualize_prediction(volume, true_points=None, pred_points=None):
    plotter = pv.Plotter()
    
    grid = pv.UniformGrid()
    grid.dimensions = volume.shape
    grid.point_data["values"] = volume.flatten(order="F")
    contours = grid.contour([0.5])
    plotter.add_mesh(contours, opacity=0.5)
    
    if true_points is not None:
        plotter.add_points(true_points, color='green', point_size=10)
    if pred_points is not None:
        plotter.add_points(pred_points, color='red', point_size=5)
        
    plotter.show_axes()
    plotter.show()