
import numpy as np
import scipy
import cv2
import pyvista as pv
import pyvistaqt as pvqt

from typing import List
from numpy.typing import ArrayLike
import time

def addCamera(plotter:pv.Plotter, cam_name:str, cam_t:ArrayLike, cam_R:ArrayLike, scale=0.04):
    points = np.array([[ 1.0,  0.8, 3.0],
                       [-1.0,  0.8, 3.0],
                       [-1.0, -0.8, 3.0],
                       [ 1.0, -0.8, 3.0],
                       [0.0, 0.0, 0.0]]) * 0.5 * scale
    points = (points - np.asarray(cam_t).reshape(1, 3)) @ np.asarray(cam_R).reshape(3,3)
    cam_mesh = pv.Pyramid(points)
    
    plotter.add_mesh(cam_mesh)
    plotter.add_point_labels(points[-1:], [cam_name], shape_opacity=0.2)

def addPoints(plotter:pv.Plotter, points:ArrayLike):
    # points: the first frame of the points, 
    #         the shape should be either [N, 3] or [N, m, 3], 
    #         where m is the number of points of a face
    points = np.asarray(points)
    assert points.ndim == 2 or points.ndim == 3
    assert points.shape[-1] == 3
    
    points = points.copy()
    points[abs(points) > 10] = 0
    
    if points.ndim == 2:
        mesh = plotter.add_points(points, render_points_as_spheres=True, point_size=0.01)
        return mesh
    
    faces = [([points.shape[1]] + [i*points.shape[1] + j for j in range(points.shape[1])]) for i in range(points.shape[0])]
    mesh = pv.PolyData(points.reshape(-1, 3), faces=faces)
    plotter.add_mesh(mesh)
    return mesh
    

def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cameras', type=str)
    parser.add_argument('-p', '--points', type=str)
    
    args = parser.parse_args()
    
    if ((args.cameras is None or len(args.cameras) == 0) and 
        (args.points is None or len(args.points) == 0)):
        parser.print_help()
        return
    
    points = None
    if args.points is not None:
        plotter = pvqt.BackgroundPlotter(show=True, auto_update=60)
        
        points = np.load(args.points)        
        
        points[abs(points) > 10] = 0
        mesh = addPoints(plotter, points[0])
    else:
        plotter = pv.Plotter()
        
        
    # pv.set_plot_theme("paraview")
    # pv.set_plot_theme("dark")
        
    if args.cameras is not None:
        with open(args.cameras, 'r') as f:
            cameras = json.load(f)
            
        cam_keys = []
        if isinstance(cameras, list):
            cam_keys = list(range(len(cameras)))
        elif isinstance(cameras, dict):
            cam_keys = cameras.keys()
            
        for ck in cam_keys:
            cam = cameras[ck]
            name = f'cam_{ck}' if isinstance(ck, int) else ck
            addCamera(plotter, name, cam['tvec'], cam['rmat'])
            
            
    plotter.add_mesh(pv.Arrow(direction=(1,0,0),scale=0.1), color='r')
    plotter.add_mesh(pv.Arrow(direction=(0,1,0),scale=0.1), color='g')
    plotter.add_mesh(pv.Arrow(direction=(0,0,1),scale=0.1), color='b')
    plotter.show_axes()
    # plotter.view_xz()
    plotter.view_isometric()
    plotter.show_bounds(grid='front', location='outer', all_edges=False, bold=False)
    
    if points is None:
        plotter.show()
        return
    
    plotter.show()
    def update_points():
        cnt = 1
        while plotter.app_window.isVisible():
            mesh.points = points[cnt].reshape(-1, 3)
            cnt = (cnt + 1) % points.shape[0]
            time.sleep(1/60)
                    
    from threading import Thread
    thread = Thread(target=update_points)
    thread.start()
        
    plotter.app.exec()

if __name__ == '__main__':
    main()