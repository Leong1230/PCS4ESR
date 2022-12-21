import open3d as o3d
import numpy as np

ev = o3d.visualization.ExternalVisualizer()
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.random.rand(100,3)))
ev.set(pcd)