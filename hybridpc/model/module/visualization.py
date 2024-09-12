from pyexpat import features
import time
import os
import numpy as np
import math
import torchmetrics
import torch
from einops import repeat
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Callable, Tuple
import open3d as o3d
import pytorch_lightning as pl

def visualize_tool(dense_pointcloud, gt_pointcloud, input_pointcloud, scene_name):
    cmap = {
            0: [1, 0, 0],      # Red
            1: [0, 1, 0],      # Green
            2: [0, 0, 1],      # Blue
            3: [1, 1, 0],      # Yellow
            4: [0, 1, 1],      # Cyan
            5: [1, 0, 1],      # Magenta
            6: [0.5, 0.5, 0],  # Olive
            7: [0.5, 0, 0.5],   # Purple
            8: [0.5, 0.5, 0.5] # Gray
        }

    # Create an array of colors for each point cloud based on the color map
    original_colors = np.array([cmap[4] for _ in range(len(gt_pointcloud))])
    dense_colors = np.array([cmap[8] for _ in range(len(dense_pointcloud))])
    input_colors = np.array([cmap[5] for _ in range(len(input_pointcloud))])

    # sa_mesh = o3d.io.read_triangle_mesh("../data/Visualizations/scene0015_00.off")
    # if not sa_mesh.has_vertex_normals():
    #     sa_mesh.compute_vertex_normals()

    # sa_input = o3d.io.read_point_cloud("../data/Visualizations/scene0015_00.ply")

    # Create Open3D point cloud for gt_pointcloud
    original_points_cloud = o3d.geometry.PointCloud()
    original_points_cloud.points = o3d.utility.Vector3dVector(gt_pointcloud)
    original_points_cloud.colors = o3d.utility.Vector3dVector(original_colors)

    # Create Open3D point cloud for dense_pointcloud
    dense_points_cloud = o3d.geometry.PointCloud()
    dense_points_cloud.points = o3d.utility.Vector3dVector(dense_pointcloud)
    dense_points_cloud.colors = o3d.utility.Vector3dVector(dense_colors)

    # # convert dense_pointcloud to mesh
    # ratio = 1
    # alpha = 0.03

    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    #     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(dense_points_cloud, alpha)
    # print(mesh)
    # o3d.visualization.draw_geometries([mesh])
    
    # Create Open3D point cloud for dense_pointcloud
    input_points_cloud = o3d.geometry.PointCloud()
    input_points_cloud.points = o3d.utility.Vector3dVector(input_pointcloud)
    input_points_cloud.colors = o3d.utility.Vector3dVector(input_colors)

    # Save the point clouds to files
    o3d.io.write_point_cloud(f"../data/visualizations/{scene_name}_dense_pointcloud.ply", dense_points_cloud)
    o3d.io.write_point_cloud(f"./data/visualizations/{scene_name}_gt_pointcloud.ply", original_points_cloud)
    o3d.io.write_point_cloud(f"./data/visualizations/{scene_name}_input_pointcloud.ply", input_points_cloud)
    # o3d.io.write_triangle_mesh("./visualizations/reconstructed_mesh.ply", mesh)
    # trimesh.Trimesh(vertices=dense_pointcloud, faces=[]).export(
    #     "./visualizations/dense_point_cloud_{}.off")

class PointCloudVisualizer():
    def __init__(self, hparams, dense_generator):
        self.hparams = hparams
        self.dense_generator = dense_generator
        
    def udf_visualization(self, data_dict, encodes_dict, current_epoch, udf_loss, semantic_predictions):
        voxel_latents = encodes_dict['latent_codes']
        voxel_num = voxel_latents.shape[0]
        scene_name = data_dict['scene_names'][0]
        loss_type = 'L1'

        # save_dir = os.path.join(
        #     self.hparams.exp_output_root_path, 'inference', self.hparams.model.inference.split,
        #     'visualizations'
        # )
        save_dir = "/visualizations"

        if self.hparams.model.dense_generator.type == 'voxel':
            # for voxel_id in range(voxel_num):
            voxel_id = 2
            original_points = encodes_dict['relative_coords'][encodes_dict['indices'] == voxel_id].cpu().numpy()
            gt_labels = data_dict['labels'][encodes_dict['indices'] == voxel_id].detach().cpu().numpy()
            predicted_labels = semantic_predictions[encodes_dict['indices'] == voxel_id].detach().cpu().numpy()

            # Color mapping
            unique_labels = np.unique(gt_labels)
            mapping = {label: idx for idx, label in enumerate(unique_labels)}
            # cmap = plt.cm.get_cmap('jet', len(unique_labels))
            cmap = {
                0: [1, 0, 0],      # Red
                1: [0, 1, 0],      # Green
                2: [0, 0, 1],      # Blue
                3: [1, 1, 0],      # Yellow
                4: [0, 1, 1],      # Cyan
                5: [1, 0, 1],      # Magenta
                6: [0.5, 0.5, 0],  # Olive
                7: [0.5, 0, 0.5]   # Purple
            }

            # Number of unique classes
            num_gt_classes = len(np.unique(gt_labels))
            num_pred_classes = len(np.unique(predicted_labels))

            # Point-level accuracy
            correct_predictions = np.sum(gt_labels == predicted_labels)
            total_points = len(gt_labels)
            accuracy = correct_predictions / total_points
            print(f"Number of GT classes: {num_gt_classes}, Number of predicted classes: {num_pred_classes}, Point-level accuracy: {accuracy:.4f}")

            # 1. Original Point Cloud colored by GT labels
            original_colors = np.array([cmap[mapping[labels]] for labels in gt_labels])[:, :3]
            original_points_cloud = o3d.geometry.PointCloud()
            original_points_cloud.points = o3d.utility.Vector3dVector(original_points)
            original_points_cloud.colors = o3d.utility.Vector3dVector(original_colors)

            # 2. Prediction Point Cloud colored by predicted labels
            pred_colors = np.array([cmap[mapping[labels]] for labels in predicted_labels])[:, :3]
            prediction_pointcloud = o3d.geometry.PointCloud()
            prediction_pointcloud.points = o3d.utility.Vector3dVector(original_points)
            prediction_pointcloud.colors = o3d.utility.Vector3dVector(pred_colors)

            # 3. Error Point Cloud
            error_colors = np.array([[0.5, 0.5, 0.5] if gt == pred else [1, 0, 0] for gt, pred in zip(gt_labels, predicted_labels)])
            error_pointcloud = o3d.geometry.PointCloud()
            error_pointcloud.points = o3d.utility.Vector3dVector(original_points)
            error_pointcloud.colors = o3d.utility.Vector3dVector(error_colors)

            # 4. Dense Point Cloud colored uniformly
            dense_points, duration = self.dense_generator.generate_point_cloud(data_dict, encodes_dict, voxel_latents, voxel_id)
            uniform_color = [0, 0, 1]  # e.g. blue color
            dense_points_cloud = o3d.geometry.PointCloud()
            dense_points_cloud.points = o3d.utility.Vector3dVector(dense_points)
            dense_points_cloud.colors = o3d.utility.Vector3dVector(np.tile(uniform_color, (len(dense_points), 1)))

            if self.hparams.model.inference.show_visualizations:
                o3d.visualization.draw_geometries([original_points_cloud])
                o3d.visualization.draw_geometries([prediction_pointcloud])
                o3d.visualization.draw_geometries([error_pointcloud])
                o3d.visualization.draw_geometries([dense_points_cloud])

            if self.hparams.model.inference.save_predictions:
                # Save the visualizations with modified filenames
                folder_name = f'scene_{scene_name}_voxel-id-{voxel_id}'
                full_folder_path = os.path.join(save_dir, folder_name)
                if not os.path.exists(full_folder_path):
                    os.makedirs(full_folder_path)
                filename_base = f'{self.hparams.model.module}_Multiple_voxels_{self.hparams.model.voxel_size}'

                # o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_origin_class_num_{num_gt_classes}.ply'), original_points_cloud)
                # o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_prediction_class_num_{num_pred_classes}_error_accuracy_{accuracy:.4f}.ply'), prediction_pointcloud)
                # o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_error_accuracy_{accuracy:.4f}.ply'), error_pointcloud)
                # o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_udf-loss-{udf_loss}_dense.ply'), dense_points_cloud)

                # Save rotating videos with modified filenames
                self.save_rotating_video_from_object(original_points_cloud, os.path.join(save_dir, f'{filename_base}_origin_class_num_{num_gt_classes}.mp4'))
                self.save_rotating_video_from_object(prediction_pointcloud, os.path.join(save_dir, f'{filename_base}_prediction_class_num_{num_pred_classes}_error_accuracy_{accuracy:.4f}.mp4'))
                self.save_rotating_video_from_object(error_pointcloud, os.path.join(save_dir, f'{filename_base}_error_accuracy_{accuracy:.4f}.mp4'))
                self.save_rotating_video_from_object(dense_points_cloud, os.path.join(save_dir, f'{filename_base}_udf-loss-{udf_loss}_dense.mp4'))
        
        elif self.hparams.model.dense_generator.type == 'multiple_voxels':
            voxel_id = voxel_num - 10
            voxel_center = encodes_dict['voxel_coords'][:, 1:4] * self.hparams.data.voxel_size + self.hparams.data.voxel_size/2
            query_point = voxel_center[voxel_id]
            query_indices, _, _ = knn(query_point, voxel_center, self.hparams.model.dense_generator.k_neighbors)
            if self.hparams.model.inference.visualize_given_voxels:
                voxel_ids = self.hparams.model.inference.visualize_ids
            else:
                voxel_ids = query_indices

            all_original_points = []  
            all_dense_points = []
            # all_query_points = []
            all_gt_colors = []
            # all_query_points_colors = []
            all_dense_points_colors = []
            all_error_colors = []
            all_pred_colors = []
            all_gt_labels = []
            all_pred_labels = []

            # Color mapping
            cmap = {
                0: [1, 0, 0],
                1: [0, 1, 0],
                2: [0, 0, 1],
                3: [1, 1, 0],
                4: [0, 1, 1],
                5: [1, 0, 1],
                6: [0.5, 0.5, 0],
                7: [0.5, 0, 0.5]
            }

            for index, voxel_id in enumerate(voxel_ids):
                voxel_center = encodes_dict['voxel_coords'][:, 1:4][voxel_id] * self.hparams.data.voxel_size + self.hparams.data.voxel_size/2
                relative_points = (encodes_dict['relative_coords'][encodes_dict['indices'] == voxel_id]).cpu().numpy()
                absolute_points = relative_points + voxel_center.cpu().numpy()
                
                gt_labels = data_dict['labels'][encodes_dict['indices'] == voxel_id].detach().cpu().numpy()
                predicted_labels = semantic_predictions[encodes_dict['indices'] == voxel_id].detach().cpu().numpy()

                gt_colors = np.array([cmap[labels] for labels in gt_labels])[:, :3]
                pred_colors = np.array([cmap[labels] for labels in predicted_labels])[:, :3]
                error_colors = np.array([[0.5, 0.5, 0.5] if gt == pred else [1, 0, 0] for gt, pred in zip(gt_labels, predicted_labels)])
                
                all_original_points.append(absolute_points)
                all_gt_colors.append(gt_colors)
                all_pred_colors.append(pred_colors)
                all_error_colors.append(error_colors)
                all_gt_labels.append(gt_labels)
                all_pred_labels.append(predicted_labels)

                query_relative_points = (encodes_dict['query_relative_coords'][encodes_dict['query_indices'] == voxel_id]).cpu().numpy()
                query_absolute_points = query_relative_points + voxel_center.cpu().numpy()
                # all_query_points.append(query_absolute_points)

                current_voxel_id = voxel_id
                dense_points, duration = self.dense_generator.generate_point_cloud(data_dict, encodes_dict, voxel_latents, voxel_id)
                all_dense_points.append(dense_points + voxel_center.cpu().numpy())

                uniform_color = [0, 0, 1]
                all_dense_points_colors.append(np.repeat([uniform_color], dense_points.shape[0], axis=0))
                
            # Stacking colors and points for visualization
            all_points_np = np.vstack(all_original_points)
            all_gt_colors_np = np.vstack(all_gt_colors)
            all_pred_colors_np = np.vstack(all_pred_colors)
            all_dense_points_np = np.vstack(all_dense_points)
            all_dense_points_colors_np = np.vstack(all_dense_points_colors)
            all_error_colors_np = np.vstack(all_error_colors)
            all_gt_labels = np.vstack(all_gt_labels)
            all_pred_labels = np.vstack(all_pred_labels)

            # compute accuracy
            num_gt_classes = len(np.unique(all_gt_labels))
            num_pred_classes = len(np.unique(all_pred_labels))
            correct_predictions = np.sum(all_gt_labels == all_pred_labels)
            total_points = len(all_gt_labels)
            accuracy = correct_predictions / total_points
            print(f"Number of GT classes: {num_gt_classes}, Number of predicted classes: {num_pred_classes}, Point-level accuracy: {accuracy:.4f}")

            # Use open3d to construct point clouds
            original_points_cloud = o3d.geometry.PointCloud()
            original_points_cloud.points = o3d.utility.Vector3dVector(all_points_np)
            original_points_cloud.colors = o3d.utility.Vector3dVector(all_gt_colors_np)

            dense_points_cloud = o3d.geometry.PointCloud()
            dense_points_cloud.points = o3d.utility.Vector3dVector(all_dense_points_np)
            dense_points_cloud.colors = o3d.utility.Vector3dVector(all_dense_points_colors_np)

            error_pointcloud = o3d.geometry.PointCloud()
            error_pointcloud.points = o3d.utility.Vector3dVector(all_points_np)
            error_pointcloud.colors = o3d.utility.Vector3dVector(all_error_colors_np)

            pred_pointcloud = o3d.geometry.PointCloud()
            pred_pointcloud.points = o3d.utility.Vector3dVector(all_points_np)
            pred_pointcloud.colors = o3d.utility.Vector3dVector(all_pred_colors_np)

            if self.hparams.model.inference.show_visualizations:
                o3d.visualization.draw_geometries([original_points_cloud])
                o3d.visualization.draw_geometries([pred_pointcloud])
                o3d.visualization.draw_geometries([dense_points_cloud])
                o3d.visualization.draw_geometries([error_pointcloud])

            if self.hparams.model.inference.save_predictions:
                folder_name = f'scene_{scene_name}_voxel-id-{voxel_id}'
                full_folder_path = os.path.join(save_dir, folder_name)
                if not os.path.exists(full_folder_path):
                    os.makedirs(full_folder_path)
                filename_base = f'{self.hparams.model.module}_Multiple_voxels_{self.hparams.model.voxel_size}'
                
                # o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_dense.ply'), dense_points_cloud)
                # o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_origin.ply'), original_points_cloud)
                # o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_pred.ply'), pred_pointcloud)
                # o3d.io.write_point_cloud(os.path.join(save_dir, f'{filename_base}_error.ply'), error_pointcloud)
                
                self.save_rotating_video_from_object(original_points_cloud, os.path.join(save_dir, f'{filename_base}.mp4'))
                self.save_rotating_video_from_object(pred_pointcloud, os.path.join(save_dir, f'{filename_base}_error_accuracy_{accuracy:.4f}.mp4'))
                self.save_rotating_video_from_object(error_pointcloud, os.path.join(save_dir, f'{filename_base}_error_accuracy_{accuracy:.4f}.mp4'))
                self.save_rotating_video_from_object(dense_points_cloud, os.path.join(save_dir, f'{filename_base}_udf-loss-{udf_loss}_dense.mp4'))

                print(f"\nPredictions saved at {os.path.abspath(save_dir)}")

        else:
            original_points = data_dict["xyz"].cpu().numpy()
            original_points_cloud = o3d.geometry.PointCloud()
            original_points_cloud.points = o3d.utility.Vector3dVector(original_points)

            # dense_points, duration = self.dense_generator.generate_all_voxel_point_clouds(data_dict, functa_modulations['voxel_features'].F)
            dense_points, duration = self.dense_generator.generate_point_cloud(data_dict, functa_modulations['voxel_features'].F, 0)

            dense_points_cloud = o3d.geometry.PointCloud()
            dense_points_cloud.points = o3d.utility.Vector3dVector(dense_points)

            if self.hparams.model.inference.show_visualizations:
                o3d.visualization.draw_geometries([dense_points_cloud])
                o3d.visualization.draw_geometries([original_points_cloud])

            if self.hparams.model.inference.save_predictions:
                o3d.io.write_point_cloud(os.path.join(save_dir, f'{scene_name}_voxel_{self.hparams.model.network.encoder.voxel_size_out}_{loss_type}_udf_loss_{udf_loss:.5f}_dense.ply'), dense_points_cloud)
                o3d.io.write_point_cloud(os.path.join(save_dir, f'{scene_name}_voxel_{self.hparams.model.network.encoder.voxel_size_out}_{loss_type}_udf_loss_{udf_loss:.5f}_origin.ply'), original_points_cloud)            

            print(f"\nPredictions saved at {os.path.abspath(save_dir)}")



    def save_rotating_video_from_object(self, obj: o3d.geometry.Geometry3D, save_dir: str):
        """
        Generates a rotating video from an Open3D object.

        Parameters:
        - obj: The Open3D object to visualize.
        - save_dir: The directory (including filename) to save the video.
        """
        # Create a visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)  # invisible window for offscreen rendering
        vis.add_geometry(obj)
        
        # Rotate the view every frame and capture frames
        frames = []
        for i in range(360):  # 360 frames for a complete rotation
            ctr = vis.get_view_control()
            ctr.rotate(5.0, 0.0)  # rotate 5 degrees around z-axis
            vis.poll_events()
            vis.update_renderer()
            frame = vis.capture_screen_float_buffer(False)
            frames.append((np.asarray(frame) * 255).astype(np.uint8))
        
        # Save the frames as a video using imageio
        imageio.mimsave(save_dir, frames, fps=30)
        
        vis.destroy_window()