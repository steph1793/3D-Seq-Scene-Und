from PIL import Image
import math
import matplotlib
import numpy as np
import os
import pathlib
import random
import scenenet_pb2 as sn
import sys
import tensorflow as tf
from glob import glob
from data_utils import Util
from torch.utils.data import Dataset, DataLoader, IterableDataset

class Dataset_(Dataset): #### put in place a cache system

  def __init__(self):
    super(Dataset_).__init__()
    self.trajectories = sn.Trajectories()
    with open(Util.protobuf_path,'rb') as f:
        self.trajectories.ParseFromString(f.read())

    self.cached_pixel_to_ray_array = Util.normalised_pixel_to_ray_array()
    self.instance_class_map = {}

  def __len__(self):
    return 1000 * 300 # 250 images sur 300 (250 sequences) pour chacune des 209 premieres trajectoires
    # on garde le dernier pour les evals
    

  def __getitem__(self, key):
    i, j = key//300, key%300
    
    traj = self.trajectories.trajectories[i]
    for instance in traj.instances:
      instance_type = sn.Instance.InstanceType.Name(instance.instance_type)
      if instance.instance_type != sn.Instance.BACKGROUND:
        self.instance_class_map[instance.instance_id] = Util.NYU_WNID_TO_CLASS[instance.semantic_wordnet_id]

    #points_colors = []
    #segmentations = []
    #camera_poses = []

    #for t in range(j, j+Util.seq_len):
    view = traj.views[j] 

    depth_path = Util.path_from_view(traj.render_path, view, 'depth')
    depth_map = Util.load_depth_map_in_m(str(depth_path))
    depth_map[depth_map == 0.0] = 1000.0 #### to comment

    image_path = Util.path_from_view(traj.render_path, view, 'photo', "jpg")
    image_map = Util.load_instance(str(image_path))

    label_path = Util.path_from_view(traj.render_path, view, 'instance')
    label_map = Util.load_label(str(label_path))

    class_img, class_img_rgb = Util.get_class_from_instance(label_map, self.instance_class_map)

    # This is a 320x240x3 array, with each 'pixel' containing the 3D point in camera coords
    points_in_camera = Util.points_in_camera_coords(depth_map,self.cached_pixel_to_ray_array)

    # Transform point from camera coordinates into world coordinates
    ground_truth_pose = Util.interpolate_poses(view.shutter_open,view.shutter_close,0.5)
    camera_to_world_matrix = Util.camera_to_world_with_pose(ground_truth_pose)
    points_in_world = Util.transform_points(camera_to_world_matrix,points_in_camera)

    points_in_world = points_in_world[:,:,:3].reshape(-1,3)
    image_map = np.array(image_map).reshape(-1, 3)
    class_img = np.array(class_img).reshape(-1)
    class_img_rgb = np.array(class_img_rgb).reshape(-1,3)

    mean = np.mean(points_in_world, 0, keepdims=True)
    std = np.std(points_in_world, 0, keepdims=True)
    points_in_world = (points_in_world-mean)/std

    idxs = np.random.choice(len(points_in_world), Util.num_points)

    points_in_world, colors, class_img, class_img_rgb =  points_in_world[idxs], \
          image_map[idxs], class_img[idxs], class_img_rgb[idxs]

    points_color = np.transpose(np.concatenate([points_in_world, colors/255.], axis=-1), (1,0))
    segmentation = class_img
    c_ = ground_truth_pose.camera
    camera_pose = np.array([c_.x, c_.y, c_.z])

    #points_colors.append(points_color)
    #segmentations.append(segmentation)
    #camera_poses.append(camera_pose)

    # id = pool-id_traj-id_frame-id
    return {"frame_id" : f"5_{i}_{j}",
            "points_idxs_in_original" : idxs,
            "points":np.array(points_color), \
            "camera_poses":np.array(camera_pose), \
            "segmentation":np.array(segmentation)}



# Create a dictionary describing the features.
descriptor = {
    'frame_id': tf.io.FixedLenFeature([1], tf.string),
    'points_idxs_in_original': tf.io.FixedLenFeature([8192], tf.int64),
    'points': tf.io.FixedLenFeature([8192*6], tf.float32),
    'camera_poses': tf.io.FixedLenFeature([3], tf.float32),
    'segmentation': tf.io.FixedLenFeature([8192], tf.int64)
}

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, descriptor)


class Dataset2(IterableDataset):

  def __init__(self): 
    super(Dataset2).__init__()
    ds = glob(Util.tfrecords_data+"/*.pt")
    self.dataset = tf.data.TFRecordDataset(ds).map(_parse_image_function).batch(Util.seq_len, drop_remainder=True).shuffle(10).batch(Util.batch_size)

  def __iter__(self):
    return self.dataset.as_numpy_iterator()
    #return iter(range(iter_start, iter_end))




