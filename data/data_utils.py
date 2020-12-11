import math
import numpy as np
import os
from PIL import Image
import scenenet_pb2 as sn
from collections import Counter



class Util:

  data_root_path = "/content/sample_data/train/"
  protobuf_path = "/content/drive/My Drive/SceneNet/train_protobufs/scenenet_rgbd_train_5.pb"

  tfrecords_data = "/content/drive/My Drive/SceneNet/data"


  num_points = 8192
  seq_len = 4
  batch_size = 3

  num_classes = 14
  extra_features_channels=3
  pre_latent_channels = 256
  latent_size=256

  learning_rate = 1e-3
  epochs = 50
  device='cuda'

  colour_code = np.array([[0, 0, 0],
                       [0, 0, 1],
                       [0.9137,0.3490,0.1882], #BOOKS
                       [0, 0.8549, 0], #CEILING
                       [0.5843,0,0.9412], #CHAIR
                       [0.8706,0.9451,0.0941], #FLOOR
                       [1.0000,0.8078,0.8078], #FURNITURE
                       [0,0.8784,0.8980], #OBJECTS
                       [0.4157,0.5333,0.8000], #PAINTING
                       [0.4588,0.1137,0.1608], #SOFA
                       [0.9412,0.1373,0.9216], #TABLE
                       [0,0.6549,0.6118], #TV
                       [0.9765,0.5451,0], #WALL
                       [0.8824,0.8980,0.7608]])

  NYU_WNID_TO_CLASS = {
    '04593077':4, '03262932':4, '02933112':6, '03207941':7, '03063968':10, '04398044':7, '04515003':7,
    '00017222':7, '02964075':10, '03246933':10, '03904060':10, '03018349':6, '03786621':4, '04225987':7,
    '04284002':7, '03211117':11, '02920259':1, '03782190':11, '03761084':7, '03710193':7, '03367059':7,
    '02747177':7, '03063599':7, '04599124':7, '20000036':10, '03085219':7, '04255586':7, '03165096':1,
    '03938244':1, '14845743':7, '03609235':7, '03238586':10, '03797390':7, '04152829':11, '04553920':7,
    '04608329':10, '20000016':4, '02883344':7, '04590933':4, '04466871':7, '03168217':4, '03490884':7,
    '04569063':7, '03071021':7, '03221720':12, '03309808':7, '04380533':7, '02839910':7, '03179701':10,
    '02823510':7, '03376595':4, '03891251':4, '03438257':7, '02686379':7, '03488438':7, '04118021':5,
    '03513137':7, '04315948':7, '03092883':10, '15101854':6, '03982430':10, '02920083':1, '02990373':3,
    '03346455':12, '03452594':7, '03612814':7, '06415419':7, '03025755':7, '02777927':12, '04546855':12,
    '20000040':10, '20000041':10, '04533802':7, '04459362':7, '04177755':9, '03206908':7, '20000021':4,
    '03624134':7, '04186051':7, '04152593':11, '03643737':7, '02676566':7, '02789487':6, '03237340':6,
    '04502670':7, '04208936':7, '20000024':4, '04401088':7, '04372370':12, '20000025':4, '03956922':7,
    '04379243':10, '04447028':7, '03147509':7, '03640988':7, '03916031':7, '03906997':7, '04190052':6,
    '02828884':4, '03962852':1, '03665366':7, '02881193':7, '03920867':4, '03773035':12, '03046257':12,
    '04516116':7, '00266645':7, '03665924':7, '03261776':7, '03991062':7, '03908831':7, '03759954':7,
    '04164868':7, '04004475':7, '03642806':7, '04589593':13, '04522168':7, '04446276':7, '08647616':4,
    '02808440':7, '08266235':10, '03467517':7, '04256520':9, '04337974':7, '03990474':7, '03116530':6,
    '03649674':4, '04349401':7, '01091234':7, '15075141':7, '20000028':9, '02960903':7, '04254009':7,
    '20000018':4, '20000020':4, '03676759':11, '20000022':4, '20000023':4, '02946921':7, '03957315':7,
    '20000026':4, '20000027':4, '04381587':10, '04101232':7, '03691459':7, '03273913':7, '02843684':7,
    '04183516':7, '04587648':13, '02815950':3, '03653583':6, '03525454':7, '03405725':6, '03636248':7,
    '03211616':11, '04177820':4, '04099969':4, '03928116':7, '04586225':7, '02738535':4, '20000039':10,
    '20000038':10, '04476259':7, '04009801':11, '03909406':12, '03002711':7, '03085602':11, '03233905':6,
    '20000037':10, '02801938':7, '03899768':7, '04343346':7, '03603722':7, '03593526':7, '02954340':7,
    '02694662':7, '04209613':7, '02951358':7, '03115762':9, '04038727':6, '03005285':7, '04559451':7,
    '03775636':7, '03620967':10, '02773838':7, '20000008':6, '04526964':7, '06508816':7, '20000009':6,
    '03379051':7, '04062428':7, '04074963':7, '04047401':7, '03881893':13, '03959485':7, '03391301':7,
    '03151077':12, '04590263':13, '20000006':1, '03148324':6, '20000004':1, '04453156':7, '02840245':2,
    '04591713':7, '03050864':7, '03727837':5, '06277280':11, '03365592':5, '03876519':8, '03179910':7,
    '06709442':7, '03482252':7, '04223580':7, '02880940':7, '04554684':7, '20000030':9, '03085013':7,
    '03169390':7, '04192858':7, '20000029':9, '04331277':4, '03452741':7, '03485997':7, '20000007':1,
    '02942699':7, '03231368':10, '03337140':7, '03001627':4, '20000011':6, '20000010':6, '20000013':6,
    '04603729':10, '20000015':4, '04548280':12, '06410904':2, '04398951':10, '03693474':9, '04330267':7,
    '03015149':9, '04460038':7, '03128519':7, '04306847':7, '03677231':7, '02871439':6, '04550184':6,
    '14974264':7, '04344873':9, '03636649':7, '20000012':6, '02876657':7, '03325088':7, '04253437':7,
    '02992529':7, '03222722':12, '04373704':4, '02851099':13, '04061681':10, '04529681':7,
}


  def normalize(v):
    return v/np.linalg.norm(v)

  def pixel_to_ray(pixel,vfov=45,hfov=60,pixel_width=320,pixel_height=240):
    x, y = pixel
    x_vect = math.tan(math.radians(hfov/2.0)) * ((2.0 * ((x+0.5)/pixel_width)) - 1.0)
    y_vect = math.tan(math.radians(vfov/2.0)) * ((2.0 * ((y+0.5)/pixel_height)) - 1.0)
    return (x_vect,y_vect,1.0)

  def normalised_pixel_to_ray_array(width=320,height=240):
    pixel_to_ray_array = np.zeros((height,width,3))
    for y in range(height):
        for x in range(width):
            pixel_to_ray_array[y,x] = Util.normalize(np.array(Util.pixel_to_ray((x,y),pixel_height=height,pixel_width=width)))
    return pixel_to_ray_array

  def path_from_view(render_path,view, type_, extension='png'):
    photo_path = os.path.join(render_path, type_)
    depth_path = os.path.join(photo_path,'{0}.{1}'.format(view.frame_num, extension))
    return os.path.join(Util.data_root_path,depth_path)

  
  def load_depth_map_in_m(file_name):
    image = Image.open(file_name)
    #if Util.resize != image.size:
    #  image = image.resize(Util.resize, Image.BILINEAR)
    pixel = np.array(image)
    return (pixel * 0.001)

  def load_instance(file_name):
    image = Image.open(file_name)
    #if Util.resize != image.size:
    #  image = image.resize(Util.resize, Image.BILINEAR)
    return np.array(image)

  def load_label(file_name):
    image = Image.open(file_name)
    #if Util.resize != image.size:
    #  image = image.resize(Util.resize, Image.BILINEAR)
    return np.array(image)

  def points_in_camera_coords(depth_map,pixel_to_ray_array):
    assert depth_map.shape[0] == pixel_to_ray_array.shape[0]
    assert depth_map.shape[1] == pixel_to_ray_array.shape[1]
    assert len(depth_map.shape) == 2
    assert pixel_to_ray_array.shape[2] == 3
    camera_relative_xyz = np.ones((depth_map.shape[0],depth_map.shape[1],4))
    for i in range(3):
        camera_relative_xyz[:,:,i] = depth_map * pixel_to_ray_array[:,:,i]
    return camera_relative_xyz

  def position_to_np_array(position):
    return np.array([position.x,position.y,position.z])

  def interpolate_poses(start_pose,end_pose,alpha):
    assert alpha >= 0.0
    assert alpha <= 1.0
    camera_pose = alpha * Util.position_to_np_array(end_pose.camera)
    camera_pose += (1.0 - alpha) * Util.position_to_np_array(start_pose.camera)
    lookat_pose = alpha * Util.position_to_np_array(end_pose.lookat)
    lookat_pose += (1.0 - alpha) * Util.position_to_np_array(start_pose.lookat)
    timestamp = alpha * end_pose.timestamp + (1.0 - alpha) * start_pose.timestamp
    pose = sn.Pose()
    pose.camera.x = camera_pose[0]
    pose.camera.y = camera_pose[1]
    pose.camera.z = camera_pose[2]
    pose.lookat.x = lookat_pose[0]
    pose.lookat.y = lookat_pose[1]
    pose.lookat.z = lookat_pose[2]
    pose.timestamp = timestamp
    return pose

  def world_to_camera_with_pose(view_pose):
    lookat_pose = Util.position_to_np_array(view_pose.lookat)
    camera_pose = Util.position_to_np_array(view_pose.camera)
    up = np.array([0,1,0])
    R = np.diag(np.ones(4))
    R[2,:3] = Util.normalize(lookat_pose - camera_pose)
    R[0,:3] = Util.normalize(np.cross(R[2,:3],up))
    R[1,:3] = -Util.normalize(np.cross(R[0,:3],R[2,:3]))
    T = np.diag(np.ones(4))
    T[:3,3] = -camera_pose
    return R.dot(T)

  def camera_to_world_with_pose(view_pose):
    return np.linalg.inv(Util.world_to_camera_with_pose(view_pose))

  def flatten_points(points):
    return points.reshape(-1, 4)

  def reshape_points(height,width,points):
    other_dim = points.shape[1]
    return points.reshape(height,width,other_dim)

  def transform_points(transform,points):
    assert points.shape[2] == 4
    height = points.shape[0]
    width = points.shape[1]
    points = Util.flatten_points(points)
    return Util.reshape_points(height,width,(transform.dot(points.T)).T)


  def get_class_from_instance(instance_img, mapping):
    class_img = np.zeros(instance_img.shape)
    h,w  = instance_img.shape

    class_img_rgb = np.zeros((h,w,3),dtype=np.uint8)
    r = class_img_rgb[:,:,0]
    g = class_img_rgb[:,:,1]
    b = class_img_rgb[:,:,2]

    for instance, semantic_class in mapping.items():
        class_img[instance_img == instance] = semantic_class
        r[instance_img==instance] = np.uint8(Util.colour_code[semantic_class][0]*255)
        g[instance_img==instance] = np.uint8(Util.colour_code[semantic_class][1]*255)
        b[instance_img==instance] = np.uint8(Util.colour_code[semantic_class][2]*255)

    class_img_rgb[:,:,0] = r
    class_img_rgb[:,:,1] = g
    class_img_rgb[:,:,2] = b

    class_img = Image.fromarray(np.uint8(class_img))
    class_img_rgb = Image.fromarray(class_img_rgb)

    return class_img, class_img_rgb



  def compute_weight(seg):
    weight = Counter(seg.cpu().data.numpy().flatten())
    len_, n_points = len(weight), Util.batch_size*Util.seq_len*Util.num_points
    weight = dict({(i,(n_points/len_)/o) for i,o in weight.items()})
    weight_ = [0]*Util.num_classes
    for i,x in weight.items():
      weight_[i] = x
    return weight_


