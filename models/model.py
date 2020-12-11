import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributions as D
from models.utils import create_pointnet2_sa_components, create_pointnet2_fp_modules, create_mlp_components
from data_utils import Util

__all__ = ['Encoder', 'Decoder', 'Dynamic_Model', 'Filter', 'First_Latent', 'Prior', 'Model']

class Encoder(nn.Module):
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 2, 16), (256, 0.2, 32, (64, 128))),
        ((128, 2, 8), (64, 0.4, 32, (128, 128))),
        (None, (16, 0.8, 32, (128, 256))),
        (None, (1, 0.16, 16, (256, 256))),
    ]
    def __init__(self, num_classes, extra_feature_channels, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = extra_feature_channels + 3

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)
        self.channels_sa_features = channels_sa_features
        self.sa_in_channels = sa_in_channels

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['features']

        coords, features = inputs[:, :3, :].contiguous(), inputs
        coords_list, in_features_list = [], []
        for sa_blocks in self.sa_layers:
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords = sa_blocks((features, coords))
        in_features_list[0] = inputs[:, 3:, :].contiguous()

        return {"coords_list":coords_list, "coords":coords, "features":features, "in_features_list":in_features_list}



class MultivariateNormalDiag(nn.Module):
  def __init__(self,in_channels,  channels, latent_size):
    super(MultivariateNormalDiag, self).__init__()
    self.latent_size = latent_size
    self.dense1 = nn.Linear(in_channels, channels)
    self.dense2 = nn.Linear(channels, channels)
    self.output_layer = nn.Linear(channels, 2 * latent_size)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      inputs = torch.cat(inputs, axis=-1)
    else:
      inputs, = inputs
    out = F.leaky_relu(self.dense1(inputs))
    out = F.leaky_relu(self.dense2(out))
    out = self.output_layer(out)
    loc = out[..., :self.latent_size]
    cov_mat = torch.diag_embed(F.softplus(out[..., self.latent_size:]))

    return D.multivariate_normal.MultivariateNormal(loc, cov_mat)


class ConstantMultivariateNormalDiag(nn.Module):
  def __init__(self, latent_size):
    super(ConstantMultivariateNormalDiag, self).__init__()
    self.latent_size = latent_size

  def __call__(self, batch_size):
    loc = torch.zeros((batch_size, self.latent_size)).to(Util.device)
    cov_mat = torch.diag_embed(torch.ones((batch_size, self.latent_size))).to(Util.device)
    return D.multivariate_normal.MultivariateNormal(loc, cov_mat)


Dynamic_Model = MultivariateNormalDiag
Filter  = MultivariateNormalDiag
First_Latent = MultivariateNormalDiag
Prior = ConstantMultivariateNormalDiag





class Decoder(nn.Module):
    fp_blocks = [
        ((256,), (256, 1, None)),
        ((256,), (256, 1, None)),
        ((256, 128), (128, 1, None)),
        ((128, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 1, 32)),
    ]
    def __init__(self, num_classes, extra_feature_channels, channels_sa_features, \
        sa_in_channels, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = extra_feature_channels + 3

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[64, 0.5, num_classes],
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[64, 0.5, 32, 3],
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.reconstructor = nn.Sequential(*layers)


    def forward(self, coords_list, coords, features, in_features_list):
        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords = fp_blocks((coords_list[-1-fp_idx], coords, features, in_features_list[-1-fp_idx]))
        return {"segmentation":self.classifier(features), "reconstruction":self.reconstructor(features)}



class Model(nn.Module):

  def __init__(self):
    super().__init__()
    self.encoder = Encoder(Util.num_classes, Util.extra_features_channels)
    self.decoders = Decoder(Util.num_classes, Util.extra_features_channels, \
        self.encoder.channels_sa_features, self.encoder.sa_in_channels)

    self.filter = Filter(515, Util.pre_latent_channels, Util.latent_size)
    self.dynamic_model = Dynamic_Model(259, Util.pre_latent_channels, Util.latent_size)
    self.first_latent = First_Latent(256, Util.pre_latent_channels, Util.latent_size)
    self.prior = Prior(Util.latent_size)

  def  forward(self, inputs):
    # inputs dict : 
    # inputs["points"] : [B, T, channels * num_points]
    # inputs["camera_mov"] : [B, T, 3]

    batch_size, timesteps = inputs["points"].shape[:2]

    points = inputs["points"].reshape(batch_size*timesteps, 6, Util.num_points) # [B*T, channels, num_points]
    camera_poses = inputs["camera_poses"].reshape(batch_size*timesteps,3) #  [B*T, 3]

    encoder_output = self.encoder(points) 
    encoding = encoder_output["features"].squeeze(-1)\
          .reshape(batch_size, timesteps, -1).permute(1,0,2) # [T, B, channels]
    camera_poses = camera_poses.reshape(batch_size, timesteps, 3).permute(1,0,2) # [T, B, 3]

    first_latent_rep_dist = self.first_latent(encoding[0])
    first_latent_rep_sample = first_latent_rep_dist.rsample() # [B, latent_size]

    filter_latent_dists = []
    filter_latent_samples = []
    dynamic_latent_dists = []
    dynamic_latent_samples = []

    t_th_latent_rep =  first_latent_rep_sample
    t_th_dynamic_latent_pred = first_latent_rep_sample
    for t in range(1, timesteps):
      action = camera_poses[t] - camera_poses[t-1]

      filter_input = torch.cat([t_th_latent_rep, encoding[t], action], axis=-1) # [B, latent_size + 3 + channels]
      next_latent_rep_dist = self.filter(filter_input) 
      next_latent_rep_sample = next_latent_rep_dist.rsample() # [B, latent_size]

      next_dynamic_latent_dist = self.dynamic_model(torch.cat([t_th_dynamic_latent_pred, action], axis=-1))
      next_dynamic_latent_sample = next_dynamic_latent_dist.rsample() # [B, latent_size]

      t_th_latent_rep = next_latent_rep_sample
      t_th_dynamic_latent_pred = next_dynamic_latent_sample

      filter_latent_dists.append(next_latent_rep_dist)
      filter_latent_samples.append(next_latent_rep_sample.reshape(batch_size,1,-1))
      dynamic_latent_dists.append(next_dynamic_latent_dist)
      dynamic_latent_samples.append(next_dynamic_latent_sample.reshape(batch_size,1,-1))

    filter_latents = torch.stack([first_latent_rep_sample.reshape(batch_size,1,-1), *filter_latent_samples], 1) # [B, T, latent_size]
    dynamic_latents = torch.stack(dynamic_latent_samples, 1) # [B, T, latent_size]

    encoder_output["features"] = filter_latents.reshape(batch_size*timesteps, -1 , 1)
    res = self.decoders(**encoder_output)

    res["filter_latent_dists"] = filter_latent_dists
    res["dynamic_latent_dists"] = dynamic_latent_dists
    res["first_latent_rep_dist"] = first_latent_rep_dist

    return res

  def compute_losses(self, inputs, outputs):
    timesteps = len(outputs["filter_latent_dists"])+1
    batch_size = outputs["filter_latent_dists"][0].batch_shape[0]

    kl_1 = D.kl.kl_divergence(outputs["first_latent_rep_dist"], self.prior(batch_size))
    kl_t = 0
    for t in range(timesteps-1):
      kl_t += D.kl.kl_divergence(outputs["filter_latent_dists"][t], outputs["dynamic_latent_dists"][t])

    inp = inputs["points"].reshape(-1, 6, 8192)[:,:3,:]
    inp_seg = inputs["segmentation"].reshape(-1, 8192)
    
    reconstruction_loss = F.smooth_l1_loss(inp, outputs["reconstruction"], reduction='mean')*3*timesteps 

    weight_ = Util.compute_weight(inputs["segmentation"])
    segmentation_loss = F.cross_entropy(outputs["segmentation"], inp_seg, weight=torch.tensor(weight_).to(Util.device)) * timesteps

    return  dict({"reconstruction_loss":reconstruction_loss,\
                  "segmentation_loss":segmentation_loss,\
                  "kl_1":kl_1.mean(),\
                  "kl_t":kl_t.mean()})





