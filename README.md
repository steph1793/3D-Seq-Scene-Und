# 3D-Seq-Scene-Und

## Indoor Scene Understanding through Sequential Latent representation

In this project, I am investigating semantic segmentation of 3D indoor scenes in video streams. Inspired by the work by ... , in ..., I am using sequential latent representation to capture the time dependency in the videos. The work done here by the way re-uses some code from [link] for 3D semantic segmentation and [link] for data processing.

## Dataset
I used SceneNet dataset (link) which is made of RGB-D frames of different indoor scenes. This a very realistic synthetic dataset with a realistic camera used to film synthetic rooms.
The entire dataset has a total of 15,000 trajectories videos with about 5 million RGB-D frames. Each frame contains :
* A photo of a given scene
* A depth map
* A semantic segmentation
* An Instance segmentation
* An Optical flow

To prepare the dataset, each frame is mapped into a real world 3D point cloud using the synthetic camera poses and intrisics. Then I subsampled a smaller point cloud for computations sake.

[dataset image]


## Model

[ image graphical model ]

The image above represents the graphical model of our Deep Learning model. $`x_t`$ Each frame of the scene is encoded into a latent representation, from which the semantic labelling is decoded. But since we are processing videos, there is a time dependency which I modelled trhough the latent representations, thus the sequential latent representation Learning. 
From this graphical and some prior insights, the objective function to be optimize (ELBO) can be derived.
