# 3D-Seq-Scene-Und

In this project, I am investigating semantic segmentation of 3D indoor scenes in video streams. Inspired by the work by ... , in ..., I am using sequential latent representation to capture the time dependency in the videos.

## Dataset
I used SceneNet dataset (link) which is made of RGB-D frames of different indoor scenes.
The entire dataset has a total of 15,000 trajectories videos with about 5 million RGB-D frames. Each frame contains :
* A photo of a given scene
* A depth map
* A semantic segmentation
* An Instance segmentation
* An Optical flow
