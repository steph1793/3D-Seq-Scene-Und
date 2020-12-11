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

The image above represents the graphical model of our Deep Learning model. But since we are processing videos, there is a time dependency which I modelled trhough the latent representations, thus the sequential latent representation Learning. Each frame of the scene is encoded into a latent representation, from which the semantic labelling is decoded. `x_t` = the incoming video frames, `z_t` the latent representations, `y_t` the semantic labelling, `a_t` the actions that lead to each frame (the camera motion for example).
From this graphical model and some prior insights, the objective function to be optimize (ELBO) can be derived as well as better insights about the design of the model : 
* An encoder to extract features from each frame
* A dynamic model able to predict future states from the previous ones and the actions taken
* A filter that fuses the informations from the incoming frames, the previous states (latent representations) and the actions (the camera movement)
* A  reconstructor 
* A decoder


For the encoder and the decoder, I used a benchmark method for 3D data semantic segmentation : PVCNN [link]. This method outperformed numerous benchmark methods (PointNet++, PointCNN etc) and has proven efficient in terms of memory footprint, and computation time complexity.

[PVCNN] image


### Requirements

* python>=3.6
* Ninja
* torch 1.7.0
* numpy

### Training
To train a model, we can change some parameters in data_utils file. I will be releasing only a part of the code for now, it may not be fully ready for use.
```
python main.py --mode train --save_folder ...
```

### Evaluation

Coming next


## Next

- [x] Dataset (gathering and processing)
- [x] Model building
- [ ] Training an experiments
- [ ] Ablation studies
- [ ] Comparative Studies
- [ ] Comet Haley

