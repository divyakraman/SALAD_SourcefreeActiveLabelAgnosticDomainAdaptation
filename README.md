### Paper - [**SALAD: Source-free Active Label Agnostic Domain Adaptation (WACV 2023)**](https://arxiv.org/abs/2205.12840)

Please cite our paper if you find it useful.

```
@inproceedings{kothandaraman2023salad,
  title={SALAD: Source-free Active Label-Agnostic Domain Adaptation for Classification, Segmentation and Detection},
  author={Kothandaraman, Divya and Shekhar, Sumit and Sancheti, Abhilasha and Ghuhan, Manoj and Shukla, Tripti and Manocha, Dinesh},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={382--391},
  year={2023}
}
```


The code structure is as follows:

Active Learning heuristics: <br>
AL/badge_full.py - BADGE active learning sampler <br>
AL/entropy.py - Entropy based active learning sampler <br>
AL/gradient.py - Gradient based active learning sampler <br>

Model folder: <br>
model/attention.py - contains code for guided spatial attention and guided channel attention modules <br>
model/deeplab_multi.py - contains code for the entire SALAD model for semantic segmentation using DeepLab backbone architecture <br>
model/sfda_net.py - contains code for the modulation network <br>

train.py - training script for our model for the task of semantic segmentation <br>
eval_segmentation.py - evaluation script for our model for thetask of semantic segmentation <br>
al_sampler.py - script for choosing samples using our active learning algorithm <br>

