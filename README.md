# Water level prediction from social media images with a multi-task ranking approach

This is an implementation of [paper](https://arxiv.org/abs/2007.06749) and [alternate link](https://doi.org/10.1016/j.isprsjprs.2020.07.003) on Python 3 and PyTorch.

In earlier work (Chaudhary et al., 2019) we have presented a model to predict flood height from images gathered from social media platforms in a fully automated way using a deep learning framework. The proposed model performed object instance segmentation and predicted flood level whenever an instance of some specific object was detected. Although the trained model performs rather well, the effort required to build a large, pixel-accurate annotated dataset for instance segmentation of flood images is considerable. To tackle this problem, we propose in this paper a deep learning approach where we define the flood estimation as a per-image regression problem and combine it with a ranking loss to further reduce the labelling load. We propose to avoid the tedious, and hardly scalable, procedure of pixel-accurate object instance labelling per image by 

* directly regressing one representative water level value per image and, more importantly, 
* exploiting relative ranking of the water levels in pairs of images, which is much easier to annotate.

To gain access to the dataset please send us an email at priyanka.chaudhary@geod.baug.ethz.ch. 


