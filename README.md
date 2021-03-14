# Water level prediction from social media images with a multi-task ranking approach

This is an implementation of [paper](https://arxiv.org/abs/2007.06749) and [alternate link](https://doi.org/10.1016/j.isprsjprs.2020.07.003) on Python 3 and PyTorch.

In earlier work (Chaudhary et al., 2019) we have presented a model to predict flood height from images gathered from social media platforms in a fully automated way using a deep learning framework. The proposed model performed object instance segmentation and predicted flood level whenever an instance of some specific object was detected. Although the trained model performs rather well, the effort required to build a large, pixel-accurate annotated dataset for instance segmentation of flood images is considerable. To tackle this problem, we propose in this paper a deep learning approach where we define the flood estimation as a per-image regression problem and combine it with a ranking loss to further reduce the labelling load. We propose to avoid the tedious, and hardly scalable, procedure of pixel-accurate object instance labelling per image by (i) directly regressing one representative water level value per image and, more importantly, (ii) exploiting relative ranking of the water levels in pairs of images, which is much easier to annotate.

Moving from pixel-accurate object delineation as in Chaudhary et al. (2019) to annotating only a single water depth per image comes at a price. While the regression task might, in principle, be easier than detailed object detection and segmentation, the supervision signal for a machine learning system is much weaker (e.g., we no longer tell the system to turn its attention to certain types of objects that reoccur with similar metric height). Furthermore, even in the presence of known objects it is often hard for a human operator to determine the water depth in individual images on an absolute scale. On the contrary it is a much simpler task to rank images via pairwise comparisons. People can, with no or little training, quickly decide which of two images shows a higher water level. In this way it becomes feasible to outsource the labelling effort to large groups of untrained annotators, for instance through an online tool. Using ranking as a complementary task can be seen as a variant of weak supervision, or alternatively the ranking information can be interpreted as a regulariser for the otherwise data-limited regression task. The idea is that a large volume of weaker ranking labels should be able to largely compensate for the small amount of strong water depth labels, and lead to better regression performance. 

To achieve that, we proposed:

(i) We propose a deep learning approach that learns to estimate water level from social media images by combining water level regression with a relative ranking of image pairs. The water level regression part is fully supervised while pairwise image ranking adds a weak supervision signal to improve overall accuracy. The general idea is that the fully supervised signal (i.e., water level regression) from a small, expensive label set is supported by a closely related, weak supervision signal (i.e., pairwise water level ranking), where collecting large amounts of labels is cheap.

(ii) We introduce a new, large-scale dataset DeepFlood with 8000 images. DeepFlood is comprised of two sub-datasets called DF-OBJ and DF-IMG which we use for our regression and ranking sub-tasks respectively. We make all data available on request via email to one of the authors of this paper. 
