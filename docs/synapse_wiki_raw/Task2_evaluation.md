# Evaluation

Synapse wiki: syn74274097/wiki/639598
Modified: 2026-03-28T00:46:52.021Z

The evaluation of **aforementioned six subregions** follows the general methodology established by other BraTS Lighthouse Challenges, using two primary metrics:
* **Lesion-wise Dice Similarity Coefficient (DSC)**: Measures voxel-level segmentation overlap between predicted and reference segmentations on a lesion-by-lesion basis, explicitly ignoring true-negative voxels.
* **Normalized Surface Distance (NSD)**: Assesses boundary-level overlap between predicted and reference segmentations.
The lesion-wise DSC metric specifically evaluates performance at the individual lesion level rather than across the entire image. This approach prevents bias toward models that preferentially detect larger lesions—a common limitation associated with conventional DSC calculations. By evaluating predictions lesion-by-lesion, we can more accurately determine a model’s effectiveness in segmenting multi-focal and multi-centric disease.

