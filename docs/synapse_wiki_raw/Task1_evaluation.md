# Evaluation

Synapse wiki: syn74274097/wiki/639602
Modified: 2026-04-08T14:40:59.446Z

_The following metrics will be used for assessing the algorithm:_

In terms of segmentation evaluation metrics, we use the following subject-wise metrics: 
i) Dice Similarity Coefficient (DSC), which is commonly used in the assessment of segmentation performance
ii) Normalized Surface Distance (NSD), which introduces a tolerance parameter and is complementary to traditional metrics such as DSC. 

In terms of lesion detection evaluation, we use the following lesion-wise metrics:  
i) F1 score – the harmonic mean of precision and recall, to determine whether an algorithm has the tendency to over- or undersegment
ii) AUC over multiple F1 scores, obtained at different detection threshold values 
* We apply the detection evaluation for every single lesion within one MRI study. For the detection arm of the challenge we define a lesion as a collective term including the Enhancing Tumor, the Non-enhancing Tumor Core and the Resection cavity. 
* We will apply the segmentation evaluation metrics only for lesions larger than 275 mm3 (The images in the dataset are co-registered to 1mm slice thickness) 

_Ranking details:_
We will follow the DELPHI-based recommendations for image analysis validation [1,2], incorporating i) algorithmic ranking, and ii) statistical significance testing. For ranking of multidimensional outcomes (or metrics), for each team, we will compute the summation of their ranks across the average of the metrics described above as a univariate overall summary measure. This measure will decide the overall ranking for each specific team. All teams will then be placed in a ranked order and their average rankings will be randomly permuted (i.e., 500,000 permutations), in a pair-wise manner. Corresponding pairwise p-values will be computed to determine the pair-wise statistical significance and report actual differences between the ordered ranked approaches. These p-values will be reported in an upper triangular matrix revealing the statistical insignificance of potential teams that will be grouped together in tiers and the significant superiority among others that we will clearly indicate. This is an evolved version of the systematic ranking that has been used on previous years for BraTS and other challenges, and will be packaged & distributed as an independent tool allowing reproducibility and use in other challenges.

For the cases in which the algorithm fails to produce a result metric for a specific test case, there will be no penalties, i.e. the metric **won't** be set to its worst possible value (e.g., 0 for the DSC and the NSD). 

[1] Reinke et al. Understanding metric-related pitfalls in image analysis validation. Nat Methods. 2024 Feb;21(2):182-194.
[2] Maier-Hein et al. Metrics reloaded: recommendations for image analysis validation. Nat Methods. 2024 Feb;21(2):195-212.
