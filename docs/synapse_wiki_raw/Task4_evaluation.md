# Evaluation

Synapse wiki: syn74274097/wiki/639595
Modified: 2026-04-07T11:45:23.189Z

To measure the performance of the contributions, we will evaluate the quality of the inpainted regions. We will use the following set of well-established metrics to quantify how realistic the synthesized image regions are compared to real ones:
* structural similarity index measure (SSIM)
* peak-signal-to-noise-ratio (PSNR)
* mean-square-error (MSE)

See this  [GitHub repository](https://github.com/BraTS-inpainting/2026_challenge/tree/main/evaluation) for the metrics' implementation. For the final ranking of the MICCAI challenge, an equally weighted rank-sum is computed for each case of the test set, considering all the aforementioned metrics. This will rank algorithms for each case according to each metric and then sum up all ranks.  The  [manuscript](https://arxiv.org/pdf/2305.08992.pdf)  illustrates the evaluation procedure in more detail.

