# 4. MR Image Inpainting for BraTS

Synapse wiki: syn74274097/wiki/639580
Modified: 2026-04-07T12:03:35.702Z

We invite participants to develop innovative algorithms to synthesize 3D healthy brain tissue in the region affected by glioma, a type of brain tumor. We frame this challenge as an inpainting task, where the goal is to realistically fill the missing tumor area within MRI scans.
![inpainting challenge](https://github.com/BraTS-inpainting/2025_challenge/raw/main/3lions.png)

**Inpainting for Medical Imaging**: Inpainting is a well-established technique in computer vision, with successful algorithms designed to fill missing regions in 2D images. However, applying these methods to 3D medical images like MRI scans remains an open question. This challenge provides a platform to explore this potential and benchmark new inpainting techniques for brain tumor analysis.

**Clinical Significance**: Many algorithms for automatic brain MRI analysis support clinical decision-making. However, these algorithms often struggle with images containing tumors, as they are typically trained on healthy brains. This challenge addresses this limitation by focusing on synthesizing healthy tissue representations in tumor-affected regions. This could significantly improve the performance of existing algorithms for tasks like, but not limited to:

* Brain anatomy parcellation
* Tissue segmentation
* Brain extraction
* Brain tumor growth modeling

By overcoming the challenges of tumor-affected scans, we aim to improve the accuracy and reliability of automated brain image analysis in clinical settings.

**Part of MICCAI 2026**: We are thrilled to announce that this challenge, like last year, is being held as part of the MICCAI conference! We encourage participation from the medical imaging and computer vision communities to advance the field of brain tumor analysis.

### Task

The participants' task is to create models that synthesize healthy brains from voided brains accompanied by an inpainting mask, see figure.

${image?fileName=the%5Ftask%5Fwith%5Flines%2Epng&align=None&scale=100&responsive=true&altText=task}

To illustrate the task in more detail, we created a [GitHub repository](https://github.com/BraTS-inpainting/2026_challenge/tree/main/baseline) training a baseline model.

### Manuscript / Citation
Please see our [manuscript](https://arxiv.org/pdf/2305.08992.pdf) on arxiv for further details. Challenge participants who submit interesting algorithms will be invited to co-author the manuscript.

### Prizes
Besides fame, recognition, and the opportunity to contribute to the manuscript challenge, winners can expect special awards to be announced soon.

### Connect
All relevant challenge communication will be announced via [Twitter](https://twitter.com/BraTS_inpaint) and in the [challenge thread](https://www.synapse.org/Synapse:syn64153130/discussion/threadId=11890)  on the Synapse Forums. Consider following us to stay current. 

### Questions / Feedback
For questions and feedback regarding the challenge, please visit the **Community** tab.
For inquiries not suitable for the forum, please write to florian.kofler [at] tum.de

### Organizers
A multi-institutional and international team organizes the challenge. For a full list of organizers, please see our [manuscript](https://arxiv.org/pdf/2305.08992.pdf).

