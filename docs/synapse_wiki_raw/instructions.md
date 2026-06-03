# Submission Instructions

Synapse wiki: syn74274097/wiki/639582
Modified: 2026-05-27T19:53:21.268Z

BraTS 2026 features 5 challenge tasks. You are welcome to participate in just one, or tackle all of them!

This guide outlines the complete submission lifecycle for this challenge:

1. develop and fine-tune your algorithm (**File Predictions**)
2. document your methodology (**Short Paper**)
3. submit your final algorithm for official ranking (**Containerized Algorithm**)

Please read the sections relevant to your chosen task(s) carefully, as data formats, spatial requirements, and execution environments vary.

### 🧠 File Predictions

<details>

<summary>Show instructions</summary>

To help fine-tune your algorithm before containerization, generate predictions locally using the provided validation data and upload them to Synapse for preliminary evaluation.

#### **Tasks 1 - 3: Segmentation Tasks (NIfTI)**

* For each task, the submission must be a **single zip (`*.zip`) or tarball (`*.tar[.gz]`)** containing your prediction files for that task.  You may name this compressed file whatever you like.

* All individual files must be in NIfTI format and use the `nii.gz` file extension.

* Submitted data files <u>must precisely match</u> the spatial characteristics of their corresponding images. The array dimensions, voxel spacing, image origin, and spatial orientation must be identical to the source images.  This is especially important for the Metastasis task, as this dataset is in its native space, which inherently includes variability. **Discrepancies will lead to scoring issues and submission invalidation.** You may use [CaPTk](https://cbica.github.io/CaPTk/) to verify and/or visualize your files.

* Output filenames **must end with** the 5-digit case ID and 3-digit timepoint (found in the folder name), followed directly by `.nii.gz`.

   * _Example: If the input folder is BraTS-MET-12345-100/, a valid output file could be `BraTS-MET-12345-100.nii.gz` or simply `12345-100.nii.gz`_

#### **Task 4: Inpainting Tasks (NIfTI)**

* Submission must be a **single zip (`*.zip`) or tarball (`*.tar[.gz]`)** containing all of your prediction files.  You may name this compressed file whatever you like.

* All individual files must be in NIfTI format and use the `nii.gz` file extension.

* All individual segmentations must have dimensions of 240x240x155 and an origin at [0, -239, 0]. You may use [CaPTk](https://cbica.github.io/CaPTk/) to verify and/or visualize your files.

* Output filenames **must end with** the 5-digit case ID and 3-digit timepoint (found in the folder name),  followed directly by `-t1n-inference.nii.gz`

    * _Example: If the input folder is BraTS-GLI-12345-000/, a valid output file could be `BraTS-GLI-12345-000-t1n-inference.nii.gz`_

#### **Task 5: Pathology Task (CSV)**

Generate a single 2-column CSV file with your predictions. You must name this file `predictions.csv`. Ensure it strictly adheres to the following format:

**Column Name** | **Column Type** | **Accepted Values** | **Example**
`SubjectID ` | str | Filename of the digitized tissue section (including `.jpg`) | `BraTSPath_2026_Val_12345.jpg`
`Prediction ` | int | One of: [0-9] where: <br/>- 0: CT <br/>- 1: DM <br/>- 2: IC <br/>- 3: LI <br/>- 4: MP <br/>- 5: NC <br/>- 6: PL <br/>- 7: PN <br/>- 8: WM <br/>- 9: NOTA | 1

</details>

---

### 📝 Short Paper

> **❗️ Important**: Submission of this short paper is **mandatory**. Failure to provide a short paper will disqualify your Docker submission from evaluation and final ranking.

<details>

<summary>Show instructions</summary>

All participants are encouraged to submit a short paper describing their methods and results via Microsoft CMT\*. The challenge proceedings will be distributed by Springer LNCS.

#### **Paper Rules and Format**

Your paper must be **8 - 10 page long** (without references), and use the [Springer LNCS format](bit.ly/2TEcZNF).

Your paper must include these sections:

- Abstract (no citations allowed)
- Appropriate keywords  
- Introduction
- Methods
- Results
- Discussion
- Acknowledgements (optional)
- References (**MUST** include the citations listed in the **Challenge Rules**)

_Note: Citations do not count toward the paper length limit. In addition to the required citations, you may add any additional you deem appropriate._


<small>\*The <a href="https://cmt3.research.microsoft.com">Microsoft CMT service</a> was used for managing the peer-reviewing process for this conference. This service was provided for free by Microsoft and they bore all expenses, including costs for Azure cloud services as well as for software development and support.</small>

#### **Linking Your Paper to Your Synapse Team**

When submitting through the BrainLes CMT system, you will be asked for your Synapse team name. Please provide the team you used to register for this challenge, so that we can link your short paper with your Docker submission.

#### **Submission Process**

**Paper submission**

Submit your manuscript using the button below:

> Link coming soon

<!--
${buttonlink?text=Link to CMT Submission System &url=https://cmt3.research.microsoft.com/BraTS2025/}
-->

**Copyright form**

Organisers will share the copyright form at the time of camera-ready submission.

 **Reviewer responsibilities**

After submissions close, you will be invited to conduct a light peer review of 2-3 papers for potential errors or inappropriate language. Please review these papers with the following two questions in mind:

- Is there anything missing in the methodological description?
- Is there anything currently missing that the authors could address within a week to improve their paper?

**Publication withdrawal**

If you do not want your paper published, please email us at BraTSChallengeOrganizers@synapse.org. We will remove your paper from publication and will not ask you to review others.

_Note: If you do not want your paper published as part of the LNCS proceedings, you will also not be included as a co-author on the [BraTS Challenge journal manuscript](https://arxiv.org/abs/2107.02314) submission._

</details>

---

### 🐳 Containerized Algorithm

> **❗️ Important:** Please confirm your short paper is submitted on CMT and your Synapse team name in that system is correct. We will only run Docker submissions linked to a short paper (see above for more details).

<details>

<summary>Show instructions</summary>

To be considered for final ranking, participants must submit a containerized version of their algorithm as a Docker image.

#### **General Rules** 

Your container will be evaluated in an automated environment under the following conditions:

* **Zero network access**: Submissions are run without internet. All dependencies must be installed within your Dockerfile, not via runtime scripts.

* **Input data**: The hidden test dataset will be mounted to the absolute path `/input`. It is strictly <u>read-only</u>. Any attempt by your model to write to or modify the `/input` directory will crash your run, thus invalidating your final submission

* **Expected output**: Your model must write all final predictions directly to the absolute path `/output` as a flat structure

#### **Template Resources**

* Tasks 1 - 3: _Coming soon_

* Task 4: You can use our provided [Jupyter Notebook](https://github.com/BraTS-inpainting/2026_challenge/blob/main/evaluation/evaluation.ipynb) to start generating predictions and to score your code locally.

* Task 5: _Coming soon_

#### **Compute Constraints**

| **Hardware** | **Time Limit** (for total inference)
**Tasks 1 - 3** (segmentation) | - NVIDIA A10G GPU (24GB VRAM)<br/>- 32 GiB RAM<br/>- 200 GB storage | 8 hours
**Task 4** (inpainting) | _TBA_ | _TBA_
**Task 5** (classification) | - NVIDIA A6000 GPU (48GB VRAM)<br/>- 128GB RAM<br/>- 512GB storage | 24 hours

#### **Execution Pipeline** 

* **Tasks 1 - 4**

   Your model must iterate through each folder in `/input` and generate a single `.nii.gz` prediction file per case. You must write all files directly to `/output` as a flat structure. Do NOT create sub-folders; doing so will invalidate your final submission.

   For example:
   
   ${image?fileName=brats%5Finput%5Foutput%2Epng&align=None&scale=75&responsive=true&altText=}
      
   _Note: The example above shows the expected naming format for the segmentation tasks. Ensure you use the correct suffix if participating in the Inpainting task._


* **Task 5**

   Your model must process every sample found in `/input` and compile all predictions into a single file named `predictions.csv` (formatted exactly as described in the Validation Round) and save it directly to `/output/predictions.csv`

</details>

---

### Ready to Submit?

For file and Docker submissions, navigate to the "Submission" section of your chosen **Task** tab, or use the direct links below:

* [Task 1 - Brain Metastases](https://challenges.synapse.org/Challenges/DetailsPage/Task1?id=syn74274097#Submission)
* [Task 2 - Pediatric Tumors](https://challenges.synapse.org/Challenges/DetailsPage/Task2?id=syn74274097#Submission)
* [Task 3 - Generalizability](https://challenges.synapse.org/Challenges/DetailsPage/Task3?id=syn74274097#Submission)
* [Task 4 - Inpainting](https://challenges.synapse.org/Challenges/DetailsPage/Task4?id=syn74274097#Submission)
* [Task 5 - Pathology](https://challenges.synapse.org/Challenges/DetailsPage/Task5?id=syn74274097#Submission)

For short paper submissions, please follow the instructions listed under "📝 Short Paper".

### Need Help?

If you have any questions, encounter technical issues, or need clarification on any of the submission requirements, please feel free to ask in the **Community** tab. The organizers and fellow participants are there to help!

