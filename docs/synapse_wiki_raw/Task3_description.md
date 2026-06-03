#  3. BraTS-GoAT: Segmentation Generalizability of Brain Tumor Sub-regions Across Tumors in Pre-operative MRI

Synapse wiki: syn74274097/wiki/639579
Modified: 2026-04-08T13:53:30.945Z

The International Brain Tumor Segmentation (BraTS) challenge has focused, since its inception in 2012, on establishing a benchmarking environment and a dataset for delineating adult brain gliomas. The focus of the BraTS 2026 challenge remained the same: generating a standard benchmark environment. At the same time, the dataset expanded into explicitly addressing 1) the same adult glioma population, as well as 2) the underserved sub-Saharan African brain glioma patient population, 3) brain/intracranial meningioma, 4) brain metastasis, and 5) pediatric brain tumor patients.

Although segmentation is the most widely investigated medical image processing task, the various challenges have been organized to focus only on specific clinical tasks. In this challenge, the BraTS Generalizability Across Tumors, we will be focusing on assessing the algorithmic generalizability beyond each individual patient population and focusing across all of them. The hypothesis is that a method capable of performing well on multiple segmentation tasks will generalize well to unseen tasks.

Participants can decide whether to explicitly participate in this part of the competition or just in one/some of the previous tasks/challenges. Regardless of the participant’s choice (which might be driven by accessibility to computational resources), the organizers of the challenge will require the submission of a containerized algorithm that could be retrained by the organizers in the complete dataset, enabling the fair comparison across all submission methods in the test datasets across the Tasks/Challenges.

We challenge participants to create a segmentation algorithm capable of adapting and generalizing to different brain tumors with limited information and data on the target classes. Our aim is to simulate a clinical scenario in which we develop a segmentation tool agnostic to future clinical applications (i.e., a tool trained on specific diseases that will be applied to new ones with limited access to additional training data).

Specifically, the candidate algorithms should be able to generalize across:
- Lesion types (i.e., different number of lesions per scan, lesion sizes, and locations in the brain).
- Institutions (i.e., different MRI scanners, acquisition protocols).
- Demographics (i.e., different age, sex, etc.).

Additionally, lesions will differ in imaging features; for example, some will miss or have limited necrotic core, edema, or contrast enhancement. So, while the segmentation mask labels will be **consistent** across disease types (i.e., the necrotic core, edema, and contrast enhancement masks will always have the same value), the presence of each label will vary across the training, validation, and test data.

### Task: Tumor Sub-region Segmentation

The participants are called to address this task by using the provided clinically acquired training data to develop their method and produce segmentation labels of the different tumor sub-regions. **The sub-regions considered for evaluation are the "enhancing tumor" (ET), the "tumor core" (TC), and the "whole tumor" (WT)**.

The **ET** is described by areas that show hyper-intensity in post-contrast T1 (T1Gd) when compared to pre-contrast T1, but also when compared to "healthy" white matter in T1Gd. The **TC **describes the bulk of the tumor, which is what is typically resected. The TC entails the tumor's ET and the necrotic (NCR) parts. The appearance of NCR is typically hypo-intense in T1-Gd when compared to T1. The **WT** describes the complete extent of the disease, as it entails the TC and the peritumoral edematous/invaded tissue (ED), typically depicted by a hyper-intense signal in FLAIR.

The provided segmentation labels have values of:

* 1 for NCR (necrosis)
* 2 for ED (edema/invaded tissue)
* 3 for ET (enhancing tumor)
* 0 for everything else.

**Note**: as a reminder, while the labels will be consistent across tumor types in all data subsets (i.e., training, validation, and test sets), the presence of each label (i.e., tumor sub-region) in each exam can vary within and between tumor types.

###  Organizers

* **Ujjwal Baid, Ph.D.  ** <br/> Emory University, GA, USA  <br/> ubaid@emory.edu

* **Gian Marco Conte, M.D. Ph.D.** <br/> Mayo Clinic, MN, USA <br/> giemmecci@synapse.org

* **Spyridon Bakas, Ph.D. ** <br/> University of Pennsylvania, PA, USA <br/> sbakas@synapse.org

