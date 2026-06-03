# 2. Multi-Consortium International Pediatric Brain Tumor Segmentation

Synapse wiki: syn74274097/wiki/639578
Modified: 2026-04-06T13:43:02.106Z

Brain tumors are among the deadliest types of cancer, and the BraTS Challenge has a successful history of creating benchmark datasets and evaluation frameworks for segmentation and analysis of aggressive primary brain tumors. Although rare, pediatric brain and central nervous system tumors are the leading cause of
cancer-related mortality in children and present distinct biological, clinical, and imaging characteristics compared to adult tumors. Pediatric diffuse midline gliomas (DMGs), for example, are high-grade tumors with poor prognosis, typically arising in the pons during early childhood, and often lack the classic imaging features observed in adult GBM, such as well-defined contrast enhancement and necrosis. These differences underscore the need for pediatric-specific imaging datasets and AI methods tailored to this population.

Building on the inclusion of pediatric DMGs in the BraTS 2022 test set and the success of the multi-institutional BraTS-PEDs 2023-2025 challenges, the BraTS-PEDs initiative continues to expand in scope and scale. For the BraTS-PEDs 2026 challenge, we will include both treatment-naïve and post-treatment pediatric brain tumor data. The treatment-naïve cohort comprises 457 pediatric subjects that have been publicly released through The Cancer Imaging Archive ([TCIA BraTS-PEDs](https://www.cancerimagingarchive.net/collection/brats-peds/)). In addition, the 2026 challenge introduces a longitudinal post-treatment cohort consisting of 35 patients with at least three MRI timepoints per subject (105 total post-treatment cases), enabling the evaluation of segmentation robustness in the setting of therapy-related changes.

The dataset is collected through multiple international consortia and institutions, including the Children’s Brain Tumor Network (CBTN) and the DIPG/DMG Registry, and includes multiparametric MRI with expert-validated tumor subregion annotations. Participants will develop, containerize, and submit their algorithms via the Synapse platform, where models will be evaluated on unseen validation and testing cohorts under a standardized, reproducible framework. By incorporating both cross-sectional treatment-naïve imaging and longitudinal post-treatment data, BraTS-PEDs 2026 aims to advance robust, clinically relevant pediatric brain tumor segmentation methods and to better reflect real-world clinical scenarios encountered in pediatric neuro-oncology.


### Overview

${image?fileName=Figure1%2Epng&align=Center&scale=100&responsive=true&altText=}
The **BraTS-PEDs challenge** would not be possible without the incredible support of multi-disciplinary, national and international societies and consortia.  Image data of high grade pediatric brain tumors were collected through multiple leading consortia  in the field of pediatric neuron-oncology, including the Children’s Brain Tumor Network ([CBTN](https://cbtn.org/)), and the International DIPG/DMG Registry ([DIPGr](https://www.dipgregistry.org/)). Additional data from participating pediatric institutions listed on this website were included in the cohort. Data annotation was provided with the participation of the American Society of Neuroradiology ([ASNR](https://www.asnr.org/)).  Last, but not least, we thank the Medical Image Computing and Computer Assisted Intervention ([MICCAI](http://www.miccai.org/)) Society for being the home of our challenge. The organizers are immensely grateful for everyone's timely and committed support!

The challenge will use the Synapse platform. These data will be used to develop, containerize, and evaluate their algorithms in unseen validation data until the end of **July 2026** when the organizers will stop accepting new submissions and evaluate the submitted algorithms in the pediatric patient population.

### Task: Segmentation of Pediatric Brain Tumor Subregions

####! **Label Description**

The segmentation of pediatric brain tumors into four main subregions was recommended by RAPNO working group for evaluation of the treatment response in high-grade gliomas and DIPGs.

For **BraTS-PEDs 2026 Challenge**, the goal is to automatically segment these pediatric brain tumor subregions using a 4-label system:

* **Enhancing tumor (ET; label 1):** Areas with enhancement (brightness) on contrast-enhanced T1 sequences (T1C) as compared to pre-contrast T1 (T1N) sequences. In case of mild enhancement, checking the signal intensity of normal brain structure can be helpful.

* **Nonenhancing tumor (NET; label 2):** Any other abnormal signal intensity within the tumorous region that cannot be defined as enhancing or cystic. For example, the abnormal signal intensity on T1N, T2F, and T2W sequences that is not enhancing on T1C sequences should be considered as nonenhancing portion.

* **Cystic component (CC; label 3): ** Typically appearing with hyperintense signal (very bright) on T2W sequences and hypointense signal (dark) on T1C sequences. The cystic portion should be within the tumor, either centrally or peripherally (as compared to ED which is peritumoral). The brightness of CC is here defined as comparable or close to cerebrospinal fluid (CSF). 

* **Peritumoral edema (ED; label 4): ** Abnormal hyperintense signal (very bright) on T2F sequences. ED is finger-like spreading that preserves underlying brain structure and surrounds the tumor.

* **Tumor core (TC; label 1+2+3):** Including the part of the tumor that is typically resected (enhancing tumor, nonenhancing tumor, and cystic component). 

* **Whole tumor (WT; label 1+2+3+4):** Including all tumor subregions (tumor core and edema). 

### BraTS PEDs Challenge Manuscripts

* **2023:**
     * arXiv [https://arxiv.org/abs/2305.17033] 
     * DOI [https://doi.org/10.48550/arXiv.2305.17033] 
* **2024-2025:**
    * arXiv [https://arxiv.org/abs/2404.15009]
    * DOI [https://doi.org/10.48550/arXiv.2404.15009] 

### Organizers

####! **Chairs**

* **Marius George Linguraru, D.Phil., M.A., M.Sc. — [Co-Lead Organizer] ** <br/> Children’s National Hospital / George Washington University <br/> mlingura@childrensnational.org

* **Anahita Fathi Kazerooni, Ph.D., M.Sc. — [Co-Lead Organizer] ** <br/> Children’s Hospital of Philadelphia / University of Pennsylvania <br/> fathikazea@chop.edu

####! **Organizing Team **
Zhifan Jiang,  Ph.D.|Children’s National Hospital |
Xinyang Liu, Ph.D.|Children’s National Hospital |
Deep Gandhi, M.Sc.|Children's Hospital of Philadelphia|Data Preparation
Spyridon Bakas, Ph.D.|University of Indiana| BraTS Challenges
Mehdi Astaraki, Ph.D.|Karolinska Institute| BraTS Challenges
Ujjwal Baid, Ph.D.|Emory University
Keyvan Farahani, Ph.D.| National Institutes of Health
Jake Albrecht, Ph.D.| Sage Bionetworks
Verena Chung | Sage Bionetworks

####! **Data Approvers**
Arastoo Vossough, M.D.|Children's Hospital of Philadelphia|
Jeffrey B. Ware, M.D.|University of Pennsylvania
Ali Nabavizadeh, M.D.|University of Pennsylvania
Mariam Aboian|Children's Hospital of Philadelphia|

####! **Data Contributors**

* **For CBTN**
Adam Resnick, Ph.D.|Children's Hospital of Philadelphia
Brian Rood, M.D.|Children’s National Hospital
Ali Nabavizadeh, M.D.|University of Pennsylvania

* ** For DIPGr**
Peter de Blank, M.D.|Cincinnati Children's Hospital
Lindsey Hoffman,D.O.|Phoenix Children's Hospital
Trent Hummel, M.D.|Cincinnati Children's Hospital

* **For Boston Children's Hospital**
Benjamin Kann, M.D.| Dana-Farber Brigham Cancer Center and Boston Children’s Hospital
Tina Young Poussaint, M.D., FACR|Dana-Farber Brigham Cancer Center and Boston Children’s Hospital
Anna Zapaishchykova, M.D.|Dana-Farber Brigham Cancer Center and Boston Children’s Hospital

* **For Yale**
Mariam Aboian, M.D.| Children's Hospital of Philadelphia
Nazanin Maleki, M.D.| Children's Hospital of Philadelphia 

* **For Duke**
Evan Calabrese, M.D.| Duke University 
Avani Mangoli, M.D.| Duke University
Ethan Castellino| Duke University 


* **Other Data Contributors**
Miriam Bornhorst,M.D.|Children’s National Hospital
Roger Packer, M.D.|Children’s National Hospital
Maryam Fouladi, M.D.|Nationwide Children's Hospital
Margot Lazow,M.D.|Nationwide Children's Hospital
Michelle Deutsch,M.S.|Nationwide Children's Hospital
Leonie Mikael,Ph.D.|Nationwide Children's Hospital

####! **Data Annotators**
Nastaran Khalili, M.D. | Moderator
Neda Khalili, M.D. | Moderator
Wenxin Tu, BSc. | Moderator
Shuvanjan Haldar, BSc.| Moderator
Bhavyasri Vunnava| Moderator
Ibraheem Shaikh, MD.| Annotator

* **For ASNR**
Aaron McAllister, MD.| Annotator
Andres Felipe Rodriguez, MD.| Annotator
Khanak Nandolia, MD.| Annotator
Mariana Sánchez Montaño, MD.| Annotator
Nakul Sheth, MD.| Annotator
Sanjay Prabhu, MD.| Annotator


