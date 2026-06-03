# Data

Synapse wiki: syn74274097/wiki/639600
Modified: 2026-04-28T21:24:08.298Z

The BraTS 2026 Brain Metastases dataset consists of a retrospective compilation of **pre- and posttreatment brain metastases mpMRI scans obtained from various institutions** under standard clinical conditions. This array of data, collected from different equipment and imaging protocols, offers a broad spectrum of image quality, thereby reflecting the **diverse clinical practices across institutions**. For the purposes of the challenge, we are utilizing the dataset used in 2025, so all the images from the training and testing sets are annotated. In 2026, we will be performing a Quality Control on all the annotations within the dataset. In addition to that,  we will also be offering to the participants non-annotated cases to facilitate innovations in semi-supervised learning approaches. 

The dataset comprises multiparametric MRI (mpMRI) scans, which include the following series:
* **pre-contrast T1-weighted (T1W)**
* **post-contrast T1-weighted (T1C)**
* ** T2-weighted (T2W)**
* **T2-weighted Fluid Attenuated Inversion Recovery (FLAIR)**

In 2025, T2W became non-mandatory in BraTS-METS. Some cases have native T2, some have synthetic T2, some don't have T2. All imaging volumes were segmented using the STAPLE fusion of different brain metastases segmentation algorithms. These fused **labels were then manually refined** by neuroradiology experts of varying rank and experience, adhering to a consistently communicated annotation protocol. Experienced board-certified attending neuroradiologists **approved the manually refined annotations**.

###! Datasets included in the challenge

**Dataset**|**Training**|**Validation**|**Testing**|**Registered in:**|**Contributor**|**Institution**|**Annotated**
Duke|37|15|30|SRI24 space|Devon Godfrey PhD <br/> Scott Floyd MD/PhD|Duke University|yes
NCI|35|n/a|1|SRI24 space|Ayda Youssef MD|National Cancer Institute|yes
Missouri|22|25|35|SRI24 space|Nourel hoda Tahon MD, Msc <br/> Ayman Nada MD/PhD|University of Missouri|yes
WashU|39|2|12|SRI24 space|Satrajit Chakrabarty|Washington University|yes
Yale|195|n/a|12|SRI24 space|Mariam Aboian MD/PhD|Yale university|yes
UCSF|322|n/a|n/a|Native space|Jeffrey Rudie MD|University of California, San Francisco|yes
NW|n/a|46|n/a|SRI24 space|Yuri S. Velichko PhD|Northwestern University|yes
UCSD|646|91|213|Native space|Maria Correia de Verdier MD <br/> Jeffrey Rudie MD/PhD|University of California, San Diego|yes
Ulm|200|0|0|Native space|Nico Sollman MD/PhD|Ulm University, Germany|no
**In total**|**1496**|**179**|**303**|**n/a**|**n/a**|**n/a**|**yes**

___

_Additional details:_

**The University of California San Diego Brain Metastases Longitudinal MRI Dataset**
There are currently 646 training cases in this dataset. The dataset comprises progressive, longitudinal data. There could be cases that have been subject to non-surgical treatment, i.e. there could be cases with empty masks. Disclaimer: Before releasing this dataset, a QC will be performed. We will run one of the algorithms from the BraTS 2025 winners and one of the algorithms that was never trained on BraTS data (Rudie et al., 2021) on the training data and identify cases with dice <1; these cases will be re-annotated. 

_Image registration: _

The BraTS 2025 Metastases dataset consists of a mix of cases in native space, co-registered to T1C 1mm^3 and cases registered in SRI24 space. All cases provided by Ulm University, UCSF and UCSD are in native space, totaling 1268 cases. In contrast, the remaining cases are registered in SRI24 space, amounting to 328.

Registering neuroimaging cases in a common space, such as SRI24, allows for a consistent anatomical reference that facilitates comparisons across different subjects, studies, and datasets. However, it is more natural for radiologists to review cases in their native space, as interpolation can distort images and obscure small lesions.

---

###! Data Access

In addition to registering for the challenge, you must also request access to the data:

1. **Submit the form:** complete the [Data Access Google form](https://forms.gle/UiCpXos2zKFPdMnK6). Only one access form is needed across all 5 challenge tasks and their training + validation datasets\*.

2. **Accept your invite:** once your details are verified, the BraTS Service Account will email you an invitation to join the @BraTS2026DataAccessTeam . Accept it to unlock the files below!

<small> \* *Note: test datasets and validation GT labels will not be released to the public.*</small>

