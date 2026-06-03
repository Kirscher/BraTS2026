# 1. Segmentation of Pre- and Post-Treatment Brain Metastases

Synapse wiki: syn74274097/wiki/639577
Modified: 2026-05-01T22:20:35.251Z

-> ${image?fileName=BraTS%5F2026%5Flogo%5Fbiorender%2Epng&scale=50&responsive=true&altText=} <-

**Issue:**
1. Monitoring metastatic brain disease is laborious and time-consuming, especially when managing multiple metastases and relying on manual techniques.
2. Brain metastases are commonly assessed by measuring their largest unidimensional diameter according to RANO-BM guidelines. However, accurate volumetric estimates of the lesions and surrounding edema are essential for effective clinical decision-making and to improve treatment outcome predictions.
3. Brain metastases are frequently small, complicating the detection and segmentation of lesions smaller than 10 mm. Such tasks have previously resulted in low dice similarity coefficients.


**Proposed solution:**
* To use a **machine learning approach to automatically detect and segment brain metastases, perilesional edema and resection cavities**, enhancing time efficiency, reproducibility, and robustness against inter-rater variability.


**Impact:**
* This challenge aims to provide **essential algorithms that can be utilized in both current and post-treatment settings**, potentially revolutionizing the management and monitoring of patients.

---

### Task

** Develop a versatile autosegmentation algorithm that reliably detects and precisely delineates brain metastases of varying sizes and is applicable for both pre- and post-treatment cases.**

${image?fileName=BraTS%5F2025%5FTask%5FFigure%2Epng&align=Center&scale=100&responsive=true&altText=}

For BraTS 2026 Brain Metastases, the following 4-label system is used:

* **Nonenhancing tumor core (NETC; Label 1):** All portions of tumor core without contrast enhancement that are enclosed by enhancing tumor (ET). It represents the  bulk of the tumor, which is what is typically considered for surgical excision.

* **Surrounding non-enhancing FLAIR hyperintensity (SNFH; Label 2):** Peritumoral edematous and infiltrated tissue, defined by the abnormal hyperintense signal envelope on the T2 FLAIR volumes, which includes the infiltrative non enhancing tumor, as well as vasogenic edema in the peritumoral region. Non tumor related FLAIR signal abnormality such as prior infarcts or microvascular ischemic white matter changes are NOT included.

* **Enhancing Tumor (ET; Label 3):** All tumor portions with noticeable contrast enhancement on postcontrast T1-weighted images.  Adjacent blood vessels, bleeding or intrinsic T1 hyperintensity are NOT included in this label.

* **Resection Cavity (RC; Label 4):** Delineates the resection of region within the brain in post-treatment cases.

For 2026, we will add a detection leaderboard, with the intention of promoting algorithms sensitive in detecting lesions. This is clinically relevant to cases of small (<27mm^3) lesions that either need to be counted as separate entities, or that they need to be quantified separately. 

{row}

{column width=4 height=4}
{column}

{column width=4 height=4}
${image?fileName=Mariam%5FAboian%5Fresize%2Ejpeg&align=Center&scale=93&responsive=true&altText=}
->**Mariam Aboian, MD/PhD <br/>Lead Co-Organizer** <br/>Department of Radiology <br/> Children's Hospital of Philadelphia<-
{column}

{row}
{column width=4 height=4}
${image?fileName=profile%2Dpicture%2Epng&align=None&scale=100&responsive=true&altText=}
 ->**Nikolay Yordanov, MD  <br/>Co-Organizer** <br/>Faculty of Medicine <br/>Medical University - Sofia <br/>Sofia, Bulgaria<-
{column}

{column width=4 height=4}
${image?fileName=Maleki%5FNazanin%2Ejpg&align=None&scale=93&responsive=true&altText=}
 ->** Nazanin Maleki, MD <br/>Co-Organizer** <br/> Department of Radiology  <br/>Children’s Hospital of Philadelphia (CHOP) <-
{column}

{column width=4 height=4}
${image?fileName=raisa%2Ejpeg&align=None&scale=107&responsive=true&altText=}
 ->**Raisa Amiruddin, MBBS <br/>Co-Organizer ** <br/> Department of Radiology <br/>Children’s Hospital of Philadelphia (CHOP) <-
{column}

{row}
{column width=4 height=4}
${image?fileName=Fabian Umeh%2Epng&align=None&scale=100&responsive=true&altText=}
 ->**Fabian Umeh  <br/>Co-Organizer**  <br/>Teesside University, UK <-
{column}

{column width=4 height=4}
${image?fileName=Headshot%5FCrystalChukwurah%2Ejpeg&align=None&scale=100&responsive=true&altText=}
 ->**Crystal Chukwurah  <br/>Co-Organizer** <br/>Medical Student<br/>Yale School of Medicine <-
{column}

{column width=4 height=4}
${image?fileName=1769399087620%2Epng&align=None&scale=100&responsive=true&altText=}
 ->**Monika Pytlarz <br/>Co-Organizer ** <br/> PhD Student <br/>Sano – Centre for Computational Personalised Medicine<-
{column}


