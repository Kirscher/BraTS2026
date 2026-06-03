# 5. Assessing the Heterogeneous Histologic Landscape of Glioma

Synapse wiki: syn74274097/wiki/639581
Modified: 2026-04-08T19:41:59.284Z

${image?fileName=Untitled%2Epng&align=Center&scale=100&responsive=true&altText=}

### Overview

Glioblastoma is the most common malignant primary parenchymal tumor of the brain. Clinically, glioblastoma has a grim prognosis with unusually short duration antecedent symptoms and median survival of 12-18 months. This tumor is widely infiltrative in the cerebral hemispheres and well-characterized by heterogeneous molecular profiles as well as histopathologic features. This molecular and micro-environmental landscape heterogeneity is a major obstacle in treating these tumors. Correctly diagnosing these tumors and assessing their heterogeneity is crucial for choosing the precise treatment and potentially enhancing patient survival rates. In the gold-standard histopathology-based approach to tumor diagnosis, detecting various morpho-pathological features of distinct histology throughout digitized tissue sections is crucial. Such ‘features’ include the presence of cellular tumors, geographic necrosis, pseudopalisading necrosis, areas abundant in microvascular proliferation, infiltration into the cortex, wide extension in subcortical white matter, leptomeningeal infiltration, regions dense with macrophages, and the presence of perivascular or scattered lymphocytes.

The BraTS-Pathology challenge aims to develop deep-learning models capable of identifying tumor sub-regions of distinct histologic profiles. These models aim to assist in the diagnosis and grading of conditions in a consistent manner.

### Task:

The tissue sections exhibit various features indicative of the diagnosis of glioblastoma and have been annotated by expert neuropathologists. These annotated regions are divided into same-size patches, each representing either a specific class present in that patch. Individual patches are treated as individual cases in this challenge to enable a deeper analysis and provide a deeper understanding of these distinct features/classes in a finer-grained resolution. 

Summary of specific histologic areas of interest (Each representing one class):

 i) presence of cellular tumor (CT)
ii) pseudopalisading necrosis (PN) 
iii) areas abundant in microvascular proliferation (MP)
iv) geographic necrosis (NC)
v) infiltration into the cortex (IC)
vi) penetration into white matter (WM)
vii) Leptomeningial Infiltration (LI)
viii) Regions with Dense Macrophages (DM)
ix) Presence of Lymphocytes (PL)
x) None of the above (NOTA)

### Manuscript

arXiv: BraTS-Path Challenge: Assessing Heterogeneous Histopathologic Brain Tumor Sub-regions : (Paper)[https://arxiv.org/abs/2405.10871]
Please cite any usage of the data in the challenge with the following citation
```
@misc{bakas2024bratspath,
      title={BraTS-Path Challenge: Assessing Heterogeneous Histopathologic Brain Tumor Sub-regions}, 
      author={Spyridon Bakas and Siddhesh P. Thakur and Shahriar Faghani and Mana Moassefi and Ujjwal Baid and Verena Chung and Sarthak Pati and Shubham Innani and Bhakti Baheti and Jake Albrecht and Alexandros Karargyris and Hasan Kassem and MacLean P. Nasrallah and Jared T. Ahrendsen and Valeria Barresi and Maria A. Gubbiotti and Giselle Y. López and Calixto-Hope G. Lucas and Michael L. Miller and Lee A. D. Cooper and Jason T. Huse and William R. Bell},
      year={2024},
      eprint={2405.10871},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### BraTS Pathology Leading Organizers

* **Spyridon Bakas, PhD ** Indiana University  — [Contact Person]
* **Suhang (Jayden) You, PhD ** Indiana University
* **Siddhesh Thakur,** Indiana University

####! **Organizing Committee** (C: clinical)
* **Jared T. Ahrendsen,** Northwestern University (C)
* **Seemaab Ali,** Indiana University (C)
* **Mehdi Astaraki,** Karolinska Institute
* **Ujjwal Baid, PhD ** Indiana University 
* **Valeria Barresi,** Fondazione IRCCS Istituto Neurologico Carlo Besta
* **William R. Bell,** Indiana University (C)
* **Verena Chung,**  Sage Bionetworks
* **Gian Marco Conte,** Mayo Clinic
* **Lee A D Cooper, PhD** Northwestern University
* **Maria Correia de Verdier,** Uppsala University (C)
* **Keyvan Farahani,** NIH
* **Maria A Gubbiotti,** MD Anderson Cancer Center (C)
* **Hannah Harmsen,** Indiana University (C)
* **Raymond Huang,** Harvard Medical School (C)
* **Jason T Huse, MD** MD Anderson Cancer Center (C)
* **Zhifan Jiang,** Children’s National Hospital, Washington
* **Dominic LaBella,** Duke University Medical Center (C)
* **Giselle Y López,** Duke University (C)
* **Calixto-Hope J Lucas,** Johns Hopkins University (C)
* **Marwan M Majeed** Indiana University (C)
* **Michael L Miller,** Columbia University (C)
* **Jeff Rudie,** University of California, San Diego (C)
* **Benedikt Wiestler,** TU Munich

####! **Data Contributors**

1) Henry Ford Hospital (MI, USA),
2) University of California (CA, USA),
3) MD Anderson Cancer Center (TX, USA),
4) Emory University (GA, USA),
5) Mayo Clinic (MN, USA),
6) Thomas Jefferson University (PA, USA),
7) Duke University School of Medicine (NC, USA),
8) Saint Joseph Hospital and Medical Center (AZ, USA),
9) Case Western Reserve University (OH, USA),
10) University of North Carolina (NC, USA),
11) Fondazione IRCCS Instituto Neuroligico C. Besta, (Italy)
