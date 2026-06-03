# Challenge Rules

Synapse wiki: syn74274097/wiki/639585
Modified: 2026-04-27T16:59:23.247Z

_Terms and Conditions last updated: 27 April 2026_

Anyone downloading any of the MICCAI BraTS 2026 challenge data agrees that they will comply with the [EO 14117](https://www.presidency.ucsb.edu/documents/executive-order-14117-preventing-access-americans-bulk-sensitive-personal-data-and-united), the [28 CFR Part 202](https://www.ecfr.gov/current/title-28/chapter-I/part-202), and the [Guide Notice NOT-OD-25-083](https://grants.nih.gov/grants/guide/notice-files/NOT-OD-25-083.html), as indicated in the [MICCAI 2026 Challenge Requirements](https://miccai.org/index.php/events/upcoming-conferences/miccai-2026-challenge-requirements/).

To participate in the BraTS 2026 Challenge, you must review and accept the following terms. Please read them carefully and proceed only if you agree to comply with the conditions outlined below.

**Terms of Service and Rules**: <br/> Challenge participants must abide by the [Sage Bionetworks Terms of Service](https://www.synapse.org/TrustCenter:TermsOfService)  and by the Challenge rules below and on the Rules & Resources page.
 
**Challenge Flow, Publication, and Licensing Requirements**: <br/>You must agree to the following:
* You must register to download data, including providing contact information. 
* You must submit methods for validation.
* You must submit a summary paper to the challenge organizers for determining eligibility for participating in the last phase of the challenge. Methods for paper submission will be provided by the challenge organizers, e.g., submitted in a BOX folder, or using the CMT platform. 
* If a short paper is submitted, you may submit methods for ranking.
* Following the evaluation of methods: 
   * If the method performs poorly, once the final results are announced at MICCAI 2026:
      * You are permitted to opt-out from the final meta-analysis manuscript. 
      * Organizers will still report poorly performing methods, but they will not associate the methods with the participants’ names that decide to opt-out.
   * Top-ranked method performance: 
      * You MUST have a published article (even as a preprint, e.g., arXiv, Zenodo) prior to the announcement. The summary article submitted during validation may be used for the published content.
      * Results are announced at MICCAI  
* Organizers report methods in their meta-analysis manuscript and cite the aforementioned published article. 
* Organizers will be irrevocably permitted to make the submitted container publicly accessible on an [Apache v.2.0 license](https://www.apache.org/licenses/LICENSE-2.0), unless another license is otherwise indicated by the participants.

**Submission Reuse**: <br/> By registering for the Challenge, Challenge Participants acknowledge that submissions may be rerun by the BraTS Challenge Organizers, and subjected to analysis on data sets other than the final testing data in order to advance the scientific aims of the competition.

**Recontact**:<br/> By completing the linked [Google Form](https://forms.gle/UiCpXos2zKFPdMnK6), Challenge participants consent to being recontacted by the Challenge Organizers for purposes related to the Challenge, including but not limited to follow-up communications, feedback requests, and potential future research opportunities.  

**Publication Embargo**: <br/> Use of Challenge results in a publication by Challenge participants is permitted if it is restricted to the results of your Challenge method and your Team ranking. Additionally, you agree not to report overall Challenge results or any analysis of the overall results until the organizers and Challenge participants have jointly published (or pre-published) an overview paper on the results from the Challenge and the best performing strategies used in the Challenge. You will be contacted through your Synapse-affiliated email address when this condition has been met. This information will also be posted within this Synapse project.

**Acknowledgement and Citation**: <br/> Challenge participants are permitted to use, publish and present the Challenge results, after the embargo period, provided they acknowledge the BraTS Challenge Organizing members as follows: "Data used in this publication were obtained as part of the Challenge project through Synapse ID (syn74274097)."

**Use of Data in addition to BraTS and pre-trained models**: <br/> Participants are allowed to use additional publicly available annotated datasets and publicly available pre-trained models. These pre-trained models should not be generated using previous BraTS challenge datasets. Participants should not use data from other BraTS sub-challenges to train their models unless allowed on the challenge webpage. For example, participants developing methods for Meningioma segmentation should not use data from pediatric challenges. If participants are using private data (from their own institutions) for extending the provided data, it should be mentioned in the manuscript and an additional MLCube must be submitted during the testing phase of the challenge. This is due to our intentions to provide a fair comparison among the participating methods.

In addition to this there are some challenge specific rules given below:
* For Segmentation - BraTS-Africa Challenge: Participants are allowed to use data from BraTS Adult Glioma Challenge.
* For BraTS Generalizability Across Tumors (BraTS-GoAT) Challenge: participants will ONLY be allowed to use the data provided through the BraTS-GoAT sub-challenge website. The use of any additional data, either private or publicly available (including previous BraTS challenges), will result in the announcement of immediate disqualification from current and future BraTS challenges in the challenge forum.

**Data Usage Agreement / Citations**<br/> BraTS data is subject to a [CC BY-NC license](https://creativecommons.org/licenses/by-nc/4.0/), unless otherwise indicated. You are free to use, redistribute and/or refer to the BraTS datasets in your own research, provided that the data is used for non-commercial use (if CC BY-NC) and you and any other users always cite the flagship manuscript (published or pre-published) resulting from the challenge, as well as the challenge-specific manuscripts listed below. These are subject to occasional updates. 

_List of manuscripts last updated: 20 June 2025_

**Dataset** | **Citations Needed**
-- | --
Any dataset and/or MedPerf client | <ul><li>A. Karargyris, R. Umeton, M.J. Sheller, A. Aristizabal, J. George, A. Wuest, S. Pati, et al. "Federated benchmarking of medical artificial intelligence with MedPerf". Nature Machine Intelligence. 5:799–810 (2023).</li><li> DOI: https://doi.org/10.1038/s42256-023-00652-2</li></ul>
||
BraTS-GLI | <b>Pre-treatment (2023):</b> <ul><li>[1] U.Baid, et al., The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification, arXiv:2107.02314, 2021.</li> <li>[2] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694 </li><li>[3] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117</li></ul>In addition, if there are no restrictions imposed from the journal/conference you submit your paper about citing "Data Citations", please be specific and also cite the following: <ul><li>[4] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q </li><li>[5] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.GJQ7R0EF</li></ul> <hr> <b>Post-treatment (2024):</b><ul><li>arXiv: https://arxiv.org/abs/2405.18368</li><li>DOI: https://doi.org/10.48550/arXiv.2405.18368</li></ul>
||
BraTS-Local-Inpainting | <ul><li>arXiv: https://arxiv.org/abs/2305.08992 </li><li> DOI: https://doi.org/10.48550/arXiv.2305.08992</li></ul> Please kindly also cite the **BraTS-GLI 2023 manuscripts**, as the inpainting challenge operates on the BraTS-GLI 2023 dataset.
||
BraTS-MEN | <ul><li>arXiv: https://arxiv.org/abs/2305.07642 </li><li> DOI: https://doi.org/10.48550/arXiv.2305.07642</li></ul>
||
BraTS-MEN-RT | <ul><li>arXiv: https://arxiv.org/abs/2405.18383</li></ul>
||
BraTS-MET | <b>2023:</b> <ul><li>arXiv: https://arxiv.org/abs/2306.00838 </li><li>DOI: https://doi.org/10.48550/arXiv.2306.00838</li></ul><hr><b>2024:</b><ul><li>Coming soon</li></ul>
||
BraSyn | <ul><li> arXiv: https://arxiv.org/abs/2305.09011 </li><li> DOI: https://doi.org/10.48550/arXiv.2305.09011</li></ul> Please kindly also cite the **BraTS-GLI 2023 manuscripts**, as the BraSyn challenge operates on the BraTS-GLI 2023 dataset.
||
BraTS-Path | <ul><li>arXiv: https://arxiv.org/abs/2405.10871</li></ul>
||
BraTS-PED | <b>2023</b>: <ul><li>arXiv: https://arxiv.org/abs/2305.17033</li><li>DOI: https://doi.org/10.48550/arXiv.2305.17033</li></ul><hr><b>2024</b>: <ul><li>arXiv: https://arxiv.org/abs/2404.15009</li><li>DOI: https://doi.org/10.48550/arXiv.2404.15009</li></ul>
||
BraTS-SSA | <ul><li>[1] Adewole M, Rudie JD, Gbadamosi A, et al. The Brain Tumor Segmentation (BraTS) Challenge 2023: Glioma Segmentation in Sub-Saharan Africa Patient Population (BraTS-Africa). arXiv:2305.19369 [eess.IV] (2023).</li></ul><ul><li>arXiv: https://arxiv.org/abs/2305.19369 </li><li>DOI: https://doi.org/10.48550/arXiv.2305.19369</li></ul>

- **Note**: Challenge participants agree to cite the initial challenge pre publication manuscript (or the final publication manuscript). You will be contacted through your Synapse affiliated email when the manuscript has been released for citation.

- **Note**: Use of the BraTS datasets for creating and submitting benchmark results for publication on [MLPerf.org](https://mlcommons.org/en/) is considered non-commercial use. It is further acceptable to republish results published on MLPerf.org, as well as to create unverified benchmark results consistent with the MLPerf.org rules in other locations. Please note that you should always adhere to the BraTS data usage guidelines and cite appropriately the aforementioned publications, as well as to the terms of use required by MLPerf.org.

