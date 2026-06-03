# Data

Synapse wiki: syn74274097/wiki/639594
Modified: 2026-04-28T21:24:24.288Z

The challenge exclusively employs T1 scans from the multi-modal BraTS glioma segmentation challenge. Hence, we provide a training set of 1251 cases and a validation set of 219 cases. Furthermore, there is a non-public test set of several hundred cases. In 2025 we introduced a second test set from the BraTS meningioma segmentation challenge to investigate how well the algorithms are able to generalize.

${image?fileName=Datasets%5Ftransparent%2Epng&align=Center&scale=100&responsive=true&altText=dataset}
The figure illustrates which data is provided in which challenge phase. For inference, only a voided image and the inpainting mask are provided. For training, we additionally provide the normal (non-voided) T1 image as well as masks depicting healthy and unhealthy areas.

Further, this [GitHub repository](https://github.com/BraTS-inpainting/2026_challenge/tree/main/dataset) details the dataset structure and provides means to (re-)generate the masks of the dataset. Participants are encouraged to use it as a starting point to further augment the dataset by generating more masks.

---

###! Data Access

In addition to registering for the challenge, you must also request access to the data:

1. **Submit the form:** complete the [Data Access Google form](https://forms.gle/UiCpXos2zKFPdMnK6). Only one access form is needed across all 5 challenge tasks and their training + validation datasets\*.

2. **Accept your invite:** once your details are verified, the BraTS Service Account will email you an invitation to join the @BraTS2026DataAccessTeam . Accept it to unlock the files below!

<small> \* *Note: test datasets and validation GT labels will not be released to the public.*</small>
