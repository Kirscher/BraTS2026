# Data

Synapse wiki: syn74274097/wiki/639588
Modified: 2026-06-02T06:26:32.556Z

In the BraTS-Path challenge, we use publicly available H&E-stained, FFPE-digitzed tissue sections from The Cancer Imaging Archive's TCGA-GBM and TCGA-LGG collections. The dataset includes a retrospective, multi-institutional cohort of patients with de novo diffuse gliomas, providing a clinically rich foundation for the challenge. Tissue sections have been reclassified by our team according to the latest WHO criteria, with a focus on glioblastoma, and include annotations from expert neuropathologists who have identified and segmented distinct histological regions into patches. These patches are then classified based on specific histological characteristics for a classification task.

Training data: 1M+ labeled images (from 200+ digitized tissue sections) and 80 large-size images (unlabeled for unsupervised learning)
Validation data: 100,000+ images (from 20+ digitized tissue sections)
Testing data: 600,000+ images (from 50+ digitized tissue sections)

**Important:** Participants are allowed to use additional data from publicly available datasets and their institutions for further complementing the data, but if they do so, they MUST also discuss the potential difference in their results after using only the BraTS 2026 data, since our intention is to solve the particular classification problem, but importantly, to provide a fair comparison among the participating methods.

Participants will be given training and validation datasets stored as tar shards, which can be loaded with WebDataset. For the labelled training dataset, the tar shards contain .jpg image patches at 90% quality and .cls files containing label text (0~9). For the labelled training set, we ensured that each label appeared in every tar shard file and did our best to distribute patches evenly across the shards. They are randomly distributed with a fixed seed. We also provide a CSV file that maps each patch's name to its patient (patient 0-125) and slide (slide 0-254) ID for the labelled training dataset.

We also provide large images in .tiff format for unsupervised learning, along with the corresponding foreground tissue mask. These .tiff images are extracted from the level 0 of the pyramids from the original whole slide images and compressed to 90% JPEG quality. Meanwhile, we also provide scripts and instructions for patching the tissue and packing them into tar shards.

To load the data with webdataset, an example loading script snippet for tar shards is provided as follows. For more information on webdataset, please refer to the PyPi release site https://pypi.org/project/webdataset/


<!--
```
import webdataset as wds
from torchvision import transforms
from torch.utils.data import DataLoader

transform_toTensor = transforms.Compose([
    transforms.ToTensor(),
])
dataset = (
    wds.WebDataset("BraTS-Path2026-Train-TARS/shard-{000000..000024}.tar")
    .shuffle(1000)
    .decode("pil")
    .to_tuple("jpg", "cls")
)

# convert label to int
def preprocess(sample, image_transform = transform_toTensor):
    img, cls = sample
    if isinstance(cls, bytes):
        label = int(cls.decode("utf-8").strip())
    else:
        label = int(cls)
    return image_transform(img), label

dataset = dataset.map(preprocess)


loader = DataLoader(dataset, batch_size=32, num_workers=4)

for imgs, labels in loader:
    pass
```
-->
```
import torch
from torch.utils.data import DataLoader
import webdataset as wds
# This package is required to be installed use
# conda install conda-forge::webdataset
# or
# pip install webdataset
from torchvision import transforms

# --- image transform ---
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def to_int_label(x):
    # WebDataset often returns label bytes or strings
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    return int(x)


def to_filename(x):
    # __key__ is the sample key inside the shard
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    return x

def build_selected_shards(base_path, indices):
    """
    base_path: e.g. "data/train"
    indices: list like [0, 34, 79, 23]
    """
    shards = [
        f"{base_path}/shard-{i:06d}.tar"
        for i in indices
    ]
    return shards

def build_dataloader(shards, batch_size=32, num_workers=8, shuffle=True): # shuffle set to true in training and False for evaluation
    """
    shards: a string like 'data/shards/train-{000000..000099}.tar'
            or a list of tar shard paths.

    Each sample is expected to contain:
      - image: .jpg or .png
      - label: .cls or .txt
      - filename/key: use __key__ as the file name
    """

    dataset = (
        wds.WebDataset(shards, shardshuffle=shuffle) # this shuffles among the selected shard; Set shardshuffle=False for evaluation
        .shuffle(5000) # This shuffles the files inside the shard file. larger number meaning larger RAM consumption; delete this for evaluation
        .decode("pil")
        .to_tuple("__key__", "jpg", "cls")
        .map_tuple(
            to_filename,
            image_transform,
            to_int_label,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,  # WebDataset handles shard/sample shuffling Please ensure it is false.
        drop_last=True,
    )
    return loader


# -------------------------
# Example training loop
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    for step, (file_names, images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step={step}, loss={loss.item():.4f}")
            print("example file names:", file_names[:3])


# -------------------------
# Usage
# -------------------------
if __name__ == "__main__":

    # the following uses specific shards to train
    selected_indices = [0, 34, 9, 23] # this is the intended indeces in the path folder
    shards = build_selected_shards(
        base_path="BraTS-Path2026-Train-TARS/", # folder that contain all the shards files for training
        indices=selected_indices
    )

    # the following uses order from 00 to 39 shard to train
    # shards = "BraTS-Path2026-Train-TARS/shard-{000000..000039}.tar"
    loader = build_dataloader(
        shards=shards,
        batch_size=64,
        num_workers=8,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # example model
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 224 * 224, 1000),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train_one_epoch(model, loader, optimizer, criterion, device)
```

---

###! Data Access

In addition to registering for the challenge, you must also request access to the data:

1. **Submit the form:** complete the [Data Access Google form](https://forms.gle/UiCpXos2zKFPdMnK6). Only one access form is needed across all 5 challenge tasks and their training + validation datasets\*.

2. **Accept your invite:** once your details are verified, the BraTS Service Account will email you an invitation to join the @BraTS2026DataAccessTeam . Accept it to unlock the files below!

<small> \* *Note: test datasets and validation GT labels will not be released to the public.*</small>


