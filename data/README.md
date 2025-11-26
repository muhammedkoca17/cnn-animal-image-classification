📂 Dataset – Animals with Attributes 2 (AwA2)

This project uses the Animals with Attributes 2 (AwA2) dataset for multi-class animal image classification using a custom CNN model.

⚠️ The full dataset (~14 GB) is NOT stored in this repository due to size and license restrictions.
Instead, this folder includes clear instructions on how to use the dataset:

Locally (VS Code)

Google Colab

Kaggle Notebook (the method I currently use while developing this project)

Optional small sample dataset for fast debugging

🔗 1. Download (Optional)

➡ You do NOT need to download the dataset manually
if you run this project in:

Kaggle Notebook → dataset is mounted automatically

Google Colab → can be downloaded programmatically

Manual download is required only for VS Code usage.

Kaggle dataset link:

https://www.kaggle.com/datasets/rrebirrth/animals-with-attributes-2

After downloading (for VS Code users), extract the folder so the structure is:

data/
└── Animals_with_Attributes2/
    ├── JPEGImages/
    ├── licenses/
    ├── README-attributes.txt
    ├── README-images.txt
    ├── classes.txt
    ├── predicates.txt
    ├── trainclasses.txt
    └── testclasses.txt

📍 2. Paths Used in the Code

Inside src/dataset.py, two key constants define the dataset locations:

RAW_PATH = "data/Animals_with_Attributes2/JPEGImages"
TARGET_PATH = "data/FilteredImages"

🔹 RAW_PATH

The directory containing the original images from Kaggle.

🔹 TARGET_PATH

A folder where the project automatically creates a filtered, cleaned, and balanced dataset for training.

✔ You do not need to manually create class folders inside FilteredImages.
✔ They are generated automatically when running:

prepare_filtered_dataset(...)

🖥️ 3. Using the Dataset in Different Environments

You can run this project in three different setups.
Below are all options + which one I personally use.

✔ A) Running Locally (VS Code) – Manual Download Required

Download dataset from Kaggle

Extract under:

project/
 └─ data/
      └─ Animals_with_Attributes2/


Run training:

python src/train.py


Paths match automatically—no edits needed.

✔ B) Running in Google Colab – No Manual Download Needed

You have three ways to load data in Colab:

Option 1 — Download inside Colab
!kaggle datasets download -d rrebirrth/animals-with-attributes-2
!unzip animals-with-attributes-2.zip -d data/

Option 2 — Use Google Drive
from google.colab import drive
drive.mount('/content/drive')

RAW_PATH = "/content/drive/MyDrive/AwA2/JPEGImages"
TARGET_PATH = "/content/FilteredImages"

Option 3 — Upload a tiny sample dataset

Useful for testing code structure, not full training.

✔ C) Running in Kaggle Notebook – Recommended (I am currently using this option)

This is the fastest and easiest method.

Open a Kaggle Notebook

Add the dataset as input:

Animals with Attributes 2

Kaggle automatically mounts the dataset under:

/kaggle/input/animals-with-attributes-2/Animals_with_Attributes2/JPEGImages


Update the Python paths accordingly:

RAW_PATH = "/kaggle/input/animals-with-attributes-2/Animals_with_Attributes2/JPEGImages"
TARGET_PATH = "/kaggle/working/FilteredImages"


✔ No download
✔ No setup
✔ No folder creation
✔ Training runs immediately

(This is the configuration I used while developing this project.)

🧪 4. Optional: Small Sample Dataset (for quick debugging)

If you want to debug the pipeline quickly without downloading 14 GB:

data/Sample/
 ├── collie/
 ├── dolphin/
 └── elephant/


Then update:

RAW_PATH = "data/Sample"
TARGET_PATH = "data/SampleFiltered"


Useful for:

Testing train.py

Verifying model imports

Debugging augmentation & preprocessing

📝 5. License & Credits

AwA2 dataset belongs to the original authors.
Please refer to:

Original: http://cvml.ist.ac.at/AwA2/

Kaggle mirror: https://www.kaggle.com/datasets/rrebirrth/animals-with-attributes-2

This repository contains only code, and optionally a very small sample subset for demonstration purposes—NOT the full dataset.

✔ Summary for Users
Environment	Needs Download?	Recommended?	Path Setup
Kaggle Notebook	❌ No	⭐ Yes	Automatic + simple RAW_PATH/TARGET_PATH update
Google Colab	❌ No	✔ Good	Auto-download or Drive
VS Code (Local)	✅ Yes	⚠️ Heavy	Manual download required
