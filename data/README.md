# 📂 Dataset – Animals with Attributes 2 (AwA2)

This project uses the **Animals with Attributes 2 (AwA2)** dataset for multi-class animal image classification with a CNN model.

> ⚠️ The full dataset is **NOT** included in this repository due to its size (~14 GB) and licensing limitations.  
> Please download it manually from Kaggle and place it in the correct folder as described below.

## 🔗 Download Instructions

Download from Kaggle:  
https://www.kaggle.com/datasets/rrebirrth/animals-with-attributes-2

After extracting the dataset, make sure the folder structure matches:

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

## 📍 Dataset Paths Used in This Project

Inside `src/`, dataset paths are referenced as:

RAW_PATH = "data/Animals_with_Attributes2/JPEGImages"  
TARGET_PATH = "data/FilteredImages"

RAW_PATH → Points to the original AwA2 images inside JPEGImages/.  
TARGET_PATH → Auto-created directory that stores the filtered, balanced subset.  
No manual folder creation is needed.

Filtering operation is handled by:

prepare_filtered_dataset(RAW_PATH, TARGET_PATH, ...)

## 🧪 Optional: Tiny Sample for Testing

If you want a lightweight version without downloading 14 GB:

Create:

data/  
└── Sample/  
  ├── collie/  
  ├── dolphin/  
  └── elephant/  

Put 5–10 images per class.

Update dataset paths in `src/dataset.py`:

RAW_PATH = "data/Sample"  
TARGET_PATH = "data/SampleFiltered"

Perfect for debugging or fast experimentation.

## 🧬 Purpose of This Dataset in the Project

The AwA2 dataset is used for:

• Supervised CNN training  
• Multi-class classification  
• Attribute-based experimentation  

The filtering ensures:

• Selected classes only  
• Limited number of images per class  
• Balanced dataset  
• Reproducibility across runs  

## 📜 License & Credits

AwA2 dataset belongs to its original authors.

Original project:  
http://cvml.ist.ac.at/AwA2/

Kaggle mirror:  
https://www.kaggle.com/datasets/rrebirrth/animals-with-attributes-2

This repository contains **only code** and optionally a very small demo subset—not the full dataset.

## ✅ Quick Setup Summary

1. Download AwA2 from Kaggle  
2. Extract into: data/Animals_with_Attributes2/  
3. Ensure JPEGImages/ is present  
4. Run the project — filtering happens automatically  


