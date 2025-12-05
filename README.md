# 🧠 Drug Sentiment Classification  
### Comparing Traditional Classification Models vs. Bio_ClinicalBERT on Patient Drug Reviews

**Author:** Chadric Garrick  
**Course:** MSBD-566
**Date:** October 24, 2025  

---

## 📘 Overview
This project explores how sentiment can be classified from **patient-authored drug reviews**, aiming to identify whether each review expresses a **positive, neutral, or negative** experience based on descriptions of effectiveness and side effects.  
The comparison focuses on two modeling approaches:

1. **Classical machine learning classifiers** using TF-IDF text representations  
2. **Domain-specific transformer model:** `emilyalsentzer/Bio_ClinicalBERT`

The goal was to evaluate which method best captures medical context, subtle emotional tone, and contradictory phrasing common in patient narratives.

---

## 🧩 Dataset
- **Source:** [UCI Machine Learning Repository — Drug Review Dataset (Drugs.com / DrugLib.com)](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+(Drugs.com))
- **Records:** ≈ 4,142 cleaned patient reviews
- **Fields Used:**
  - `urlDrugName` – drug name  
  - `rating` – satisfaction score (1–10)  
  - `benefitsReview`, `sideEffectsReview`, `commentsReview` – free-text narratives  
- **Preprocessing:**
  - Merged all three text fields into one unified column: `text`
  - Converted `rating` into categorical sentiment labels:  
    - 1–3 → Negative  
    - 4–6 → Neutral  
    - 7–10 → Positive  

---

## ⚙️ Methods & Models

### Pipeline Architecture
**ClinicalBERT Embeddings → PCA → Feedforward ANN → Sentiment Classification**

The main pipeline uses:
1. **ClinicalBERT Feature Extraction**: Extract [CLS] token embeddings (768-dim) from pre-trained `emilyalsentzer/Bio_ClinicalBERT`
2. **PCA Dimensionality Reduction**: Reduce embeddings to 50 dimensions while preserving variance
3. **Feedforward ANN**: 2-layer neural network with ReLU and Dropout for classification

| Component | Description |
|--------|----------------|
| **ClinicalBERT Embeddings** | 768-dimensional contextual embeddings from biomedical BERT |
| **PCA** | Dimensionality reduction (768 → 50 components) |
| **Feedforward ANN** | 2 hidden layers (64 units each), ReLU activation, Dropout (0.2) |

### Training Details
- **Split:** 80/20 train–validation (stratified)
- **Metrics:** Accuracy, Macro Recall, Macro F1
- **Optimizer:** Adam (ANN)
- **Learning Rate:** 0.001  
- **Epochs:** 10  
- **Batch Size:** 32 (ANN), 16 (BERT embedding extraction)
- **PCA Components:** 50 (adjustable: 32, 50, 100, etc.)  

---

## 📊 Results
| Model | Accuracy | Macro Recall | Macro F1 |
|--------|-----------|---------------|-----------|
| Bio_ClinicalBERT | 0.768 | 0.590 | 0.574 |
| TF-IDF + Linear SVM | 0.734 | 0.512 | 0.520 |
| TF-IDF + Logistic Regression | 0.705 | 0.408 | 0.395 |

**Confusion matrices** showed that:
- ClinicalBERT captured nuanced sentiment (e.g., *"effective but caused fatigue"*).  
- SVM handled polarized phrases but missed mixed tones.  
- Logistic Regression favored the positive class and misread mild or neutral phrasing.  
- All models struggled most with the **neutral** category due to overlapping language.

---

## 💬 Interpretation
ClinicalBERT outperformed both traditional models by understanding full sentence context rather than treating words as isolated features.  
Its biomedical pretraining allowed it to connect phrases like *"effective but caused nausea"* to both positive and negative sentiment cues simultaneously.

The TF-IDF classifiers remain valuable for their interpretability and low compute cost, but they lack contextual depth.  
Future improvements may include class-weighted loss for underrepresented categories, longer sequence lengths, or aspect-based sentiment analysis (ABSA) to separate opinions about **effectiveness**, **side effects**, and **overall satisfaction**.

---

## 🧠 Key Takeaways
- Transformer models better capture **contextual sentiment** in medical language.  
- Classical classifiers perform well for **explicit sentiment** but miss nuance.  
- Neutral class imbalance remains a consistent challenge.  
- This work demonstrates how **domain-adapted NLP models** can support pharmacovigilance, clinical insights, and real-world evidence extraction.

---

## 🧾 References
- Alsentzer, E. et al. (2019). *Publicly available clinical BERT embeddings.* Proceedings of the 2nd Clinical NLP Workshop.  
- Dua, D. & Graff, C. (2019). *UCI Machine Learning Repository: Drug Review Dataset (Drugs.com & DrugLib.com).*  
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.  
- Sparck Jones, K. (1972). *A statistical interpretation of term specificity and its application in retrieval.* Journal of Documentation, 28(1), 11–21.  
- Wolf, T. et al. (2020). *Transformers: State-of-the-art natural language processing.* EMNLP 2020.

---

## 📂 Repository Structure
```
drug-sentiment-classification/
├── data/
│   ├── drugLibTrain_raw.tsv      # Training data
│   ├── drugLibTest_raw.tsv       # Test data
│   └── cleaned_drug_data.csv     # Processed dataset (generated)
├── Drug_Sentiment_Classification_Git_Final.ipynb  # Main notebook (generic)
├── Drug_Sentiment_Classification_Colab.ipynb      # Google Colab version
├── Drug_Sentiment_Classification_Metal.ipynb      # Metal GPU (Apple Silicon) version
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
└── .gitignore                     # Git ignore rules
```

## 📱 Notebook Versions

### 1. **Drug_Sentiment_Classification_Git_Final.ipynb** (Generic)
- Standard version for general use
- Works on CPU, CUDA, or MPS
- Uses relative paths: `data/`

### 2. **Drug_Sentiment_Classification_Colab.ipynb** (Google Colab)
- Optimized for Google Colab environment
- Includes data upload interface
- Uses `/content/data/` paths
- Auto-detects Colab GPU (T4, V100, etc.)
- **Usage:**
  1. Upload notebook to Google Colab
  2. Run first cell to upload data files
  3. Enable GPU: Runtime > Change runtime type > GPU
  4. Run all cells

### 3. **Drug_Sentiment_Classification_Metal.ipynb** (Apple Silicon)
- Optimized for Apple Silicon Macs (M1/M2/M3)
- Uses Metal Performance Shaders (MPS) for GPU acceleration
- Includes MPS memory management
- Uses local paths: `data/`
- **Requirements:**
  ```bash
  pip install torch torchvision torchaudio
  pip install transformers datasets scikit-learn
  ```
- **Usage:**
  1. Ensure PyTorch with MPS support is installed
  2. Place data files in `./data/` directory
  3. Run notebook locally with Jupyter

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd drug-sentiment-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the appropriate notebook:
   ```bash
   # For local execution (CPU/Metal GPU)
   jupyter notebook Drug_Sentiment_Classification_Metal.ipynb
   
   # Or for generic version
   jupyter notebook Drug_Sentiment_Classification_Git_Final.ipynb
   ```

4. The notebook will:
   - Load data from `data/drugLibTrain_raw.tsv` and `data/drugLibTest_raw.tsv`
   - Clean and preprocess the data
   - Extract ClinicalBERT embeddings (768-dim)
   - Apply PCA dimensionality reduction (50 components)
   - Train feedforward ANN classifier
   - Generate classification report and confusion matrix
   - Save results to `model_performance_comparison.csv`

---

## 🩺 Author
**Chadric Garrick**  
Graduate Student — Biomedical Data Science  
*Focus: NLP in healthcare, sentiment analysis, and applied machine learning.*  

---

*This repository was developed as part of the MSBD-566 course to demonstrate applied NLP techniques for real-world healthcare data.*

