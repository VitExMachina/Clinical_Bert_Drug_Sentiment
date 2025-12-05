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
| Model | Representation | Description |
|--------|----------------|-------------|
| **TF-IDF + Logistic Regression** | 1–2 grams | Linear baseline, interpretable and efficient |
| **TF-IDF + Linear SVM** | 1–2 grams | Strong classical text classifier, robust to sparse features |
| **Bio_ClinicalBERT** | Transformer embeddings | Fine-tuned model pretrained on biomedical & clinical text |

### Training Details
- **Split:** 80/20 train–validation (stratified)
- **Metrics:** Accuracy, Macro Recall, Macro F1
- **Optimizer:** AdamW (BERT)
- **Learning Rate:** 2e-5  
- **Epochs:** 3  
- **Batch Size:** 16  

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
├── Drug_Sentiment_Classification_Git_Final.ipynb  # Main notebook
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
└── .gitignore                     # Git ignore rules
```

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

3. Run the notebook:
   ```bash
   jupyter notebook Drug_Sentiment_Classification_Git_Final.ipynb
   ```

4. The notebook will:
   - Load data from `data/drugLibTrain_raw.tsv` and `data/drugLibTest_raw.tsv`
   - Clean and preprocess the data
   - Train baseline models (TF-IDF + LinearSVC, TF-IDF + LogReg)
   - Fine-tune ClinicalBERT
   - Generate comparison metrics and confusion matrices
   - Save model outputs to `bert-drug-sentiment-best/` and `model_performance_comparison.csv`

---

## 🩺 Author
**Chadric Garrick**  
Graduate Student — Biomedical Data Science  
*Focus: NLP in healthcare, sentiment analysis, and applied machine learning.*  

---

*This repository was developed as part of the MSBD-566 course to demonstrate applied NLP techniques for real-world healthcare data.*

