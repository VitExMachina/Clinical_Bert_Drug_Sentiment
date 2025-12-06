# Cost-Benefit Analysis: PCA vs UMAP for Dimensionality Reduction
## ClinicalBERT Embeddings → Dimensionality Reduction → ANN Classifier

**Context:** Reducing 768-dimensional ClinicalBERT embeddings to ~50 dimensions for feedforward ANN classification

---

## 📊 Quick Comparison Table

| Factor | PCA | UMAP | Winner |
|--------|-----|------|--------|
| **Computational Speed** | ⚡ Very Fast (seconds) | 🐌 Slow (minutes-hours) | **PCA** |
| **Memory Usage** | ✅ Low | ⚠️ High | **PCA** |
| **Information Preservation** | ⚠️ Linear variance only | ✅ Non-linear structure | **UMAP** |
| **Hyperparameter Sensitivity** | ✅ Minimal (just n_components) | ⚠️ High (n_neighbors, min_dist, n_components) | **PCA** |
| **Reproducibility** | ✅ Deterministic | ⚠️ Stochastic (random_state helps) | **PCA** |
| **Interpretability** | ✅ Principal components = variance axes | ❌ No clear interpretation | **PCA** |
| **Scalability** | ✅ O(n²) but optimized | ⚠️ O(n log n) but memory-intensive | **PCA** |
| **Implementation Complexity** | ✅ Simple (sklearn) | ⚠️ Requires umap-learn package | **PCA** |
| **Downstream Performance** | ⚠️ May lose non-linear patterns | ✅ Better preserves local/global structure | **UMAP** |

---

## 💰 Cost Analysis

### **1. Computational Cost**

#### PCA
- **Time Complexity:** O(min(n × d², d³)) where n=samples, d=features
- **Your Dataset:** ~4,142 samples × 768 features
- **Actual Runtime:** ~1-5 seconds on CPU
- **Memory:** ~50-100 MB (stores covariance matrix)
- **GPU Acceleration:** Not typically needed/beneficial

#### UMAP
- **Time Complexity:** O(n × log(n) × n_neighbors × d)
- **Your Dataset:** With default n_neighbors=15
- **Actual Runtime:** ~5-30 minutes on CPU (depends on n_neighbors)
- **Memory:** ~500 MB - 2 GB (stores k-nearest neighbor graph)
- **GPU Acceleration:** Limited support (requires cuML/rapids)

**Cost Winner: PCA** (100-1000x faster)

---

### **2. Implementation & Maintenance Cost**

#### PCA
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=50, random_state=42)
X_reduced = pca.fit_transform(X_scaled)
```
- ✅ Single hyperparameter (n_components)
- ✅ Built into sklearn (no extra dependencies)
- ✅ Deterministic results
- ✅ Easy to save/load (just save components)

#### UMAP
```python
import umap
reducer = umap.UMAP(
    n_components=50,
    n_neighbors=15,        # Critical hyperparameter
    min_dist=0.1,         # Affects clustering tightness
    metric='euclidean',    # Distance metric choice
    random_state=42
)
X_reduced = reducer.fit_transform(X_scaled)
```
- ⚠️ 4+ hyperparameters to tune
- ⚠️ Requires `umap-learn` package (extra dependency)
- ⚠️ Stochastic (results vary slightly between runs)
- ⚠️ More complex to save/load (need to save the reducer object)

**Cost Winner: PCA** (simpler, fewer dependencies)

---

## 🎯 Benefit Analysis

### **1. Information Preservation**

#### PCA
- **What it preserves:** Linear variance (global structure)
- **What it loses:** Non-linear relationships, local neighborhoods
- **For BERT embeddings:** BERT embeddings often have non-linear structure
- **Impact:** May compress semantically similar but linearly distant embeddings

#### UMAP
- **What it preserves:** 
  - Local neighborhood structure (similar embeddings stay close)
  - Global structure (manifold topology)
  - Non-linear relationships
- **For BERT embeddings:** Better at preserving semantic clusters
- **Impact:** Sentiment classes may be more separable in UMAP space

**Benefit Winner: UMAP** (better for non-linear embeddings)

---

### **2. Downstream Classification Performance**

#### Expected Impact on ANN Performance

**PCA:**
- ✅ Fast to compute → more time for hyperparameter tuning
- ⚠️ May lose discriminative non-linear features
- ✅ Works well if classes are linearly separable in embedding space
- **Typical Result:** 0.5-2% accuracy difference vs. no reduction

**UMAP:**
- ⚠️ Slow to compute → less time for experimentation
- ✅ Better preserves class boundaries
- ✅ May improve performance on imbalanced classes (neutral vs. positive/negative)
- **Typical Result:** 1-5% accuracy improvement over PCA (but not guaranteed)

**Benefit Winner: UMAP** (potentially better, but context-dependent)

---

### **3. Interpretability & Debugging**

#### PCA
- ✅ Principal components = directions of maximum variance
- ✅ Can inspect `explained_variance_ratio_` to see information retention
- ✅ Components are orthogonal (independent)
- ✅ Easy to visualize (first 2-3 components often meaningful)

#### UMAP
- ❌ No clear interpretation of dimensions
- ❌ Dimensions are not independent
- ⚠️ Visualization is beautiful but not interpretable
- ❌ Hard to debug why certain samples cluster together

**Benefit Winner: PCA** (much more interpretable)

---

## 📈 Real-World Performance Estimates

### For Your Specific Use Case (4,142 samples, 768 → 50 dims)

| Metric | PCA | UMAP | Notes |
|--------|-----|------|-------|
| **Reduction Time** | 2-5 sec | 5-30 min | UMAP 100-1000x slower |
| **Memory Peak** | ~100 MB | ~1-2 GB | UMAP stores neighbor graph |
| **Accuracy Impact** | Baseline | +0.5-2% | UMAP may help with neutral class |
| **F1 Score Impact** | Baseline | +1-3% | UMAP better for imbalanced classes |
| **Reproducibility** | 100% | ~99% | UMAP has slight randomness |

---

## 🎯 Recommendation Matrix

### **Use PCA if:**
- ✅ Speed is critical (rapid prototyping, hyperparameter search)
- ✅ You need interpretability (research, debugging)
- ✅ Computational resources are limited (Colab free tier, local CPU)
- ✅ You want deterministic results
- ✅ Your classes are reasonably linearly separable
- ✅ You're doing many experiments and need fast iteration

### **Use UMAP if:**
- ✅ Classification accuracy is the top priority
- ✅ You have time/compute resources (GPU, long-running jobs)
- ✅ Classes are highly non-linearly separable
- ✅ You're doing final model optimization (not exploration)
- ✅ You suspect PCA is losing critical information
- ✅ You can afford 5-30 minutes per reduction

---

## 💡 Hybrid Approach (Best of Both Worlds)

### **Strategy 1: PCA for Development, UMAP for Final Model**
```python
# Development phase: Fast iteration with PCA
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

# Final model: Switch to UMAP for best performance
umap_reducer = umap.UMAP(n_components=50, n_neighbors=15, min_dist=0.1)
X_umap = umap_reducer.fit_transform(X_scaled)
```

### **Strategy 2: Compare Both**
```python
# Train ANN on PCA features
model_pca = train_ann(X_pca, y_train)

# Train ANN on UMAP features  
model_umap = train_ann(X_umap, y_train)

# Compare validation performance
# Choose best performing approach
```

### **Strategy 3: UMAP with Smaller n_neighbors**
```python
# Faster UMAP (less accurate but still better than PCA)
umap_fast = umap.UMAP(
    n_components=50,
    n_neighbors=5,      # Smaller = faster (default is 15)
    min_dist=0.5,       # Larger = faster
    n_jobs=-1           # Parallel processing
)
# Runtime: ~2-5 minutes instead of 10-30
```

---

## 🔬 Experimental Validation Plan

### **Test Both Methods:**

1. **Baseline:** No dimensionality reduction (768 dims) - may be too slow for ANN
2. **PCA:** 50 components (current implementation)
3. **UMAP:** 50 components, n_neighbors=15, min_dist=0.1
4. **UMAP Fast:** 50 components, n_neighbors=5, min_dist=0.5

**Metrics to Compare:**
- Validation Accuracy
- Macro F1 Score (especially for neutral class)
- Training Time
- Inference Time
- Memory Usage

**Decision Rule:**
- If UMAP improves F1 by >2% → Use UMAP
- If improvement <1% → Stick with PCA (faster)
- If improvement 1-2% → Consider hybrid approach

---

## 📊 Expected Outcomes for Your Dataset

### **Current Setup (PCA):**
- **Accuracy:** ~75-78% (estimated)
- **Macro F1:** ~0.55-0.60
- **Neutral Class F1:** ~0.15-0.25 (likely struggling)

### **With UMAP:**
- **Accuracy:** ~76-80% (+1-2%)
- **Macro F1:** ~0.58-0.65 (+3-5%)
- **Neutral Class F1:** ~0.20-0.35 (potentially better)

**Key Insight:** UMAP may particularly help with the **neutral class** which is underrepresented and likely has non-linear boundaries with positive/negative classes.

---

## 💰 Final Cost-Benefit Summary

### **PCA: Low Cost, Moderate Benefit**
- **Cost:** ⭐ Very Low (seconds, simple)
- **Benefit:** ⭐⭐⭐ Good (works well for most cases)
- **ROI:** ⭐⭐⭐⭐⭐ Excellent (fast iteration, good results)

### **UMAP: High Cost, Potentially Higher Benefit**
- **Cost:** ⭐⭐⭐ High (minutes-hours, complex)
- **Benefit:** ⭐⭐⭐⭐ Very Good (better for non-linear data)
- **ROI:** ⭐⭐⭐ Moderate (only worth it if accuracy gain is significant)

---

## 🎯 My Recommendation

**For your current project:**

1. **Start with PCA** (you're already doing this) ✅
2. **If neutral class F1 < 0.25**, try UMAP as an experiment
3. **If UMAP improves macro F1 by >2%**, use it for final model
4. **Otherwise, stick with PCA** for speed and simplicity

**Rationale:**
- Your dataset is relatively small (4,142 samples)
- PCA is likely sufficient for initial separation
- UMAP's benefit may not justify 10-30x slower runtime
- However, if neutral class is struggling, UMAP could be worth it

**Best Practice:** Implement both, compare on validation set, document the trade-off.

---

## 📚 References

- **PCA:** Jolliffe & Cadima (2016). "Principal component analysis: a review and recent developments"
- **UMAP:** McInnes et al. (2018). "UMAP: Uniform Manifold Approximation and Projection"
- **BERT Embeddings:** Devlin et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"

---

*Generated for: Drug Sentiment Classification Project*  
*Pipeline: ClinicalBERT (768-dim) → Dimensionality Reduction → ANN Classifier*

