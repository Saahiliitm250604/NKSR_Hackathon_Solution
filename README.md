# 📈 Implied Volatility Imputation for Index Option Chains — NKSR Hackathon

Welcome to my repository for the **NK Securities Research (NKSR) Hackathon**, where I explored various techniques to predict and impute **Implied Volatility (IV)** values in Indian index options. I secured **24th rank** on the final leaderboard.

The task involved restoring missing values (NaNs) in a test dataset using domain-specific and statistical techniques like KNN imputation, spline fitting, and smoothing.

---

## 📂 Dataset

The data can be downloaded from the official competition page on Kaggle:  
🔗 [NKSR IV Prediction Dataset](https://www.kaggle.com/competitions/nk-iv-prediction/data)

- **Train Data**: 178,340 rows × 42+52 columns (including IV columns for 26 strikes, calls & puts)
- **Test Data**: 12,065 rows with missing IV values, shuffled and anonymized

---

## 📊 Key Observations

- IV values are **temporally stable**: nearby timestamps show similar IV profiles.
- IV curves across strikes form a **volatility smile** or skew, which is mathematically predictable.
- For any row, the **ATM strikes** (near underlying price) are more reliable.
- A good imputation method must leverage this **local similarity** across rows.

---

## 🛠️ Step-by-Step Solution

### 1. Data Exploration

Start with `explore.ipynb` to understand the shape and distribution of missing values.

---

### 2. KNN-Based Matching Imputer (C++ 🏆)

- Found to be the most effective approach.
- `matching.cpp`: Implements a fast C++-based KNN (K=1) imputer using RMSE for row similarity.
- Each row is matched with others based on known IV values (minimum 6 common columns required).
- Missing values are imputed from the most similar row, with:
  - `MAX_ITER = 20` similar rows checked
  - `THRESHOLD = 0.001` RMSE tolerance
- ✅ **~96% reduction in runtime** vs Python's `KNNImputer` (489s → 5s)

> 📌 Input format: Preprocessed `.csv` file with **only IV columns**, no headers.

---

### 3. Cubic Spline Interpolation

- Used for filling remaining NaNs after KNN step.
- Located in `cubic_spline.ipynb`.
- Captures the **volatility smile** accurately across strike prices.

---

### 4. Postprocessing (Smoothing)

- Once IV values are imputed, smoothing improves consistency:
  - `pca_smoothing.ipynb`: Uses PCA + inverse transform
  - Savitzky–Golay (`savgol_filter`) applied for local smoothing
- This greatly improved test performance, especially on rows with few known IVs.

---

### 5. Final Enhancements (Optional)

- `final_day/lol.csv`: Final submission based on XGBoost trained over filled data
- `call_put_parity.ipynb`: Explores imputation using **call-put parity**
- `xgb_prev_pred_2.ipynb`: XGBoost model to predict current IV from neighboring columns

---


