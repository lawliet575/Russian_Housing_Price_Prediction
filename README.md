# 🏠 Russian Housing Price Prediction

This project showcases housing price prediction using a **synthesized version** of the original Kaggle dataset. Expanded from 30K to 260K rows via **SMOTE** and **Stresser**, it provides a rich foundation for exploring various regression models. The included Python notebook walks through data preprocessing, feature engineering, and model evaluation to identify the best-performing approach.


# Russian House Price Prediction

A detailed analysis of the models and feature preprocessing steps used for predicting house prices in Russia.

---

## Table of Contents
1. [Model Evaluations](#model-evaluations)
    - [Linear Regression](#linear-regression)
    - [Polynomial Regression](#polynomial-regression)
    - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
    - [Regression Trees](#regression-trees)
    - [Random Forest](#random-forest)
    - [AdaBoost](#adaboost)
    - [Gradient Boosting](#gradient-boosting)
    - [XGBoost](#xgboost)
    - [Neural Networks](#neural-networks)
    - [Stacking](#stacking)
2. [Feature Preprocessing](#feature-preprocessing)
    - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
    - [Feature Importance & Selection](#feature-importance--selection)
    - [Scaling & Normalization](#scaling--normalization)
    - [Handling Missing Values](#handling-missing-values)
    - [Label Encoding](#label-encoding)
3. [Challenges](#challenges)      
4. [Winner Of the Competition](#the-winner-of-the-competition-extra-trees-regressor)   

---

## Model Evaluations

### Linear Regression
- **Performance**: RMSE = 13,283,371.97  
- Experimented with solvers (`saga` and `lsqr`), but improvements were minimal.
- **Conclusion**: Performed poorly compared to other models.

---

### Polynomial Regression
- **Best Results**: Degree 2 with 75 principal components: RMSE = 13,695,385.56  
- Increasing degree (e.g., 3) worsened performance significantly: RMSE = 17,716,343.26.  
- **Conclusion**: Only marginally effective; overfitted with higher degrees.

---

### K-Nearest Neighbors (KNN)
- **Best Results**: k = 50: RMSE = 12,786,347.29.  
- Increasing neighbors to k = 150 improved accuracy but led to overfitting beyond this point.  
- **Conclusion**: Better than Polynomial and Linear Regression, but sensitive to `k`.

---

### Regression Trees
- Depth = 3 gave reasonable results.  
- Depth = 7 with min_samples_leaf = 2 overfitted, performed poorly on the test set.  
- Adjusting max_leaf_features further reduced accuracy.  
- **Conclusion**: Performed moderately but did not generalize well.

---

### Random Forest
- **Best Results**: 150 estimators with depth 10 gave good results.  
- Increasing estimators beyond 150 or depth beyond 10 decreased accuracy.  
- **Conclusion**: Worked well with default settings, though feature processing could improve results.

---

### AdaBoost
- **Best Results**: RMSE = 12,849,063.35 with learning rate = 0.2 and linear loss.  
- Increasing estimators slightly dropped accuracy; exponential loss took longer with no significant benefit.  
- **Conclusion**: Performed better than Polynomial and Linear Regression but was not as effective as Random Forest.

---

### Gradient Boosting
- **Best Results**: 150 estimators, learning rate = 0.05: RMSE = 12,695,046.73.  
- Increasing estimators or learning rate degraded performance.  
- **Conclusion**: Competitive with Random Forest and highly effective.

---

### XGBoost
- **Best Results**: Depth = 4, estimators = 400, learning rate = 0.015.  
- Increasing estimators to 1000 reduced accuracy.  
- **Conclusion**: Worked exceptionally well with fine-tuned parameters.

---

### Neural Networks
- **Performance**: Did not perform well; hidden layer sizes > 100 or iterations > 50 reduced performance.  
- Activation functions (`ReLU`, `tanh`) had minimal impact.  
- **Conclusion**: Not effective for this dataset.

---

### Stacking
- Ensemble of:
  - XGBoost (125 estimators, learning rate = 0.01)
  - Extra Trees (200 estimators)
  - Linear Regressor as the final meta-model  
  - **Performance**: RMSE = 12,440,977.00  
- Stacking with Decision Trees (depth = 25) and Linear Regression resulted in RMSE = 12,593,651.63 but was computationally expensive.  
- **Conclusion**: Promising results but time-consuming.

---

## Feature Preprocessing

### Principal Component Analysis (PCA)
- PCA with 50 components showed only a marginal improvement in accuracy.  
- Dropping 10–20 features via PCA did not significantly help.  
- **Conclusion**: Not very effective.

---

### Feature Importance & Selection
- P-value-based feature selection showed no substantial improvement.  
- Dropping irrelevant columns (e.g., `sub_area`) improved performance slightly.  
- Forward and backward selection methods were computationally expensive with no significant gains.  
- **Conclusion**: Limited success with feature selection.

---

### Scaling & Normalization
- Scaling with StandardScaler or MinMaxScaler did not improve performance significantly.  
- **Conclusion**: No substantial impact.

---

### Handling Missing Values
- No missing values in the dataset; no imputations were necessary.

---

### Label Encoding
- Categorical columns encoded using `LabelEncoder` with mapping:
  - `Yes` → 1  
  - `No` → 0  
- Significant improvement in performance after encoding these columns:  
  `['culture_objects_top_25', 'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion', 'radiation_raion', 'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion', 'detention_facility_raion', 'big_road1_1line', 'railroad_1line', 'water_1line']`.

---

## Challenges

1. **Finding the Right Model**  
   - The first challenge was identifying the best-performing model. I tackled this by **randomly testing multiple models** and selecting the one with the most promising results (Extra Trees Regressor).  

2. **Tuning Parameters**  
   - Finding the optimal parameters was another hurdle. I approached this by **iteratively testing different configurations** of the chosen model. For instance, with Extra Trees, I experimented with various numbers of estimators, split criteria, and other parameters to maximize performance.

3. **Decision on Individual vs. Ensemble Modeling**  
   - A major decision point was whether to:
     - Focus on an **individual model**.
     - Attempt **ensemble modeling** for potential improvements.  
   - I initially tried stacking two basic models and observed a **drop in accuracy**. Given the **time constraints of the competition**, I quickly decided to focus on enhancing the individual model instead of investing time in ensembling.  

4. **Quick Decision-Making**  
   - Despite the challenges, I made swift decisions to streamline the workflow, ultimately achieving a significant performance boost with the individual Extra Trees Regressor.  
   
**Result**: This strategic shift proved successful, leading to the discovery of the winning model! 🎉


## The Winner of the Competition: Extra Trees Regressor

### Why Extra Trees Regressor?
The **Extra Trees Regressor (Extremely Randomized Trees)** emerged as the best-performing model for the Russian housing price prediction. This model excels due to its use of **three levels of randomness**:
1. **Bootstrapping**: Generates diverse samples from the training data, promoting robust generalization.
2. **Random Feature Selection**: Selects a subset of features for splits, minimizing overfitting risks.
3. **Randomized Splits**: Randomizes split thresholds, reducing sensitivity to noise and complex patterns.

These features make Extra Trees particularly effective for high-dimensional data while maintaining computational efficiency.

---

### Best Configuration
The configuration that achieved the best results:  
```python
ExtraTreesRegressor(
    n_estimators=2000, 
    min_samples_split=4, 
    bootstrap=False, 
    random_state=42, 
    n_jobs=-1
)
Root Mean squared error: 12366669.20
R-squared = 0.671
KAGGLE SCORE = 12418350.13179
```
### Model Evolution
- Initially tested with **300 estimators**, yielding an **RMSE of 12,451,151.25**.  
- Incrementally increased the number of estimators to **2000**, further improving performance without any signs of overfitting.

---

### Why Increasing Estimators Doesn't Cause Overfitting?
Extra Trees leverages **averaging across a large number of trees with randomized splits**. This approach:  
- **Reduces variance** in predictions.  
- **Enhances generalization** by mitigating the impact of noisy or extreme data points.  
- **Stabilizes the model** as the number of estimators increases.  

Thus, increasing estimators improves robustness and consistency without compromising accuracy.

---

### Final Verdict
The **Extra Trees Regressor** delivered outstanding performance with excellent generalization capabilities, making it the **WINNER** for the Russian housing price prediction dataset.

