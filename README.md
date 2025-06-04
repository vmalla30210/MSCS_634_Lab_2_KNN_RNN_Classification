# Lab 2: Classification Using KNN and RNN Algorithms on Wine Dataset

**Name:** Vishnu Mallam  
**Course:** Advanced Big Data and Data Mining  
**Assignment:** KNN and RNN Classification on Wine Dataset

---

## Overview

This lab compares the classification performance of K-Nearest Neighbors (KNN) and Radius Neighbors (RNN) algorithms on the Wine Dataset from scikit-learn. The study evaluates how different parameter values affect model accuracy and discusses the suitability of each algorithm for this dataset.

---

## Dataset Information

- **Dataset:** Wine Dataset (from sklearn.datasets)  
- **Features:** 13 chemical properties of wine  
- **Classes:** 3 wine classes (`class_0`, `class_1`, `class_2`)  
- **Total Samples:** 178  
- **Class Distribution:** 59, 71, 48 samples respectively  
- **Train/Test Split:** 80/20 (142 training samples, 36 testing samples)  

---

## Methodology

### Data Preprocessing

- Applied **StandardScaler** for feature normalization to improve distance-based algorithm performance.  
- Used **Stratified train-test split** to maintain class distribution across sets.  
- Performed exploratory data analysis with visualizations to understand data distribution.

### Model Implementation

#### K-Nearest Neighbors (KNN)

- Tested **K values:** 1, 5, 11, 15, 21  
- Evaluation metric: Classification accuracy  
- Best performance at **K=11** with **100% test accuracy**

#### Radius Neighbors (RNN)

- Tested **Radius values:** 350, 400, 450, 500, 550, 600  
- Evaluation metric: Classification accuracy  
- Best performance at **Radius=350** with **38.89% test accuracy**

---

## Key Results

| Model | Best Parameter | Training Accuracy | Test Accuracy |
|-------|----------------|-------------------|---------------|
| KNN   | K = 11         | 96.48%            | 100%          |
| RNN   | Radius = 350   | 40.14%            | 38.89%        |

- **KNN** demonstrated excellent generalization with minimal overfitting.  
- **RNN** showed poor performance likely due to radius values too large for the scaled feature space.

---

## Model Comparison and Insights

- **KNN** significantly outperforms **RNN** on this dataset.  
- KNN shows robust and consistent high performance across different K values.  
- RNN's radius values might be unsuitable; smaller radius or alternate distance metrics may improve performance.  
- Wine dataset’s uniform feature distribution favors KNN.

---

## When to Use Each Algorithm

| Use KNN When:                              | Use RNN When:                            |
|-------------------------------------------|-----------------------------------------|
| Dataset has uniform density distribution  | Identifying outliers or anomalies       |
| Consistent classification performance     | Dataset has varying density regions     |
| Dataset size is manageable                 | Excluding distant neighbors by distance |
| Features are properly scaled               | Neighborhood size needs to be distance constrained |

---

## Visualizations Included

- Class distribution (bar and pie charts)  
- KNN accuracy trends across different K values  
- RNN accuracy trends across different radius values  
- Training vs test accuracy comparisons  
- Best model performance comparison

---

## Technical Implementation

- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn  
- Modular code structure with stepwise progression  
- Comprehensive error handling especially for RNN edge cases  
- Detailed logging of results and parameter testing  
- Professional visualizations with annotations for clarity

---

## Challenges and Solutions

| Challenge                     | Solution                                          |
|-------------------------------|--------------------------------------------------|
| RNN poor performance          | Tested multiple radius values, identified too-large radii |
| Feature scaling importance    | Applied StandardScaler before modeling            |
| Parameter selection complexity| Systematic hyperparameter testing                  |
| Avoiding overfitting          | Included training accuracy and classification reports |

---

## Conclusions

- KNN is the superior classifier on the Wine dataset, achieving perfect test accuracy.  
- Feature scaling is essential for distance-based classifiers.  
- RNN requires careful radius tuning; large radii degrade performance.  
- The uniform distribution of wine dataset features makes KNN highly suitable.  
- Hyperparameter tuning is crucial for optimal results.

---

## Future Improvements

- Test smaller radius values for RNN (e.g., 0.1, 0.5, 1.0, 2.0)  
- Implement cross-validation for robust model evaluation  
- Explore alternative distance metrics (Manhattan, Minkowski)  
- Perform feature selection to identify most discriminative features  
- Compare performance with other classifiers such as SVM and Random Forest
