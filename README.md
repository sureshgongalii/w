# FIFA 20 Football Analytics: Player Performance & Market Value Prediction

A comprehensive data science project analyzing FIFA 20 player data to uncover insights about football player performance, characteristics, and market values using machine learning techniques.

## 📊 Project Overview

This project performs in-depth analysis of FIFA 20 player dataset to:
- Analyze player performance metrics and characteristics
- Predict player market values using machine learning
- Generate comprehensive data quality and exploratory data analysis reports
- Identify key factors influencing player ratings and market values

## 🗂️ Project Structure

```
├── FIFA-20 (FINAL).ipynb          # Main analysis notebook
├── PRCP-1004-FIFA-20              # Project documentation
├── SWEETVIZ_REPORT.html           # Automated EDA report
├── best_model.pkl                 # Trained machine learning model
├── players_20.csv                 # FIFA 20 player dataset
└── README.md                      # Project documentation
```

## 📋 Dataset Information

The `players_20.csv` contains comprehensive information about FIFA 20 players including:
- **Player Demographics**: Name, age, nationality, club
- **Physical Attributes**: Height, weight, preferred foot
- **Performance Metrics**: Overall rating, potential, skill ratings
- **Market Information**: Value, wage, release clause
- **Position Data**: Preferred positions and positional ratings

## 🔧 Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **Sweetviz** - Automated exploratory data analysis
- **Jupyter Notebook** - Interactive development environment

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost sweetviz jupyter
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sureshgongalii/-Football-Analytics-in-FIFA-Simulation.git
cd -Football-Analytics-in-FIFA-Simulation
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open `FIFA-20 (FINAL).ipynb` to explore the analysis

## 📈 Key Features

### Data Analysis
- **Player Performance Analysis**: Statistical analysis of player ratings and skills
- **Market Value Insights**: Analysis of factors affecting player market values
- **Position-based Analysis**: Performance comparison across different playing positions
- **Age and Potential Correlation**: Relationship between player age and potential ratings

### Machine Learning
- **Predictive Modeling**: Multiple trained models to predict player market values
- **Feature Engineering**: Creation of meaningful features from raw data
- **Model Evaluation**: Comprehensive model performance assessment
- **Saved Model**: Pre-trained model (`best_model.pkl`) for future predictions

### Automated Reporting
- **Sweetviz Report**: Interactive HTML report with comprehensive EDA
- **Data Quality Assessment**: Automated data profiling and quality checks
- **Visual Analytics**: Rich visualizations and statistical summaries

## 📊 Overall Model Performance Comparison

### Clustering Model

#### K-Means Clustering
- **Silhouette Score:** 0.68  
  Suggests **moderate cluster separation**, meaning that while distinct clusters are formed, some overlap may exist.  
- **Strengths:**  
  - Well-suited for **structured datasets** with clearly defined clusters.  
  - Works effectively when combined with **PCA for dimensionality reduction**, helping remove noise and improve clustering efficiency.  
- **Limitations:**  
  - Struggles with **uneven cluster densities**—if clusters vary significantly in size or shape, performance may degrade.  
  - Sensitive to **initial centroid placement** and requires tuning of the number of clusters (`k`).  
- **Ideal Use Cases:**  
  - Customer segmentation, anomaly detection, and exploratory data analysis where predefined clusters are needed.  

---

### Supervised Learning Models

#### XGBoost Regressor
- **R² Score:** 0.990  
  Almost **perfect variance explanation**, making it the **most powerful predictive model** in the analysis.  
- **Mean Squared Error (MSE):** 0.000  
  **Extremely low error**, indicating minimal deviation from actual values.  
- **Root Mean Squared Error (RMSE):** 0.017  
  Confirms **fine-grained precision**, ensuring stable predictions even in complex datasets.  
- **Mean Absolute Error (MAE):** 0.008  
  **Highly stable predictions**, minimizing outliers and major discrepancies.  
- **Max Error:** 0.370  
  Even in worst-case scenarios, predictions stay **well within a manageable threshold**.  
- **Strengths:**  
  - **Excels at handling nonlinear relationships**, feature interactions, and large datasets.  
  - **Optimized for gradient boosting**, offering superior accuracy and computational efficiency.  
- **Ideal Use Cases:**  
  - High-stakes predictive modeling such as **financial forecasting, risk assessment, and medical diagnostics**.  

---

#### Random Forest Regressor
- **R² Score:** 0.959  
  Maintains **strong predictive reliability**, though slightly below XGBoost.  
- **Mean Squared Error (MSE):** 0.001  
  **Low error**, but higher than XGBoost, suggesting minor inconsistencies.  
- **Root Mean Squared Error (RMSE):** 0.035  
  **Reliable predictions**, but less precise than XGBoost when handling highly nonlinear features.  
- **Mean Absolute Error (MAE):** 0.022  
  **Stable performance**, though variations occur more frequently than in XGBoost.  
- **Max Error:** 0.434  
  **Worse handling of extreme cases**, meaning outliers might impact predictions.  
- **Strengths:**  
  - **Works well for general-purpose regression tasks**, particularly when explainability is important.  
  - **Less prone to overfitting compared to boosting models**, making it a robust choice.  
- **Limitations:**  
  - Computationally **more expensive** than simpler models like linear regression.  
  - Slightly **less precise in high-dimensional, complex feature spaces**.  
- **Ideal Use Cases:**  
  - **General machine learning applications**, including **market analysis, fraud detection, and recommendation systems**.  

---

#### SVM Regressor
- **R² Score:** 0.908  
  Performs **reasonably well**, but **lags behind ensemble-based models**.  
- **Mean Squared Error (MSE):** 0.003  
  **Higher error**, indicating lower precision in predictions.  
- **Root Mean Squared Error (RMSE):** 0.053  
  Predictions vary significantly—**best suited for structured data** rather than unstructured data.  
- **Mean Absolute Error (MAE):** 0.039  
  **Less consistent than XGBoost or Random Forest**, impacting stability in high-dimensional datasets.  
- **Max Error:** 0.502  
  **Largest extreme deviation**, showing vulnerability to outliers and noisy data.  
- **Strengths:**  
  - **Effective for linear and moderately nonlinear data**, especially when kernel tuning is applied.  
  - Performs well when **data is highly structured and relatively low-dimensional**.  
- **Limitations:**  
  - **Computationally expensive** in large datasets, due to the complexity of kernel transformations.  
  - Less effective when dealing with **complex feature interactions**.  
- **Ideal Use Cases:**  
  - **Text classification, bioinformatics, and small-scale predictive modeling** where precision matters more than speed.  

---

#### K-Nearest Neighbors (KNN) Regressor
- **R² Score:** 0.822  
  **Lower variance explanation**, meaning weak generalization to new data.  
- **Mean Squared Error (MSE):** 0.005  
  **Higher error**, showing its limitations for large datasets.  
- **Root Mean Squared Error (RMSE):** 0.073  
  **Significant deviation**, making it less reliable for fine-tuned predictions.  
- **Mean Absolute Error (MAE):** 0.056  
  Predictions **fluctuate heavily**, reducing stability.  
- **Max Error:** 0.419  
  Less prone to extreme failures but still **unreliable for sensitive applications**.  
- **Strengths:**  
  - **Simple and easy to interpret**, making it great for small-scale problems.  
  - Works well in **localized pattern recognition**, especially in small datasets.  
- **Limitations:**  
  - **Struggles with scalability**, meaning performance deteriorates as dataset size increases.  
  - **Highly sensitive to distance metrics**, requiring careful parameter tuning.  
- **Ideal Use Cases:**  
  - **Image recognition, personalized recommendations, and low-complexity predictions**.  

---

#### Linear Regression
- **R² Score:** 0.908  
  Similar performance to SVM, indicating **moderate predictability but lacking adaptability**.  
- **Mean Squared Error (MSE):** 0.003  
  Performs adequately but **lags behind nonlinear models** in capturing complex relationships.  
- **Root Mean Squared Error (RMSE):** 0.053  
  Suggests **moderate deviations**, making it better suited for structured datasets.  
- **Mean Absolute Error (MAE):** 0.038  
  Slightly **better than SVM**, but **limited when handling diverse feature interactions**.  
- **Max Error:** 0.576  
  **Largest worst-case deviation**, reinforcing its limitations.  
- **Strengths:**  
  - **Fast and interpretable**, great for understanding simple trends.  
  - **Baseline model** for comparison before testing complex approaches.  
- **Limitations:**  
  - **Cannot handle nonlinear dependencies**, which restricts its applicability in diverse datasets.  
  - **Susceptible to over-simplification**, leading to biased predictions.  
- **Ideal Use Cases:**  
  - **Basic forecasting, risk modeling, and preliminary trend analysis**.  

---

### **Key Observations**

1. **K-Means Clustering (Silhouette Score: 0.68)**
   - Useful for **segmentation tasks**, providing moderately well-separated clusters.    
   - Works best with **dimensionality reduction methods like PCA** to enhance structure clarity.  

2. **XGBoost Regressor Dominates Supervised Learning**
   - **Best model** for high-accuracy predictions, **handling complex nonlinear relationships** exceptionally well.  

3. **Random Forest Regressor: A Strong Alternative**
   - **More interpretable than XGBoost** and **balances efficiency with stability**.  

4. **SVM & Linear Regression Are Best for Structured Data**
   - Work well for **predictable, mathematically manageable trends**, but **not ideal for feature-rich datasets**.  

5. **KNN Shows the Weakest Generalization**
   - Performs best in **small-scale datasets** but **fails in complex environments**.  

## 🔧 Challenges and Solutions in Data Analysis

During the data preprocessing and clustering phases, several challenges emerged that required a **methodical approach to data engineering and optimization**. Below is a structured analysis of key issues encountered and the techniques applied to resolve them.

### 1. Handling Missing Data
**Problem**: The dataset contained **structured missing values**, particularly in position-specific attributes:  
- **Goalkeepers lacked attacking attributes**, as those metrics are irrelevant to their role.  
- **Non-goalkeepers lacked goalkeeping attributes**, creating an inconsistent feature distribution.  

**Approach**:
- **Logical imputation**: Missing values were replaced with `0` where the absence was **intrinsic** rather than random.  
- **Feature pruning**: Columns with excessive missing values were dropped to maintain dataset **integrity and predictive robustness**.  

**Rationale**: Imputation ensured **feature completeness** for all players while maintaining the **structural relevance of attributes**. Feature pruning prevented **bias from incomplete or skewed distributions**.

### 2. Encoding Categorical Variables
**Problem**: Categorical variables such as **player position, preferred foot, and work rate** are **not suitable for K-Means clustering**, which relies on Euclidean distances. Encoding them **directly** would distort similarity measurements.  

**Approach**:
- **Feature selection**: Excluded categorical attributes from clustering, focusing only on numerical features that provide meaningful distance calculations.  
- **Alternative representations**: Considered ordinal encoding where categorical features had clear hierarchies, though not applied in this case.  

**Rationale**: Clustering models **require numerical consistency**—direct categorical inclusion could lead to **artificial separations** rather than capturing **true patterns in player attributes**.

### 3. Managing Outliers
**Problem**: K-Means clustering is highly sensitive to scale disparities—**extreme values can distort centroid placement**, affecting cluster cohesion.  

**Approach**:
- **Standardization (Z-score normalization)**: Converted all numerical features to a common scale, ensuring proportionate influence on clustering.  
- **Robust scaling**: Considered for datasets with extreme skew, though Z-score was sufficient in this case.  

**Rationale**: Standardizing features prevents **high-magnitude attributes from overshadowing lower-magnitude ones**, leading to **balanced cluster formation**.

### 4. Addressing High Dimensionality
**Problem**: The dataset contained **104 features**, increasing computational complexity and risk of **overfitting** in clustering.  

**Approach**:
- **Principal Component Analysis (PCA)**: Reduced dimensionality while retaining **maximum variance**, ensuring meaningful clustering.  
- **Feature selection**: Verified the most informative variables through variance thresholds before applying PCA.  

**Rationale**: PCA improves **clustering efficiency** and enhances interpretability by focusing on **key data patterns** rather than raw feature count.

### 5. Cluster Optimization
**Problem**: Determining the **optimal number of clusters (`k`)** required empirical validation:  
- The **Elbow Method** provided general guidance but lacked definitive resolution.  
- **Silhouette Scores** fluctuated, leading to potential inconsistencies in cluster formation.  

**Approach**:
- **Multi-metric evaluation**:  
  - **Elbow Method** identified diminishing returns in variance reduction.  
  - **Silhouette Analysis** validated cluster cohesion.  
  - **Davies-Bouldin Index** assessed separation strength between clusters.  

**Rationale**: Using multiple evaluation techniques **ensures a robust choice of `k`**, reducing **arbitrary decision-making** and improving **clustering stability**.

## 🔍 Usage Examples

### Loading the Dataset
```python
import pandas as pd
df = pd.read_csv('players_20.csv')
print(df.head())
```

### Using the Trained Model
```python
import pickle
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)
# Use model for predictions
```

### Viewing the EDA Report
Open `SWEETVIZ_REPORT.html` in your web browser to explore the interactive data analysis report.

## 📝 Results

The project successfully:
- ✅ Analyzed 18,000+ FIFA 20 player records
- ✅ Built and compared multiple predictive models for player market values
- ✅ Achieved 99% accuracy with XGBoost Regressor (R² = 0.990)
- ✅ Generated comprehensive data quality reports
- ✅ Identified key performance indicators for football players
- ✅ Created visualizations for data-driven insights
- ✅ Implemented effective clustering for player segmentation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Suresh Gongali**
- GitHub: [@sureshgongalii](https://github.com/sureshgongalii)

## 🙏 Acknowledgments

- FIFA 20 dataset providers
- Open source community for the amazing tools and libraries
- Football analytics community for inspiration and insights

## 📞 Contact

For questions or suggestions, please open an issue on GitHub or contact the author directly.

---

⭐ If you found this project helpful, please give it a star!
