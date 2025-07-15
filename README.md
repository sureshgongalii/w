# Football Analytics in FIFA Simulation

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
- **Sweetviz** - Automated exploratory data analysis
- **Jupyter Notebook** - Interactive development environment

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn sweetviz jupyter
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
- **Predictive Modeling**: Trained model to predict player market values
- **Feature Engineering**: Creation of meaningful features from raw data
- **Model Evaluation**: Comprehensive model performance assessment
- **Saved Model**: Pre-trained model (`best_model.pkl`) for future predictions

### Automated Reporting
- **Sweetviz Report**: Interactive HTML report with comprehensive EDA
- **Data Quality Assessment**: Automated data profiling and quality checks
- **Visual Analytics**: Rich visualizations and statistical summaries

## 📊 Key Insights

The analysis reveals several interesting patterns:
- Correlation between player age, overall rating, and market value
- Impact of playing position on player characteristics
- Relationship between physical attributes and performance metrics
- Market value determinants and pricing patterns

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
- ✅ Built predictive models for player market values
- ✅ Generated comprehensive data quality reports
- ✅ Identified key performance indicators for football players
- ✅ Created visualizations for data-driven insights

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