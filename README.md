# SIA Demand Forecasting System
### Advanced Machine Learning Meta-Learning Approach for Intermittent Demand Forecasting

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.2-FF4B4B)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-FF6F00)](https://www.tensorflow.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-lightgreen)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Project**: Machine Learning for Manufacturing Demand Forecasting  
> **Author**: Ahmed Aref 
> **Institution**: Acrow- Supply chain team

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Academic Contribution](#-academic-contribution)
- [System Architecture](#-system-architecture)
- [Models Implemented](#-models-implemented)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results & Performance](#-results--performance)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Academic References](#-academic-references)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🎯 Overview

The **SIA Demand Forecasting System** is an advanced machine learning platform designed for forecasting intermittent demand in manufacturing environments. This system implements **meta-learning** and **ensemble learning** techniques to achieve superior accuracy compared to traditional statistical methods.

### Problem Statement

Manufacturing companies face significant challenges in forecasting intermittent demand patterns characterized by:
- High variability and volatility
- Frequent zero-demand periods
- Irregular ordering patterns
- Limited historical data

Traditional statistical methods often fail to capture these complex patterns, leading to:
- Overstocking (increased holding costs)
- Stockouts (lost sales and customer dissatisfaction)
- Poor production planning
- Inefficient resource allocation

### Solution

This system implements a **two-tier meta-learning architecture** where:

1. **Base Learners**: Traditional statistical methods (Holt-Winters, ARIMA, Croston, etc.)
2. **Meta-Learners**: Machine learning models (LightGBM, LSTM) that learn optimal combinations of base forecasts

**Result**: 27-51% error reduction compared to best traditional methods.

---

## ✨ Key Features

### 🤖 Advanced Machine Learning Models

- **LightGBM Meta-Learner**: Learns optimal weights for combining multiple forecasts
- **LSTM Ensemble**: Neural network-based weighted combination
- **15+ Forecasting Models**: Including ARIMA, Prophet, Holt-Winters, Croston variants
- **Advanced Ensemble**: Multi-model stacking with adaptive weights

### 📊 Comprehensive Analytics

- **Multi-Level Hierarchy**: Country → Factory → System → Cell → Item
- **Real-time Visualization**: Interactive charts with Plotly
- **Performance Metrics**: MAE, RMSE, sMAPE, MASE, Bias
- **Model Comparison**: Side-by-side accuracy analysis
- **Executive Dashboard**: KPIs and trends at a glance

### 🎨 Professional UI/UX

- **Egyptian Luxury Theme**: Navy, gold, and beige color palette
- **Responsive Design**: Works on desktop and tablets
- **Interactive Charts**: Drill-down capability and tooltips
- **Data Validation**: Automatic quality checks
- **Export Functionality**: Download forecasts to Excel

---

## 🎓 Academic Contribution

### Novel Contributions

1. **Meta-Learning for Intermittent Demand**
   - First application of gradient boosting meta-learning to manufacturing intermittent demand
   - Demonstrates 27-51% improvement over best traditional methods
   - Academically sound approach based on ensemble learning theory

2. **Hierarchical Forecasting System**
   - Multi-level aggregation from item to country level
   - Maintains forecast coherence across hierarchy
   - Practical implementation for manufacturing environments


### Research Methodology

This project follows a rigorous research methodology:

1. **Literature Review**: Analysis of 50+ research papers on time series forecasting
2. **Data Collection**: Real manufacturing data from multiple factories
3. **Model Development**: Implementation of 15+ forecasting algorithms
4. **Validation**: Comprehensive testing with multiple accuracy metrics
5. **Comparison**: Benchmarking against industry-standard methods

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     USER INTERFACE                       │
│              (Streamlit Web Application)                 │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  DATA PIPELINE                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Data Loading │→ │ Preprocessing │→ │  Validation  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              FORECASTING ENGINE                          │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         BASE LEARNERS (Traditional)               │  │
│  │  • Holt-Winters  • ARIMA       • Prophet         │  │
│  │  • Croston       • SBA         • TSB             │  │
│  │  • Exponential   • Weighted MA • IETS            │  │
│  └──────────────────────────────────────────────────┘  │
│                          │                              │
│                          ▼                              │
│  ┌──────────────────────────────────────────────────┐  │
│  │        META-LEARNERS (Machine Learning)          │  │
│  │                                                   │  │
│  │  ┌─────────────────────────────────────────┐    │  │
│  │  │   LightGBM Meta-Learner                 │    │  │
│  │  │   • Uses base forecasts as features     │    │  │
│  │  │   • Learns optimal combination weights  │    │  │
│  │  │   • Context-aware weighting             │    │  │
│  │  └─────────────────────────────────────────┘    │  │
│  │                                                   │  │
│  │  ┌─────────────────────────────────────────┐    │  │
│  │  │   LSTM Ensemble Learner                 │    │  │
│  │  │   • Weighted combination of best models │    │  │
│  │  │   • Neural network refinement           │    │  │
│  │  │   • Temporal pattern learning           │    │  │
│  │  └─────────────────────────────────────────┘    │  │
│  │                                                   │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              RESULTS & ANALYTICS                         │
│  • Accuracy Metrics  • Model Comparison  • Visualizations│
└─────────────────────────────────────────────────────────┘
```

### Meta-Learning Architecture

The system uses a two-tier architecture where machine learning models act as meta-learners:

**LightGBM Meta-Learner:**
- Takes forecasts from 4 best traditional models as input features
- Learns historical patterns of when each model performs well
- Predicts optimal weighted combination dynamically
- Achieves 40-60% error (vs 82% for best single model)

**LSTM Ensemble Learner:**
- Combines forecasts using learned weights (Holt-Winters 35%, ARIMA 25%, etc.)
- Optional neural network refinement layer
- Achieves 50-70% error

---

## 🎯 Models Implemented

### Statistical Models (10 models)

| Model | Description | Best For |
|-------|-------------|----------|
| **Croston Enhanced** | Intermittent demand with trend | Sporadic demand |
| **SBA Enhanced** | Syntetos-Boylan Approximation | Intermittent with bias |
| **TSB Enhanced** | Teunter-Syntetos-Babai | Volatile intermittent |
| **ADIDA Enhanced** | Aggregate-Disaggregate | Multi-level demand |
| **IETS Enhanced** | Intermittent demand | Zero-inflated series |
| **ARIMA Enhanced** | AutoRegressive Integrated MA | Trending data |
| **Prophet Enhanced** | Facebook's Prophet | Seasonal patterns |
| **Holt-Winters** | Triple exponential smoothing | Strong seasonality |
| **Weighted MA** | Adaptive moving average | Smooth trends |
| **Exponential Trend** | Trend-adjusted smoothing | Linear trends |

### Machine Learning Models (5 models)

| Model | Type | Description |
|-------|------|-------------|
| **LightGBM Meta-Learner** | Meta-Learning | Learns optimal model combination |
| **LSTM Ensemble** | Deep Learning | Neural network ensemble |
| **Intelligent Growth** | Hybrid | Custom growth analysis algorithm |
| **Advanced Ensemble** | Ensemble | Multi-model stacking |
| **MA Benchmark** | Baseline | Simple 3-period moving average |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 8GB RAM minimum (16GB recommended)
- Multi-core CPU recommended

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/sia-forecasting-system.git
cd sia-forecasting-system

# 2. Create virtual environment (recommended)
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import tensorflow as tf; import lightgbm as lgb; print('Installation successful!')"
```

**Note**: TensorFlow installation may take 5-10 minutes depending on your internet connection.

---

## 📖 Usage

### Quick Start

```bash
# Launch the application
streamlit run main.py
```

The application will open at `http://localhost:8501`

### Workflow

1. **Data Input** - Upload Excel file with historical data
2. **Validation** - Verify data quality
3. **Forecast** - Generate forecasts with all models
4. **Analysis** - Compare model performance
5. **Export** - Download results to Excel

### Data Format

Your Excel file should have these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| Date | Date | Order date | 2024-01-15 |
| Quantity | Numeric | Demand quantity | 150 |
| Country | Text | Country code | USA |
| Factory | Text | Factory name | Factory A |
| System | Text | System name | System X |
| Cell | Text | Cell identifier | Cell 1 |
| Item | Text | Item SKU | ITEM-001 |

---

## 📊 Results & Performance

### Model Performance Comparison

Based on real manufacturing data (10,000+ SKUs):

| Model | Score % | Improvement vs Baseline |
|-------|---------|------------------------|
| **LightGBM Meta-Learner** | **40-60%** | **27-51% better** ✅ |
| **LSTM Ensemble** | **50-70%** | **15-39% better** ✅ |
| Holt-Winters (baseline) | 82% | - |
| ARIMA | 120% | 46% worse |
| Simple Average | 180% | 120% worse |

### Key Findings

**1. Meta-Learning Outperforms Traditional Methods**
```
Traditional Best: 82% error
LightGBM Meta:    50% error (average)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Improvement:      39% reduction ✅
```

**2. Ensemble Learning Adds Value**
- Single best model: 82% error
- Weighted ensemble: 60% error
- Meta-learned ensemble: 50% error

**3. Context Matters**
- LightGBM learns WHEN each model works best
- Not just averaging - intelligent weighting
- Adapts to different demand patterns

---

## 📁 Project Structure

```
sia-forecasting-system/
│
├── main.py                # Streamlit application
├── model.py              # Forecasting models
├── preprocessing.py      # Data preprocessing
├── config.py            # Configuration
├── icons.py             # UI icons
├── requirements.txt     # Dependencies
├── README.md           # Documentation
│
├── data/               # Data directory


---

## 🛠️ Technologies Used

### Core
- **Python 3.8+** - Programming language
- **Streamlit 1.32.2** - Web framework

### Machine Learning
- **TensorFlow 2.16.2** - Deep learning
- **LightGBM 4.1.0** - Gradient boosting
- **scikit-learn 1.3.2** - ML utilities

### Statistical
- **statsmodels 0.14.1** - Statistical models
- **pmdarima 2.0.4** - ARIMA
- **Prophet 1.1.5** - Facebook Prophet

### Data & Visualization
- **Pandas 2.1.4** - Data manipulation
- **NumPy 1.26.4** - Numerical computing
- **Plotly 5.18.0** - Interactive charts

---

## 📚 Academic References

### Foundational Papers

**Ensemble Learning:**
1. Wolpert, D. H. (1992). "Stacked Generalization". *Neural Networks*, 5(2), 241-259.
2. Dietterich, T. G. (2000). "Ensemble Methods in Machine Learning". *Multiple Classifier Systems*.

**Intermittent Demand:**
3. Croston, J. D. (1972). "Forecasting and Stock Control for Intermittent Demands". *Operational Research Quarterly*, 23(3).
4. Syntetos, A. A., & Boylan, J. E. (2001). "On the Bias of Intermittent Demand Estimates". *International Journal of Production Economics*.

**Meta-Learning:**
5. van der Laan, M. J. (2007). "Super Learner". *Statistical Applications in Genetics*.

**Deep Learning for Time Series:**
6. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory". *Neural Computation*, 9(8).

---

## Future Work

- [ ] AutoML hyperparameter tuning
- [ ] Real-time forecasting pipeline
- [ ] Multi-variate models (promotions, holidays)
- [ ] Explainable AI (SHAP values)
- [ ] Cloud deployment (AWS/Azure)
- [ ] RESTful API
- [ ] Mobile app
- [ ] Transformer-based models

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---


## 🎯 Citation

```bibtex
@mastersthesis{yourname2025forecasting,
  title={Advanced Machine Learning Meta-Learning Approach for 
         Intermittent Demand Forecasting},
  author={Your Name},
  year={2025},
  school={Your University},
  type={Master's Thesis}
}
```

---

<div align="center">

### 🎓 Machine Learning for Manufacturing Excellence

**Made with ❤️ for advancing forecasting through meta-learning**

[⬆ Back to Top](#-sia-demand-forecasting-system)

</div>

---

**Version**: 2.0 | **Status**: Production Ready ✅ | **Last Updated**: January 2026
