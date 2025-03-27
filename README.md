# ML Algorithm Visualizer

An educational tool that creates step-by-step visualizations of machine learning algorithms to help students understand how they work.

## Project Overview

ML Algorithm Visualizer is designed to provide clear, intuitive animations of machine learning algorithms operating on 2D data. The goal is to make abstract ML concepts more concrete by showing exactly how algorithms progress through their steps to arrive at a solution.

### Core Features:
- Step-by-step visualization of algorithm iterations
- Focus on 2D data for intuitive visual understanding
- High-level algorithm behavior visualization (not internal calculations)
- Educational focus for students learning ML concepts

## Supported Algorithms

### Supervised Learning - Regression
- Linear Regression
- Polynomial Regression
- Ridge/Lasso Regression (regularization effects)
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression

### Supervised Learning - Classification
- Logistic Regression
- Decision Trees
- Random Forest
- Bagging (e.g., Random Forest as a specific example)
- Boosting (AdaBoost, Gradient Boosting, XGBoost)
- Support Vector Machines (with different kernels)
- K-Nearest Neighbors
- Naive Bayes

### Unsupervised Learning
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- PCA (Principal Component Analysis)
- t-SNE
- UMAP

### Neural Networks (Future)
- Simple Neural Networks (forward/backward propagation)
- Convolutional Neural Networks (visualization of filters)
- Recurrent Neural Networks (sequential data processing)

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-algorithm-visualizer.git
cd ml-algorithm-visualizer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```