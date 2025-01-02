# Diabetes Prediction Using Logistic Regression and Naive Bayes

This project aims to predict whether a person has diabetes based on various medical parameters. The dataset used is the **Pima Indians Diabetes Database**, which contains information about 768 individuals, including their medical attributes and diabetes status. The project implements two machine learning models—**Logistic Regression** and **Naive Bayes**—to predict the likelihood of diabetes.

## Project Structure

- **main.py**: Python script containing the data preprocessing, analysis, and model training for logistic regression and Naive Bayes classifiers.
- **README.md**: This file, which explains the project and its structure.

## Libraries Used

- `numpy`: For array manipulation and mathematical operations.
- `pandas`: For handling data in tabular form (DataFrame).
- `matplotlib`: For visualizing data and model results.
- `seaborn`: For plotting more advanced visualizations like heatmaps and distributions.
- `sklearn`: For implementing machine learning models, such as Logistic Regression, Naive Bayes, and evaluation metrics.

## Dataset

The project uses the **Pima Indians Diabetes Dataset**, which contains the following features:
- `Preg`: Number of pregnancies
- `Plas`: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- `Pres`: Diastolic blood pressure (mm Hg)
- `Skin`: Triceps skin fold thickness (mm)
- `Test`: 2-Hour serum insulin (mu U/ml)
- `Mass`: Body mass index (weight in kg/(height in m)^2)
- `Pedi`: Diabetes pedigree function
- `Age`: Age (years)
- `Class`: Target variable indicating whether the person has diabetes (1) or not (0)

## Data Analysis and Visualization

1. **Data Inspection**:
   - The dataset is loaded and basic information, including data types, missing values, and summary statistics, is explored.
   - Visualizations include histograms, box plots, count plots, and heatmaps to understand the distribution and relationships among features.
   
2. **Handling Missing Data**:
   - Missing values are handled by filling with appropriate strategies, such as using the mean for glucose (`Plas`), blood pressure (`Pres`), and the median for skin thickness (`skin`), serum insulin (`test`), and BMI (`mass`).

3. **Correlations**:
   - A correlation matrix is generated to identify relationships between features, particularly focusing on the correlation between the features and the target variable (`class`).

## Machine Learning Models

### 1. Logistic Regression

Logistic Regression is used to model the relationship between the medical features and the binary outcome (diabetes or not). The dataset is split into training and testing sets, and the model is trained using the training data. 

- **Evaluation**: The model's performance is assessed using accuracy, confusion matrix, and classification report, including precision, recall, and F1-score.

### 2. Naive Bayes

Naive Bayes is another classification algorithm used to predict diabetes. The model assumes independence between the features and calculates probabilities based on Bayes' theorem.

- **Evaluation**: Similar to Logistic Regression, the model's performance is evaluated using accuracy, confusion matrix, and classification report.

## Results

- **Logistic Regression** achieved an accuracy of **78.79%** with a F1-score of **0.65**.
- **Naive Bayes** achieved an accuracy of **75.76%** with a F1-score of **0.63**.

## Steps to Run the Code

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
Install the required libraries:

bash
Copy code
pip install -r requirements.txt
Run the main.py script:

bash
Copy code
python main.py
Visualizations
Histograms: The distribution of features like age, BMI, glucose, etc.
Box Plots: To detect outliers in the features.
Confusion Matrices: To evaluate the performance of the models.
ROC Curve: For Naive Bayes to visualize the tradeoff between true positive rate and false positive rate.
Conclusion
The project explores diabetes prediction using two widely used classifiers. Logistic Regression performed slightly better than Naive Bayes on the given dataset. However, both models provide valuable insights and could be further improved by experimenting with other classifiers or tuning hyperparameters.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
The Pima Indians Diabetes Database is publicly available and was used for this analysis.
sql
Copy code

Replace `yourusername` with your actual GitHub username, and feel free to adjust any sections if your
