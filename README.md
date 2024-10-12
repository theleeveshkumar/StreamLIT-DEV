# 🧑‍💻 Data Science Dashboard for Predictive Modeling

This Streamlit application is designed to simplify the process of training regression models for predictive modeling. Users can upload a dataset, preprocess the data, choose from different machine learning models, evaluate the model's performance, and download the trained model. The application supports linear regression, decision trees, and random forest regressors.

## 🚀 Features

1. **Dataset Upload**:
   - Upload CSV files for predictive modeling.
   - Preview the uploaded dataset.

2. **Data Preprocessing**:
   - Handle missing values by either dropping rows or filling missing values with the mean.
   - Encode categorical variables using Label Encoding.
   - Scale features using Standard Scaler (optional).

3. **Model Selection and Training**:
   - Choose from three machine learning models:
     - Linear Regression
     - Decision Tree Regressor
     - Random Forest Regressor
   - Select the target variable for prediction.
   - Split the data into training and test sets (80/20 split).

4. **Model Evaluation**:
   - Evaluate the model using the following metrics:
     - Mean Squared Error (MSE)
     - R-squared (R²)
   - Visualize actual vs. predicted values using a scatter plot.
   - Display feature importance for Decision Tree and Random Forest models.

5. **Model Download**:
   - Download the trained model as a `.pkl` file.

## 🛠️ Tech Stack

- **Streamlit**: Framework for building web applications in Python.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **scikit-learn**: Machine learning library for model training and evaluation.
- **Matplotlib & Seaborn**: For data visualization.
- **Pickle**: To save the trained machine learning model.

## 📂 How to Run the Application

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/data-science-dashboard.git
cd data-science-dashboard
```

### 2. Install the Required Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

### 4. Upload a Dataset

Once the app is running, you can upload your dataset in CSV format for further processing.

## 🎯 Usage Workflow

1. **Upload Dataset**: Upload your CSV file and check the dataset preview.
2. **Data Preprocessing**: Handle missing values, encode categorical variables, and scale features as needed.
3. **Model Selection**: Choose a machine learning model and target variable for training.
4. **Model Training**: The app will train the model using 80% of the data and evaluate on the remaining 20%.
5. **Evaluation**: Review the model’s performance metrics and feature importance.
6. **Download Model**: Once satisfied with the model, download the trained model for further use.

## 🖼️ Screenshots

(Include screenshots here to show the dataset upload, model selection, evaluation, and download options.)
![image](https://github.com/user-attachments/assets/2fe8daf9-6c8f-4c22-a2bc-667799b1c288)


## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Feel free to submit a pull request or open an issue to propose changes or request new features.
