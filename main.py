import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Title of the Dashboard
st.title("🧑‍💻 Data Science Dashboard for Predictive Modeling")

# Step 1: Upload Dataset
st.header("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read dataset
        df = pd.read_csv(uploaded_file)
        st.write("📋 **Dataset Preview**")
        st.write(df.head())

        # Checking for empty datasets
        if df.empty:
            st.error("The uploaded dataset is empty! Please upload a valid dataset.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()

    # Step 2: Data Preprocessing
    st.header("⚙️ Data Preprocessing")
    
    # Handling missing values
    try:
        st.subheader("Handle Missing Values")
        missing_option = st.selectbox("Select option for missing values:", ("Drop missing rows", "Fill with mean"))
        
        if missing_option == "Drop missing rows":
            df.dropna(inplace=True)
            st.write("Missing values dropped.")
        elif missing_option == "Fill with mean":
            df.fillna(df.mean(), inplace=True)
            st.write("Missing values filled with mean.")
    except Exception as e:
        st.error(f"Error in missing values handling: {e}")
        st.stop()
    
    # Encoding categorical variables
    try:
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            st.subheader("Encode Categorical Variables")
            encoding_option = st.checkbox("Apply Label Encoding to Categorical Variables?")
            if encoding_option:
                le = LabelEncoder()
                for col in categorical_cols:
                    df[col] = le.fit_transform(df[col])
                st.write("Categorical variables encoded.")
        else:
            st.write("No categorical variables found in the dataset.")
    except Exception as e:
        st.error(f"Error encoding categorical variables: {e}")
        st.stop()

    # Feature scaling
    try:
        st.subheader("Scale Features")
        scale_option = st.checkbox("Apply Standard Scaling?")
        if scale_option:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df)
            df = pd.DataFrame(scaled_features, columns=df.columns)
            st.write("Features scaled.")
    except Exception as e:
        st.error(f"Error in feature scaling: {e}")
        st.stop()

    st.write("🔍 **Preprocessed Dataset**")
    st.write(df.head())

    # Step 3: Model Selection
    st.header("🚀 Model Training")
    st.subheader("Select Target Variable")
    
    target_variable = st.selectbox("Select the target variable (prediction column):", df.columns)
    
    if target_variable:
        try:
            X = df.drop(target_variable, axis=1)
            y = df[target_variable]

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            st.subheader("Choose Model")
            model_option = st.selectbox("Choose a machine learning model:", ("Linear Regression", "Decision Tree", "Random Forest"))
            
            # Initialize and train model
            model = None
            if model_option == "Linear Regression":
                model = LinearRegression()
            elif model_option == "Decision Tree":
                model = DecisionTreeRegressor()
            elif model_option == "Random Forest":
                model = RandomForestRegressor()

            # Model training
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            except Exception as e:
                st.error(f"Error during model training: {e}")
                st.stop()

            # Step 4: Model Evaluation
            st.header("📊 Model Evaluation")

            # Metrics for regression
            try:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                st.write(f"**Mean Squared Error (MSE):** {mse}")
                st.write(f"**R-squared (R²):** {r2}")
            except Exception as e:
                st.error(f"Error calculating model metrics: {e}")
                st.stop()

            # Visualization of actual vs predicted values
            st.subheader("Actual vs Predicted Values")
            try:
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error in generating plots: {e}")
                st.stop()

            # Feature importance (for tree-based models)
            if model_option in ["Decision Tree", "Random Forest"]:
                st.subheader("Feature Importance")
                try:
                    feature_importance = model.feature_importances_
                    importance_df = pd.DataFrame({
                        "Feature": X.columns,
                        "Importance": feature_importance
                    }).sort_values(by="Importance", ascending=False)

                    fig, ax = plt.subplots()
                    sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error displaying feature importance: {e}")
                    st.stop()

        except Exception as e:
            st.error(f"Error in model selection or data splitting: {e}")
            st.stop()

    # Step 5: Download the Model
    st.header("📥 Download Trained Model")
    if st.button("Download Model"):
        try:
            with open("trained_model.pkl", "wb") as f:
                pickle.dump(model, f)
            st.download_button(label="Download Trained Model", data=open("trained_model.pkl", "rb"), file_name="trained_model.pkl")
        except Exception as e:
            st.error(f"Error saving model: {e}")
            st.stop()

else:
    st.write("Please upload a CSV file to get started!")
