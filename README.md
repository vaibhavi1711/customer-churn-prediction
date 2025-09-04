Customer Churn Prediction 📊


This project predicts whether a telecom customer will churn or stay, using the Telco Customer Churn dataset. It includes data cleaning, preprocessing, feature engineering, and model training with Logistic Regression, Random Forest, and XGBoost.

The project is fully reproducible with Python, Scikit-learn, and Jupyter, and provides ROC curve comparisons and feature importance analysis to connect technical results with real business insights.

⚙️ Tech Stack
- **Python**
- **Pandas, NumPy** – data manipulation
- **Scikit-learn** – ML models & evaluation
- **XGBoost** – gradient boosting model
- **Matplotlib, Seaborn** – data visualization
- **Jupyter Notebook** – EDA & experiments

📝 Project Workflow
1. **Data Loading & Exploration**  
   - Checked dataset size, missing values, churn distribution  
   - Visualized churn vs. non-churn balance  

2. **Data Cleaning & Preprocessing**  
   - Removed non-predictive `customerID`  
   - Converted `TotalCharges` to numeric and filled missing values  
   - Encoded categorical variables (Label Encoding for binary, Category Codes for multi-class)  

3. **Feature Engineering**  
   - Standardized features with `StandardScaler`  
   - Split dataset into **80% train / 20% test**  

4. **Model Training & Evaluation**  
   - Trained three models:  
     - Logistic Regression (baseline)  
     - Random Forest  
     - XGBoost  
   - Evaluated with **Accuracy, Precision, Recall, F1, ROC-AUC**  

5. **Visualization**  
   - Plotted **ROC curves** to compare models  
   - Extracted **feature importances** from Random Forest & XGBoost  