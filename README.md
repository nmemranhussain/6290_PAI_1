# Regression Modeling with H2O: Applying GLM and Artificial Neural Networks (ANN) for Predictive Accuracy  

### Basic Information

* **Person or organization developing model**: Patrick Hall, `jphall@gwu.edu` & N M Emran Hussain `nmemran.hussain@gwu.edu`
* **Model date**: June, 2025
* **Model version**: 1.0
* **License**: [Apache License 2.0](https://github.com/nmemranhussain/RML_A_1_Group_11/blob/main/LICENSE)
* **Model implementation code**: [Assignment_1](https://github.com/nmemranhussain/6290_PAI_1/blob/main/Assignment_1_final.ipynb), [Assignment_2](https://github.com/nmemranhussain/6290_PAI_1/blob/main/Assignment_2_final.ipynb)

### Intended Use
* **Primary intended uses**: This project is designed as educational and practical exercises to explore supervised machine learning techniques—specifically, Generalized Linear Models (GLM) and Artificial Neural Networks (ANN)—for solving regression problems using the H2O platform. The goal is to understand model formulation, data preprocessing, parameter tuning, model evaluation, and performance comparison. It serves as hands-on demonstrations of how to build predictive models on real-world tabular datasets, offering exposure to tools like H2O's Python interface, Colab integration, and model metrics such as RMSE and R². These projects are suitable for academic learning, prototyping predictive solutions, and developing foundational skills in automated machine learning systems.
* **Out-of-scope use cases**: This project is not intended for high-stakes, production-level deployment or decision-making in regulated or mission-critical environments (e.g., healthcare diagnostics, financial risk modeling, or autonomous systems). The models (GLM and ANN) are not intended for real-time, high-stakes automated decision-making without human oversight. Any use beyond an educational example is out-of-scope.

### Training Data

* **Data dictionary:**  

| Name | Modeling Role | Measurement Level | Description |  
| -------- | ---------------- | -------------------- | ------------- |  
| id | Identifier | Nominal | Unique identifier for each loan record. |  
| bad\_loan | Target | Binary | Indicates if the loan went bad (1 = bad loan, 0 = good loan). |  
| GRP\_REP\_home\_ownership | Predictor | Ordinal | Encoded home ownership category (grouped and possibly imputed). |  
| GRP\_addr\_state | Predictor | Nominal  | Encoded U.S. state of the borrower's address. |  
| GRP\_home\_ownership | Predictor | Ordinal | Encoded version of the borrower's home ownership status. |  
| GRP\_purpose | Predictor | Nominal  | Encoded purpose for which the loan was requested. |  
| GRP\_verification\_status | Predictor | Ordinal | Encoded borrower verification status (e.g., verified income). |  
| *WARN* | Not Used / Flag | Nominal | Placeholder for warnings during preprocessing (mostly NaNs). |  
| STD\_IMP\_REP\_annual\_inc | Predictor | Interval (Standardized) | Standardized and imputed annual income. |  
| STD\_IMP\_REP\_delinq\_2yrs | Predictor | Interval (Standardized) | Standardized and imputed count of delinquencies in past 2 years.|  
| STD\_IMP\_REP\_dti | Predictor | Interval (Standardized) | Standardized and imputed debt-to-income ratio. |  
| STD\_IMP\_REP\_emp\_length | Predictor | Interval (Standardized) | Standardized and imputed employment length. |  
| STD\_IMP\_REP\_int\_rate | Predictor | Interval (Standardized) | Standardized and imputed loan interest rate. |  
| STD\_IMP\_REP\_loan\_amnt | Predictor | Interval (Standardized) | Standardized and imputed loan amount. |  
| STD\_IMP\_REP\_longest\_credit\_lengt | Predictor | Interval (Standardized) | Standardized and imputed length of longest credit line. |  
| STD\_IMP\_REP\_revol\_util | Predictor | Interval (Standardized) | Standardized and imputed revolving credit utilization rate.|  
| STD\_IMP\_REP\_term\_length | Predictor | Interval (Standardized) | Standardized and imputed term length of the loan. |  
| STD\_IMP\_REP\_total\_acc | Predictor | Interval (Standardized) | Standardized and imputed total number of credit lines/accounts. |  

* **Source of training data**: [Loan_Clean.csv Trainning Datasets](https://github.com/jphall663/GWU_data_mining/blob/master/03_regression/data/loan_clean.csv)
* **How training data was divided into training and validation data**: In GLM-based Regression, the data was not explicitly split; however we tested out trained GLM model to generate a prediction for a new customer [GLM Test Data](https://github.com/nmemranhussain/6290_PAI_1/blob/main/GLM_test_data.jpg). For ANN, the datasets is divided into 40% training (65,595 rows), 30% validation (49,196 rows) and 30% test (49,196 rows). Like GLM, we tested our trained ANN model to generate a prediction for a new customer [ANN_test_data](https://github.com/nmemranhussain/6290_PAI_1/blob/main/ANN_test_data.jpg)
* **Total number of rows and columns in the dataset**: The dataset contains 163,987 rows and 18 columns. In GLM
* **Any differences in columns between training and test data**: Yes, we used different dataset to generate a prediction for a new customer.

### Model details
* **Columns used as inputs in the final model**: 'GRP_REP_home_ownership', 'GRP_addr_state', 'GRP_purpose', 'GRP_verification_status', 'STD_IMP_REP_annual_inc', 'STD_IMP_REP_delinq_2yrs', 'STD_IMP_REP_dti', 'STD_IMP_REP_emp_length', 'STD_IMP_REP_int_rate', 'STD_IMP_REP_loan_amnt', 'STD_IMP_REP_longest_credit_lengt', 'STD_IMP_REP_revol_util', 'STD_IMP_REP_term_length' and 'STD_IMP_REP_total_acc'
* **Column(s) used as target(s) in the final model**: 'bad_loan'
* **Type of model**: Generalized Linear Model (GLM) and Artificial Neural Network (ANN) -based Regression model
* **Software used to implement the model and theors version:** H20 3.46.0.7 version, Python version: 3.11.13, Pandas version: 2.2.2, NumPy version: 2.0.2 and Matplotlib version: 3.10.0
* **Hyperparameters or other settings of the model**: For [Assignment_1](https://github.com/nmemranhussain/6290_PAI_1/blob/main/Assignment_1_final.ipynb)  
| Hyperparameter | Value(s) | Purpose |  
| --------------- | --------- | ------- |  
| `alpha` | `[0.01, 0.25, 0.5, 0.99]` | Mix of L1/L2 regularization |  
| `lambda_search` | `True` | Enables automatic lambda tuning |  
| `family` | `"binomial"` | Specifies binary logistic regression |  
| `seed` | `309` | Ensures reproducible results |



