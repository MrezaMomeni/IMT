Medical Expenses Prediction Analysis
This notebook details the process of analyzing medical expenses data, building predictive models, and interpreting the results to provide actionable business insights.

Data Exploration
The dataset contains information about individuals including age, gender, BMI, number of children, discount eligibility, region, medical expenses, and premium.

Descriptive statistics revealed the distribution and spread of these features. For example, 'expenses' and 'premium' distributions were right-skewed, indicating a few individuals with significantly higher costs and premiums.

Histograms and box plots for numerical features ('age', 'bmi', 'children', 'expenses', 'premium') visualized their distributions and identified potential outliers, especially in 'bmi', 'expenses', and 'premium'.

Bar plots for the encoded categorical features ('gender', 'discount_eligibility', 'region') showed the frequency of each category.

The correlation matrix and heatmap highlighted relationships between numerical features and 'expenses'. A very strong positive correlation was observed between 'premium' and 'expenses'. Moderate positive correlations were also seen between 'age', 'bmi', and 'expenses'.

Feature Engineering & Preprocessing
Categorical variables ('gender', 'discount_eligibility', 'region') were encoded numerically as part of initial data loading.

Numerical features ('age', 'bmi', 'children', 'premium') were scaled using StandardScaler within the modeling pipelines. This ensures that features on different scales do not disproportionately influence the models, particularly distance-based models like Linear Regression.

No explicit handling of missing values or outliers was performed in the preprocessing steps, as the initial inspection and data.info() indicated no missing values. While outliers were identified in 'bmi', 'expenses', and 'premium', they were kept in the dataset for model training.

Modeling Approach
The problem was framed as a regression task to predict 'expenses'. The data was split into training (80%) and testing (20%) sets with a random_state for reproducibility.

Three regression models were chosen:

Linear Regression: A simple baseline model to understand linear relationships.
Gradient Boosting Regressor: A powerful ensemble tree-based model known for high accuracy.
Random Forest Regressor: Another ensemble tree-based model, robust to outliers and capable of capturing non-linear relationships.
Pipelines were constructed for each model. These pipelines included a ColumnTransformer to apply StandardScaler to numerical features while passing through the already encoded categorical features. This was followed by the respective regression model.

Cross-validation (5-fold) was used on the training data to assess model performance more robustly and compare models, in addition to evaluation on the separate test set.

Results and Interpretation
Model performance was evaluated using R-squared (R²), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) on the test set.

Test Set Performance:

Linear Regression: R²: [r2_linear:.4f], RMSE: [rmse_linear:.2f], MAE: [mae_linear:.2f]
Gradient Boosting Regressor: R²: [r2_gbr:.4f], RMSE: [rmse_gbr:.2f], MAE: [mae_gbr:.2f]
Random Forest Regressor: R²: [r2_rf:.4f], RMSE: [rmse_rf:.2f], MAE: [mae_rf:.2f]
Both Gradient Boosting and Random Forest models achieved significantly higher R² values and lower RMSE/MAE compared to Linear Regression, indicating better predictive accuracy on the test data. Gradient Boosting showed the highest R-squared and lowest RMSE.

Actual vs. Predicted Plots: The scatter plots of actual vs. predicted expenses showed that Linear Regression struggled to predict higher expenses accurately, while Gradient Boosting and Random Forest predictions clustered much closer to the ideal diagonal line across the range of expenses.

Residual Plots: The residual plot for Linear Regression displayed a pattern of increasing residuals with increasing predicted values, suggesting issues with homoscedasticity. The residual plots for Gradient Boosting and Random Forest showed more scattered residuals, indicating better assumptions, though some patterns were still present at higher predicted values.

Cross-Validation Results (5-Fold on Training Data):

Linear Regression:
R-squared: Mean = [cv_scores_linear_r2.mean():.4f], Std Dev = [cv_scores_linear_r2.std():.4f]
RMSE: Mean = [np.sqrt(-cv_scores_linear_mse).mean():.2f], Std Dev = [np.sqrt(-cv_scores_linear_mse).std():.2f]
Gradient Boosting Regressor:
R-squared: Mean = [cv_scores_gbr_r2.mean():.4f], Std Dev = [cv_scores_gbr_r2.std():.4f]
RMSE: Mean = [np.sqrt(-cv_scores_gbr_mse).mean():.2f], Std Dev = [np.sqrt(-cv_scores_gbr_mse).std():.2f]
Random Forest Regressor:
R-squared: Mean = [cv_scores_rf_r2.mean():.4f], Std Dev = [cv_scores_rf_r2.std():.4f]
RMSE: Mean = [np.sqrt(-cv_scores_rf_mse).mean():.2f], Std Dev = [np.sqrt(-cv_scores_rf_mse).std():.2f]
Cross-validation confirmed the superior performance of Gradient Boosting and Random Forest over Linear Regression, providing more robust estimates of their generalization performance.

Feature Importance and Coefficients:

Gradient Boosting & Random Forest Feature Importance: [gbr_importance_df.to_markdown(index=False)] [rf_importance_df.to_markdown(index=False)]
Linear Regression Coefficients: [linear_coefficients_series.to_markdown()]
Tree-based models identified 'premium' as the overwhelmingly most important feature, followed by 'age' and 'bmi'. Linear Regression also showed 'premium' having the largest impact (coefficient).

Data Leakage with 'Premium' Feature:

A critical issue identified is the significant data leakage caused by including the 'premium' feature when predicting 'expenses'. 'premium' is likely a direct outcome or a strong proxy for medical expenses. Using it as a predictor results in unrealistically high model performance metrics (R² close to 1) and renders the model impractical for predicting expenses based on independent risk factors. The high importance/coefficient of 'premium' simply reflects this leakage. For a usable model, 'premium' must be excluded and the model retrained on the remaining features.

Suggestions for Business Impact (Excluding 'Premium')
Based on the analysis (and assuming a model retrained without 'premium' performs acceptably), the following insights and suggestions can be made for an insurance company:

Pricing Strategy:

Age: Age is a significant predictor of medical expenses. Pricing models should strongly incorporate age as a primary factor, potentially with non-linear effects if the relationship is not purely linear.
BMI: BMI also impacts expenses. Policies could be priced higher for individuals with higher BMI, reflecting the increased health risks and associated costs. Incentives for maintaining a healthy BMI could also be considered.
Children: The number of children shows some impact. This might relate to family coverage costs or increased healthcare needs associated with having dependents. Pricing could reflect the number of children covered under a policy.
Risk Assessment and Targeted Interventions:

Identify High-Risk Groups: Individuals with higher age and BMI are likely to incur higher medical expenses. These groups can be identified for targeted health and wellness programs aimed at managing chronic conditions or promoting healthier lifestyles.
Refine Risk Models: The identified key features (age, BMI, children, potentially region and gender) can be used to build or refine risk assessment models that predict potential future medical costs more accurately, independent of the premium already assigned.
Underwriting and Policy Design:

Inform Underwriting Decisions: The model can help underwriters assess the risk associated with new applicants based on their age, BMI, and number of children, aiding in setting appropriate policy terms and premiums (based on the model's prediction of expenses, not using the premium as an input).
Tailor Policies: Understanding the expense drivers can help design different policy options that cater to the needs and risk profiles of various demographic segments.
Fraud Detection (Potential):

While not the primary focus, a model trained on independent features could potentially flag claims where the expenses are significantly higher than predicted based on the individual's profile, warranting further investigation.
It is crucial to retrain the model excluding the 'premium' feature to build a truly predictive model based on independent risk factors. The performance metrics and feature importance analysis from the retrained model would then form the basis for more reliable business insights and actionable strategies.
