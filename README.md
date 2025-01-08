# Final Project - Workforce wellness machine learning


## Project Overview

## Data Cleaning and Exploration

### Initial Data Cleaning
The dataset was sourced from Kaggle as a CSV file. The cleaning process began by removing null values and reducing unnecessary features:
- **Response Percentage Filtering:** Columns with less than 70% responses were excluded, leaving 48 features.
- **Unique Response Filtering:** Columns with more than 10 unique responses were further inspected. Long response fields, such as "Why or why not" questions, were removed.
- **Gender Column Normalization:** Responses in the gender column were standardized by grouping equivalent values. For instance, "Male" = "male" = "M". The gender data was then binned into three categories: "Male," "Female," and "Other."
- **Country Data Binning:** Participants were grouped based on country into three categories: "United States," "United Kingdom," and "Other," as most responses came from the U.S. or U.K. This adjustment reduced the dataset to 45 columns.
- **Significance Testing:** A chi-square test was conducted on each column to retain only significant variables, using a p-value threshold of 0.05. This process narrowed the dataset down to 32 columns.

---

## Database Integration
### Database Setup
The cleaned dataset was loaded into a SQLite database using Pandas and SQLAlchemy. Multiple tables were merged to facilitate interaction with the machine learning model. SQLite was chosen for the following reasons:
- **Efficiency:** SQLite offers fast read/write operations.
- **Simplicity:** It is easy to learn and requires no additional installation or configuration.
- **Tool Availability:** It integrates seamlessly with a variety of tools.

### Limitations
- SQLite databases are limited to 2 GB, which was sufficient for this project.

---

## Machine Learning Model - Random Forest

### Overview of Random Forest
Random forest is an ensemble learning algorithm comprising multiple decision trees. Each tree is trained on a random subset of the dataset. The final prediction is made by averaging the predictions from all trees, enhancing the model's accuracy and robustness.

### Advantages of Random Forest
- **Accuracy:** Random forest models typically outperform other algorithms in predictive tasks.
- **Robustness:** The algorithm is resistant to overfitting due to training on random subsets. It also handles outliers effectively by isolating extreme values in small leaves and averaging them.
- **Scalability:** It performs well with large datasets and numerous input variables without requiring variable elimination.
- **Feature Importance:** It provides a natural ranking of input variable importance, enabling better insights into significant features for prediction.

### Limitations
- **Interpretability:** Random forest models are harder to interpret compared to single decision trees.
- **Computational Cost:** Training numerous deep trees requires significant computational resources and memory.
- **Diminishing Returns:** Beyond a certain number of samples, performance gains taper off.

---

## Data Preprocessing

### OneHotEncoder
To process categorical data, "OneHotEncoder" from the Sklearn library was utilized. This technique transformed categorical responses into binary columns (1s and 0s), expanding each categorical feature into separate columns for each unique response.

### Fit and Transform
The encoder's `fit_transform()` method was applied to train the label encoder and convert categorical text data into numerical binary data. Scaling was unnecessary as all data was binary.

### Feature Names Extraction
To interpret the encoded data, the `get_feature_names()` method was used. This method facilitated merging the newly created OneHotEncoded features back into the dataset.

---

## Feature and Target Selection
The survey question most relevant to the research objective was identified:
- **Target:** "Have you been diagnosed with a mental health disorder - Yes."
- **Features:** Binary columns representing responses to the question above.

---

## Training and Testing Split
The data was split into training and testing subsets to validate the model. This separation ensured that the model's predictions were tested on unseen data.

---

## Random Forest Model Creation
A `RandomForestClassifier` was used with specific parameters:
- **`random_state:`** Ensured reproducibility of results.
- **`n_estimators:`** Defined the number of trees in the forest. Increasing this number generally enhances prediction stability, albeit at the cost of computational speed.

---

## Model Evaluation
Predictions were generated and evaluated using a confusion matrix to assess the model's performance. This matrix provided insights into the accuracy and reliability of the predictions.

## Analysis

#### What is the model's accuracy?
![Confusion Matrix](https://github.com/user-attachments/assets/7feb39cf-2cce-4f62-a483-184a2ebb7eca) <br>

The model achieved an accuracy score of 86.11%, indicating that it correctly predicts whether an individual has a mental health disorder 86.11% of the time based on their survey responses (assuming truthful answers). Since this model's purpose is exploratory—providing insights for individuals in tech regarding their potential likelihood of having (or developing) a mental health disorder or aiding tech companies in assessing the potential benefits of offering specific mental health services—an accuracy of 86.11% is considered adequate.<br>

In addition to accuracy, precision is also a critical metric for this question. Precision measures the reliability of positive classifications. The precision for true positives (correctly identifying individuals with mental health disorders) is 90%, and for true negatives (correctly identifying individuals without mental health disorders), it is 83%. This means that when the model predicts a participant has a mental health disorder (true positive), there is a 90% chance the prediction is correct. Similarly, if the model predicts a participant does not have a mental health disorder (true negative), there is an 83% likelihood the prediction is correct.

#### What statistics are involved and why?
Bootstrapping is a statistical technique that involves random sampling with replacement. In the context of random forests, bootstrapping is applied to individual decision trees, where some samples are used multiple times during training. The rationale behind this approach is that training each tree on slightly different samples reduces the overall variance of the forest without increasing its bias. This enhances the model's robustness and reliability.

## Dashboard
![DashboardSummaryData](https://github.com/user-attachments/assets/8ce33d3e-ade6-4d76-998a-db23464dd8fe)
