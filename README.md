# WellnessSquad: An Augmented Approach to Suicide Ideation Classification

*WellnessSquad* is an updated version of [WellnessOracle](https://github.com/stoyonaga/EECS5327_WellnessOracle), a project developed in GS/EECS 5327, Introduction to Machine Learning & Pattern Recognition by Shogo Toyonaga, Abel Habte, Dongwon Lee, and Jonathan Ramos.

I have added some improvements by:

1. Training our base models on new handcrafted features.
    - clean_text
    - average_words_per_setence
    - Sentiment (roberta-base-sentiment)
    - num_emojis

2. Feature Scaling and Pre-Processing 
    - Imputers 
    - StandardScaler 
    - OneHotEncoder 
    - TfidVectorizer

3. Data Analytics & Visualization
    - Plotly Express 
    - Emojis

4. Models & Tuning
    - VotingClassifier
    - Gradient Boosting Classifier
    - Optuna 

## Data Analytics
I have provided some preliminary data analytics and visualizations from our training dataset. You may also see them in our [Data_Visualizations.ipynb](https://github.com/stoyonaga/WellnessSquad/blob/main/Notebooks/Dataset_Visualizations.ipynb). If you have any questions, please feel free to reach out at any time! :smile: 
## Pipeline 

The new pipeline can be visualized below:
![](images/pipeline.png)

## Hyperparameters
To obtain the optimal hyperparameters, I have added some new code that leverages optuna. Additionally, it offers visualizations to see which features are the most important to improving your models performance.

## Results 
A comparison of the updated models can be seen below. The results are only with relation to the testing set.
As the project progresses, more tables and visualizations will be provided below.

| Model  | Model Accuracy | 
| ------------- | ------------- | 
| [Random Forest]()  | 89.1% |
| [Logistic Regression]() | 94.0% | 
| [Gradient Boosting Classifier]() | 89.3% | 
| [VotingClassifier]() | 93.0 %|

I would have loved to provide the optimal hyperparameters from running a comprehensive optuna study, however, I do not currently have the available compute.
Nevertheless, I have provided the programming scripts to obtain them if you have a sufficiently strong CPU and an abundance of computing time :P 

## Data Visualizations
![](images/sentiment.png)
![](images/labels.png)

### Suicidal Messages
![](images/suicide_emojis.png)
![](images/suicide_wordcloud.png)
### Non-Suicidal Messages
![](images/control_emojis.png)
![](images/control_wordcloud.png)

## References
- [The Suicide and Depression Detection Dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/data)