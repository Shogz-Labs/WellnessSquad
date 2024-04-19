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
4. Models 
    - Optuna
    - VotingClassifier 

## Data Analytics
I have provided some preliminary data analytics and visualizations from our training dataset. You may also see them in our dataloader.ipynb file. If you have any questions, please feel free to reach out at any time! :smile: 
## Pipeline 

The new pipeline can be visualized below:
![](images/pipeline.png)

## Hyperparameters
Optimal Random Forest Hyperparameters
```
```

Optimal Logistic Regression Hyperparameters
```
```

Optimal Gradient Boosting Hyperparameters
```
```

Optimal Voting Classifier Hyperparameters
```
```

## Results 
A comparison of the updated models can be seen below. The results are only with relation to the testing set.
As the project progresses, more tables and visualizations will be provided below.

| Model  | Original Model Accuracy (%) | New (Base) Accuracy (%) | New (Optimal) Accuracy (%)|
| ------------- | ------------- | -------------| ------------- |
| Random Forest  | 87.0%  | 89.1% (+1%) | ? |
| Logistic Regression | 95%  | 94.0% (-1%) | ? | 
| Gradient Boosting Classifier | N/A | 89.3%| ? | 
| VotingClassifier | N/A | TBD | ? |

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