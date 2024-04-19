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

## Pipeline 

The new pipeline can be visualized below:
![](images/pipeline.png)


##

## References
- [The Suicide and Depression Detection Dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/data)