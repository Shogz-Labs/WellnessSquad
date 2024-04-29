import streamlit as st 
import joblib
import time
from utils import predict_ideation, scrape_reddit, scrape_quora, scrape_youtube_transcript

st.set_page_config(
    page_title = 'WellnessSquad',
    page_icon= '🩺',
    layout='wide'
)

st.header('WellnessSquad: An Augmented Approach to Suicide Ideation Classification')

st.markdown(
    """
    *WellnessSquad* is an updated version of [WellnessOracle](https://github.com/stoyonaga/EECS5327_WellnessOracle), a project developed in GS/EECS 5327, Introduction to Machine Learning & Pattern Recognition by 
    Shogo Toyonaga, Abel Habte, Dongwon Lee, and Jonathan Ramos.
    """
)

models = {
    'Random Forest (rf)' : r'Models\rf_base_model.pkl',
    'Logistic Regression (lr)' : r'Models\lr_base_model.pkl',
    'Gradient Boosting Classifier (gbc)' : r'Models\gbc_base_model.pkl',
    'Voting Classifier (vc)' : r'Models\vc_base_model.pkl'
}

model = st.radio(
    label = 'Please select a model to run inference on:',
    options = models.keys(),
    index = None
)


if model != None:
    with st.spinner('Loading... please wait....'):
        model = joblib.load(models[model])
        time.sleep(2)
        tab1, tab2, tab3, tab4 = st.tabs(['Inference Sandbox', 'Reddit', 'Quora', 'YouTube'])
        with tab1:
            st.header('Inference Sandbox')
            message = st.text_input('Please enter the text that you would like to classify below.')
            if message != '':
                st.text_area(
                    label = 'The text has been classified as follows:',
                    value = predict_ideation(model, [message], False),
                    disabled = True
                )
        with tab2:
            st.header('Reddit Scraper')
            reddit_name = st.text_input('Reddit Name')
            category = st.selectbox(
                'Post Category',
                ('best', 'hot', 'new', 'top', 'rising'),
                index = None
            )
            if reddit_name != '' and category != None:
                reddit_classification = scrape_reddit(model, reddit_name, category)
                st.text(reddit_classification)
        with tab3:
            st.header('Quora Scraper')
            link = st.text_input('Quora Link')

            if link != '':
                quora_classification = scrape_quora(model, link, 20, 50, 5)
                st.text(quora_classification)
        with tab4:
            st.header('YouTube Transcript Scraper')
            link = st.text_input('YouTube URL:')
            if link != '':
                st.text(scrape_youtube_transcript(model, link))


