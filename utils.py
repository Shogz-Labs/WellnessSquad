import joblib
import torch
import re
import string
import emoji
import requests
from collections import Counter
import statistics
from playwright.async_api import async_playwright, Playwright
import asyncio
import time
from bs4 import BeautifulSoup
import nest_asyncio
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
nest_asyncio.apply()
from tqdm.auto import tqdm
import pandas as pd
import plotly.express as px
tqdm.pandas()
from youtube_transcript_api import YouTubeTranscriptApi
dv = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Auxilary Methods which are required to make predictions on new data
Do not touch the code under this line, please!
"""

def clean_text(x: str) -> str:
  """
  Goals:
    1) Remove punctuation to make the TF-IDF dictionary more accurate
    2) Remove emojis, they are unnecessary here for training purposes. (We will use emoji data in another column :3)
  """
  clean_text = x.translate(str.maketrans('', '', string.punctuation))
  clean_text = emoji.replace_emoji(clean_text, replace='')
  return clean_text

def average_words_per_sentence(message: str):
  """
  Unfortunately, due to the limitation of a non-standard ASCII table for emojis, getting the
  average words per second is not 100% accurate in certain cases.

  Please modify the regular expression for, 'sentences' to match your need if you re-use this code. Thanks :)
  """
  # Split the string based on {., !, ?} characters
  # print(f'Base Message: {message}')
  sentences = re.split('[.!?\u263a-\U0001f645\n]+', message.strip().replace('\n', ''))
  avg_words = 0
  for entry in sentences:
    # print(f'Entry: {entry}')
    words = re.split('[,;: ]', entry.strip())
    # print(f'Words: {words}')
    avg_words += len(words)
  return round(avg_words / len(sentences), 2)

def num_emojis(x: str) -> int:
  return emoji.emoji_count(x)

sentiment = pipeline(
    task='sentiment-analysis',
    model= "cardiffnlp/twitter-roberta-base-sentiment-latest", #@param {type:"string"},
    tokenizer = "cardiffnlp/twitter-roberta-base-sentiment-latest", #@param {type:"string"}
    max_length=512,
    batch_size = 25000,
    truncation=True,
    padding=True,
    device=dv
)

"""
Auxilary Methods which are required to make predictions on new data
Do not touch the code above this line, please!
"""

"""
Streamlit Functions (Feel free to touch and add more as you see fit :)
"""
def predict_ideation(model, messages, generate_visuals):
  output = ''
  container = {
      'clean_text': [],
      'average_words_per_sentence': [],
      'sentiment': [],
      'num_emojis': []
      }
  for message in messages:
    container['clean_text'].append(clean_text(message))
    container['average_words_per_sentence'].append(average_words_per_sentence(message))
    container['sentiment'].append(sentiment(message)[0]['label'])
    container['num_emojis'].append(num_emojis(message))

  container = pd.DataFrame(container)

  predictions = model.predict(container),
  class_probabilities = model.predict_proba(container)

  for i in range(len(messages)):
    output += f"\nMessage {messages[i]} has been labeled as: {[predictions[0][i]]}\n"
    output += f"\t Non-Suicide Approximation (%): {class_probabilities[i][0]}\n \t Suicide Approximation (%): {class_probabilities[i][1]}\n"

  labels = Counter(predictions[0])
  fig = px.pie(
      names = labels.keys(),
      values = labels.values()
      )
  fig.update_layout(
  title="Suicide Classification of Evaluated Messages",
  font_family="Courier New",
  title_font_family="Courier New",
  font_size=24
  )

  if generate_visuals:
    return output, fig
  else:
    return output

"""
This is the original Quora scraper that was developed in WellnessOracle with very minimal modifications
"""
def scrape_quora(model, link, number_of_long_posts, min_post_length_in_chars, loading_time):
  async def run(playwright: Playwright):
      chromium = playwright.chromium # or "firefox" or "webkit".
      browser = await chromium.launch()
      page = await browser.new_page()

      await page.goto(link)
      print("Sleeping for {} seconds (page loading)".format(loading_time))
      time.sleep(loading_time)
      for i in range(number_of_long_posts):
        try:
          button = await page.get_by_text("Continue Reading", exact=True).nth(i).click()
          print("Post #{} has been expanded...".format(i + 1))
        except:
          print("Not enough post data!! We scraped what was available to us.")
          break
        await page.mouse.wheel(0,125)
      html = page.inner_html("#mainContent")
      parser = BeautifulSoup(await html, "html.parser")
      posts = parser.find_all("div", {"class": "q-box spacing_log_answer_content puppeteer_test_answer_content"})
      answers = []
      for post in posts:
        if len(post.text.strip()) > min_post_length_in_chars:
          answers.append(post.text.strip())
      print("After pruning, {} posts (text-only) remain.".format(len(answers)))
      print(predict_ideation(model, answers, False))
      await browser.close()

  async def main():
      async with async_playwright() as playwright:
          await run(playwright)
  asyncio.run(main())


def scrape_reddit(model, subreddit_id, filter_classifier):
  url = "https://www.reddit.com/r/{}/{}".format(subreddit_id, filter_classifier)
  print("Requesting information (json file) from {}...".format(url))
  headers = {
      'User-Agent': 'shogz-bot'
  }

  response = requests.get(url + ".json", headers=headers)
  if response.ok:
    data = response.json()['data']
    reddit_title = []
    reddit_text = []
    for post in data['children']:
      reddit_title.append(post['data']['title'])
      reddit_text.append(post['data']['selftext'])
    print("Number of scraped posted: {}".format(len(reddit_title)))
    for i in range(len(reddit_text)):
      print("---------- Analysis of Comment {} ----------".format(i + 1))
      # print(predict_ideation(model, reddit_title, False))
      print(predict_ideation(model, [reddit_text[i]], False))
  else:
    print('Error {}'.format(response.status_code))

def scrape_youtube_transcript(model, url: str):
  text = ''
  for entry in YouTubeTranscriptApi.get_transcript(url[32:]):
    text += entry['text'].strip() + ' '
  text = re.sub(r'[^a-zA-Z ]+', '', ''.join(text.splitlines()))
  return predict_ideation(model, [text], False)



