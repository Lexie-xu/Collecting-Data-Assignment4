# Donald Trump's speech
### 1. Description of the corpus
The speeches are from https://millercenter.org/the-presidency/presidential-speeches?field_president_target_id[8396]=8396. It contains all the public speeches from the U.S. presidents. Here we only collect the speeches of Donald Trump from October 27, 2019, to January 19, 2021.
### 2. Target audience and intended use of the corpus
This corpus can be used to analyze the political interests of Donald Trump. From his speeches on different themes, we can know his tendency in politics. Such as, on which topic, he gave more speeches but on which one he didn't spend too much time on it. This corpus might be useful to those who are interested in the U.S. parties and policy. To understand where some of the problems in the U.S. come from.
### 3. Text selection criteria
The data was scraped from https://millercenter.org/the-presidency/presidential-speeches. The reason I chose this data is because it is publicly available. The sources are clean enough and don't need a complex process.
### 4. The data collection process
I use BeautifulSoup to scrape data from the website. 
```
url = 'https://millercenter.org/the-presidency/presidential-speeches?field_president_target_id[8396]=8396'
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html')
```
```
speeches = []
transcripts = soup.findAll("div", attrs={"class" : "views-field-title"})
```
Save all of the data as a CSV file.
```
csv_file_path = 'speech_data.csv'
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Title', 'Link', 'President', 'Date', 'Summary', 'Speech'])

    for i, transcript in enumerate(transcripts):
      if i < 24:
        link_url = transcript.find('a')['href']
        link_response = requests.get(link_url)
        link_html_content = link_response.content
        link_soup = BeautifulSoup(link_html_content, 'html.parser')
        link_text = link_soup.find("div", attrs = {'class': "transcript-inner"}).text.strip()
        speeches.append(link_text)
        title = transcript.text.strip()
        link = link_url
        president = link_soup.find("p", attrs = {'class': "president-name"}).text.strip()
        date = link_soup.find("p", attrs = {'class': "episode-date"}).text.strip()
        summary = link_soup.find("div", attrs = {'class': "about-sidebar--intro"}).text.strip()

        csv_writer.writerow([title, link, president, date, summary, link_text])
```
Cleaning and processing the data
Rename the columns and change the format of the date
```
data = pd.read_csv('speech_data.csv')
data = data.rename(columns={'Title': 'Speech Title', 'Link': 'URL', 'Speech': 'Transcript'})
```
### 5. Annotations and tools
I use spaCy in colab to analyze and process the data.
Load nlp pipeline and call the nlp model on the text.
```
nlp = spacy.load('en_core_web_sm')
doc = nlp(link_text)
```
#### Tokenization
Use pandas and spaCy to tokenize the data. Segment strings into individual words and punctuation markers.
```
def get_token(doc):
    return [(token.text) for token in doc]
```
```
data['Tokens'] = data['Doc'].apply(get_token)
data.head()
```
#### Lemmatization
The retrieval of the root word for each word in the dictionary
We can find the counts of a specific word in one column through this step.
If we want to find how many times Trump mentioned "China" in previous speeches.
```
def get_lemma(doc):
    return [(token.lemma_) for token in doc]

data['Lemmas'] = data['Doc'].apply(get_lemma)
```
```
print(f'"China" appears in the text tokens column ' + str(data['Tokens'].apply(lambda x: x.count('write')).sum()) + 'times.')
print(f'"China" appears in the lemmas column ' + str(data['Lemmas'].apply(lambda x: x.count('write')).sum()) + 'times.')
```
#### Part of Speech Tagging
Predict the simple universal part-of-speech of each token in a text
```
def get_pos(doc):
    return [(token.pos_, token.tag_) for token in doc]

data['POS'] = data['Doc'].apply(get_pos)
```
We can also only get the proper nouns.
```
def extract_proper_nouns(doc):
    return [token.text for token in doc if token.pos_ == 'PROPN']

data['Proper_Nouns'] = data['Doc'].apply(extract_proper_nouns)
```
### 6. Saving all of the columns into a CSV file.
```
annotated_corpus_path = 'annotated_corpus.csv'
data.to_csv(annotated_corpus_path, index=False, encoding='utf-8')

print(f"Annotated corpus has been saved to {annotated_corpus_path}")
```
### Donald Trump's Speeches dataset
- Speech Title - Title of the speech
- URL - The source URL for the speech
- President - The President giving the speech
- Date - Date the speech was given
- Summary - An official summary of the speech
- Transcript - An official transcript of the speech
- Doc - Stores the original text and all of the linguistic annotations 
- POS - Part of Speech Tagging
- Proper_Nouns - All of the proper nouns 
- Tokens - Individual tokens
### Acknowledgements
This data was scraped from: https://millercenter.org/the-presidency/presidential-speeches

I want to acknowledge The Miller Center at the University of Virginia for making this data publicly available.
