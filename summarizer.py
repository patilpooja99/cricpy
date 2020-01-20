from urllib import request
from bs4 import BeautifulSoup as bs
import re
import nltk
import heapq




url = "https://en.wikipedia.org/wiki/Artificial_Intelligence"
allParagraphContent = ""
htmlDoc = request.urlopen(url)
soupObject = bs(htmlDoc,'html.parser')
paragraphContents = soupObject.findAll('p')
#print(paragraphContents)

for paragraphContent in paragraphContents:
    allParagraphContent += paragraphContent.text
    #print(paragraphContent)

allParagraphContent_cleanerData = re.sub(r'\[[0-9]*\]',' ', allParagraphContent)
allParagraphContent_cleanedData = re.sub(r'\s+',' ', allParagraphContent_cleanerData)

#print(allParagraphContent_cleanedData)

#allParagraphContent_cleanedData = re.sub(r'[^a-zA-Z]',' ', allParagraphContent_cleanedData)
#allParagraphContent_cleanedData = re.sub(r'\s+',' ', allParagraphContent_cleanedData)

##### creating Sentence Tokens
sentences_tokens = nltk.sent_tokenize(allParagraphContent_cleanedData)
words_tokens     =nltk.word_tokenize(allParagraphContent_cleanedData)

##### calculate the frequency
stopwords= nltk.corpus.stopwords.words('english')
word_frequencies = {}

for word in words_tokens:
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1
#print(word_frequencies)
##### calculate weighted frequency
maximum_frequency_word = max(word_frequencies.values())
for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word]/maximum_frequency_word)
#print(word_frequencies)

#####calculate sentence score with each word weighted frequency
sentences_scores = {}
for sentence in sentences_tokens:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word_frequencies.keys():
            if len(sentence.split(' ')) < 30:
               if sentence not in sentences_scores.keys():
                  sentences_scores[sentence] = word_frequencies[word]
               else:
                   sentences_scores[sentence] += word_frequencies[word]
#print(sentences_scores)
                   
summary_artificialIntelligence = heapq.nlargest(5,sentences_scores, key=sentences_scores.get)
print(summary_artificialIntelligence)

