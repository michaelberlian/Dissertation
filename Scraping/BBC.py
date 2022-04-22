from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import numpy as np
import time 
from functions import remove_word, remove_punctuation

#preparing browser, CHANGE PATH TOU CHROMEDRIVER .exe
PATH = "/Users/michaelberlian/Desktop/UoN/Dissertation/code/chromedriver"
driver = webdriver.Chrome(PATH)

headlines = []
contents = []
categories = []
#number of news for each category scraped
numberOfNews = 100

#word that are not news content and repeatedly appear
repeatedWords = ["the bbc is not responsible for the content of external sites.","view original tweet on Twitter",
                "do you have a question about the covid restrictions in place in scotland?",
                "use the form below to send us your questions and we could be in touch.",
                "in some cases your question will be published, displaying your name, age and location as you provide it, unless you state otherwise.",
                "your contact details will never be published.", "please ensure you have read the terms and conditions.", 
                "if you are reading this page on the BBC News app, you will need to visit the mobile version of the BBC website to submit your question on this topic.",
                "do you have any questions about the cost of school uniform?",
                "how much do you spend?"]
topics = ['business','health','politics','technology','science_and_environment']
punctuations = '"#$%&\'()*+,-/:;<=>@[\\]^_`{|}~'

for topic in topics:

    #visiting website based on category
    driver.get("https://www.bbc.co.uk/news/"+topic)
    links = [] 
    
    #to simplify category name 
    if topic == 'science_and_environment':
        topic = 'science'

    #getting links of news from the pages (10)
    for i in range (10):

        searchs = driver.find_elements(By.CLASS_NAME,"lx-stream__post-container")
        for search in searchs:
            links.append(search.find_element_by_css_selector('a').get_attribute('href'))

        button = driver.find_element(By.CLASS_NAME,"qa-pagination-next-page")
        button.click()
        time.sleep(1)

    # print(topic,len(links))
    # print(links[:5])
    i = 0
    counter = 0 

    #vising the news links and scraping the content
    while (counter < numberOfNews and i < len(links)) :
        
        driver.get(links[i])
        i += 1

        try:
            # print(topic, links[i])
            headline = driver.find_element(By.CLASS_NAME,"ssrcss-gcq6xq-StyledHeading").text
            print(headline)

            texts = driver.find_elements(By.CLASS_NAME,"ssrcss-uf6wea-RichTextComponentWrapper")
            content = ""
            for text in texts:
                paragraph = text.text
                content = content + paragraph
                content = content + ' '
            
            # lower casing all the character and removing repeated words and punctuation
            content = content.lower()
            content = remove_word(repeatedWords,content)
            content = remove_punctuation(punctuations,content)

            # print(topic,links[i])
            headlines.append(headline.lower())
            contents.append(content.lower())
            categories.append(topic)

            counter += 1

        except Exception as e:
            # print(e)
            continue


# visiting the website with sport category, separated because different structure from the rest
driver.get("https://www.bbc.co.uk/sport")
links = []

# getting the news links
searchs = driver.find_elements(By.CLASS_NAME,"gs-c-promo-heading")
for search in searchs:
    links.append(search.get_attribute('href'))

# print('sport: ',len(links[i]))
# print(links)

# visit the news link and scraping the content
i = 0
counter = 0
while (counter < numberOfNews and i < len(links)) :

    driver.get(links[i])
    i += 1

    try:
        
        headline = driver.find_element(By.CLASS_NAME,"qa-story-headline").text

        texts = driver.find_elements_by_css_selector('div.qa-story-body > p') 
        content = ""
        for text in texts:
            paragraph = text.text
            content = content + paragraph
            content = content + ' '
        content = content.lower()
        content = remove_word(repeatedWords,content)
        content = remove_punctuation(punctuations,content)

        # print('sport',links[i])
        headlines.append(headline.lower())
        contents.append(content.lower())
        categories.append('sport')

        counter += 1

    except Exception as e:
        # print (e)
        continue

# closing the window
driver.quit()

#create dataframe from the scrapped data
column = np.arange(len(headlines))
df = pd.DataFrame([headlines,contents,categories],index=['headline','content','category'],columns=column)
df = df.transpose()

#combine the new data with previously collected data
try:
    df1 = pd.read_excel('../Data/Raw/BBC_news.xlsx')
    df = pd.concat([df1,df])
    df.drop_duplicates(subset=None, keep="first", inplace=True)
    df.dropna(inplace=True)
    df.to_excel('../Data/Raw/BBC_news.xlsx', sheet_name='news',index=False)
except Exception as e:
    print (e)
    df.to_excel('../Data/Raw/BBC_news.xlsx', sheet_name='news',index=False)

