from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import numpy as np
from functions import remove_word, remove_punctuation

#preparing browser, CHANGE PATH TOU CHROMEDRIVER .exe
PATH = "/Users/michaelberlian/Desktop/UoN/Dissertation/code/chromedriver"
driver = webdriver.Chrome(PATH)

headlines = []
contents = []
categories = []
#number of news for each category scraped
numberOfNews = 300

#word that are not news content and repeatedly appear
repeatedWords = ["(CNN Business)","(CNN)","(Reuters)",
                "A version of this story appeared in CNN's What Matters newsletter.", "To get it in your inbox, sign up for free here.",
                "Sign up for CNN's Wonder Theory science newsletter. Explore the universe with news on fascinating discoveries, scientific advancements and more.",
                "A version of this story first appeared in CNN Business' Before the Bell newsletter.","Not a subscriber? You can sign up right here.",
                "You can listen to an audio version of the newsletter by clicking the same link.", "A version of this story appeared in Wonder Theory newsletter","by CNN Space and Science writer Ashley Strickland, who finds wonder in planets beyond our solar system and discoveries from the ancient world.",
                "To get it in your inbox, sign up for free here.", "Sign up for CNN's Stress, But Less newsletter.",
                "Subscribe to CNN's Fitness, But Better newsletter", "Sign up for our newsletter series to ease into a healthy routine, backed by experts",
                "Sign up for CNN's Eat, But Better: Mediterranean Style. Our eight-part guide shows you a delicious expert-backed eating lifestyle that will boost your health for life.",
                "A version of this story appeared in CNN's Race Deconstructed newsletter", "A version of this story appeared in CNN's Wonder Theory newsletter",
                "Like what you've read? Oh, but there's more.", "Sign up here to receive in your inbox the next edition of Wonder Theory", ", brought to you",]
punctuations = '"#$%&\'()*+,-/:;<=>@[\\]^_`{|}~'
topics = ['health','politics','specials/space-science','business','business/tech']
category = ['health','politics','science','business','technology','sport']

for p in range (len(topics)):

    #visiting website based on category
    driver.get("https://edition.cnn.com/"+topics[p])
    links = [] 

    #getting the news link
    searchs = driver.find_elements(By.CLASS_NAME,"cd__headline")
    for search in searchs:
        links.append(search.find_element_by_css_selector('a').get_attribute('href'))

    i = 0
    counter = 0 

    #visiting the news link and scrape the content
    while (counter < numberOfNews and i < len(links)) :
        
        driver.get(links[i])
        i += 1

        try:
            headline = driver.find_element(By.CLASS_NAME,"pg-headline").text

            texts = driver.find_elements(By.CLASS_NAME,"zn-body__paragraph")
            content = ""
            for text in texts:
                paragraph = text.text
                content = content + paragraph
                content = content + ' '
            
            # lower casing all the character and removing repeated words and punctuation
            content = remove_word(repeatedWords,content)
            content = remove_punctuation(punctuations,content)

            headlines.append(headline.lower())
            contents.append(content.lower())
            categories.append(category[p])
            counter += 1 

        except Exception as e:
            # print(e)
            continue

#visiting the website with sport category, separated because different structure from the rest
driver.get("https://edition.cnn.com/sport")
links = [] 

#getting the news link
searchs = driver.find_elements(By.CLASS_NAME,"container_lead-plus-headlines__link")
for search in searchs:
    links.append(search.get_attribute('href'))

i = 0
counter = 0 

#visiting the news link and scrape the content
while (counter < numberOfNews and i < len(links)) :
    
    driver.get(links[i])
    i += 1

    try:
        headline = driver.find_element(By.CLASS_NAME,"pg-headline").text

        texts = driver.find_elements(By.CLASS_NAME,"zn-body__paragraph")
        content = ""
        for text in texts:
            paragraph = text.text
            content = content + paragraph
            content = content + ' '
        

        # lower casing all the character and removing repeated words and punctuation
        content = remove_word(repeatedWords,content)
        content = remove_punctuation(punctuations,content)
        
        headlines.append(headline.lower())
        contents.append(content.lower())
        categories.append(category[5])
        counter += 1 

    except Exception as e:
        # print(e)
        continue

#closing the browser window
driver.quit()

#create dataframe based on scraped data
column = np.arange(len(headlines))
df = pd.DataFrame([headlines,contents,categories],index=['headline','content','category'],columns=column)
df = df.transpose()

#combine the new data with previously collected data
try:
    df1 = pd.read_excel('../Data/Raw/CNN_news.xlsx')
    df = pd.concat([df1,df])
    df.drop_duplicates(subset=None, keep="first", inplace=True)
    df.dropna(inplace=True)
    df.to_excel('../Data/Raw/CNN_news.xlsx', sheet_name='news',index=False)
except :
    df.to_excel('../Data/Raw/CNN_news.xlsx', sheet_name='news',index=False)


