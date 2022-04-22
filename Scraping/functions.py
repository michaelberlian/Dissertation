
#removing words or sentences on paragraph that appears in words
def remove_word(words, paragraph):
    for word in words :
        paragraph = paragraph.replace(word,' ')
    return paragraph

#removing punctuation and replace it with space
def remove_punctuation(punctuations,content):
    spaces = ' '*len(punctuations)
    #creating dictionary to replace characters
    table = content.maketrans(punctuations,spaces)
    return content.translate(table)
