#!/usr/bin/env python
# coding: utf-8

# ## Импорт библиотек

# In[1]:


import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer 


# In[2]:


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# ## Определим необходимые части анализатора текстов

# In[3]:


from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,   
    Doc,
    NamesExtractor,
    NewsNERTagger, 
    PER
)


# In[4]:


doc = Doc("Ангелина Иванова")


# In[5]:


segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
morph_vocab = MorphVocab()

names_extractor = NamesExtractor(morph_vocab)


# In[6]:


doc.segment(segmenter)
doc.tag_morph(morph_tagger)
doc.parse_syntax(syntax_parser)
doc.tag_ner(ner_tagger)
sent = doc.sents[0]
sent.morph.print()

names_extractor = NamesExtractor(morph_vocab)


# ## Загрузка данных

# In[7]:


data = pd.read_csv('test_data.csv')
#data['text'] = data['text'].str.title()
data['text'] = data['text'].str.replace(r'[^А-Яа-я ]', '', regex=True)
data.head()


# ## Для простоты поиска имен, скачаем их из репозитория ниже
# ## https://github.com/ktaranov/db-data/blob/master/PeopleNames/russian_names.csv

# In[8]:


names_df = pd.read_csv('russian_names.csv', sep=',')
names_df.head()


# ## Подсчет количества диалогов

# In[9]:


num_of_dialogs = data[data["line_n"] == 0].shape[0]
num_of_dialogs


# ## Определим, кто первый говорил, чтобы определить примерное количество строк диалога, за которые менеджер должен был представиться

# In[10]:


data[data["line_n"] == 0]


# ## Сохраним все реплики каждого клиента

# In[11]:


client_phrases = []
client_words = []
for i in range(num_of_dialogs):
    client_phrases.append(data[data["dlg_id"] == i][data["role"] == "client"].iloc[:, -1].values)
for (i, j) in zip(client_phrases, range(num_of_dialogs)):
    client_words.append(list(map(nltk.word_tokenize, client_phrases[j])))
    temp = []
    for k in client_words[-1]:
        temp += k
    client_words[-1] = set(temp)


# ## Сохраним все реплики каждого менеджера

# In[12]:


manager_phrases = []
manager_words = []
for i in range(num_of_dialogs):
    manager_phrases.append(data[data["dlg_id"] == i][data["role"] == "manager"].iloc[:, -1].values)
for (i, j) in zip(manager_phrases, range(num_of_dialogs)):
    manager_words.append(list(map(nltk.word_tokenize, manager_phrases[j])))
    temp = []
    for k in manager_words[-1]:
        temp += k
    manager_words[-1] = set(temp)


# # Пункт a
# ## Так как разговор официальный, скорее всего в диалоге будет использовано одно из следующих приветствий: "Доброе утро","Добрый день", "Добрый вечер", "Здравствуйте". 

# In[13]:


greetings = ["доброе утро", "добрый день", "добрый вечер", "здравствуйте"]
first_mask_for_f = [False] * len(manager_phrases)
greeting_phrase_index = [-1] * len(manager_phrases)
for i in range(len(manager_phrases)):
    counter = 0
    for phrase in manager_phrases[i]:
        for greeting in greetings:
            if phrase.lower().find(greeting) != -1: 
                print("Manager number = ", i, 'Greeting: "', phrase, '"')
                first_mask_for_f[i] = True
                greeting_phrase_index[i] = counter
        counter += 1


# ## Пятеро менеджеров поздоровались с клиентом

# # Пункт b
# ## Каждый раз клиенту первым отвечал менеджер. Предположим, что в таком случае, менеджер должен был представиться клиенту за первые 2 свои реплики

# ## Проверим, были ли названы какие-либо имена за первые 2 реплики менеджера

# In[14]:


for i in range(len(manager_phrases)):
    names = []
    for j in range(min(2, len(manager_phrases[i]))):
        phrase = manager_phrases[i][j]
        names = []
        doc = Doc(phrase)        
        matches = names_extractor(phrase)
        for match in matches:
            temp_name = match.fact.first
            if temp_name is not None and names_df["Name"].str.contains(temp_name.title()).sum() != 0 and len(temp_name) > 3:
                print(phrase)
                print("Manager number = ", i, "Manager Name = ", match.fact.first)


# ## Имен не обнаружено. Таким образом, согласно предположению, раз не обнаружено имен за первые 2 реплики менеджера, то он не представился клиенту

# In[15]:


from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
text = "Добрый день День добрый"
bigram_measures = nltk.collocations.BigramAssocMeasures()
tokens = nltk.wordpunct_tokenize(text)
finder = BigramCollocationFinder.from_words(tokens)
scored = finder.score_ngrams(bigram_measures.raw_freq)
sorted(bigram for bigram, score in scored)


# # Пункт c. Извлекать имя менеджера.

# ## Найдем все имена, которые назвывал клиент. Предположим, что если  клиент представился, то он использовал в фразе слово "зовут" или слово "это" рядом с именем. Так же предположим, что в диалоге не было названо других имен до того, как клиент обратился к менеджеру по имени.

# In[16]:


manager_names = []
for i in range(len(client_phrases)):
    counter = 0
    for phrase in client_phrases[i]:
        matches = names_extractor(phrase)
        #print([matches.])
        #spans = [_.span for _ in matches]
        facts = [_.fact.as_json for _ in matches]
        names_in_phrase = []
        for name in facts:
            #print([name.keys()])
            if 'first' in name.keys():
                if names_df["Name"].str.contains(name['first'].title()).sum() != 0 and len(name['first']) > 4:
                    names_in_phrase.append(name['first'])
                    print(phrase)
        temp_eto_index = phrase.find('это')
        if(len(names_in_phrase) > 0 and (phrase.find('зовут') != -1 or (temp_eto_index != -1 and ((temp_eto_index + 4) == phrase.find(names_in_phrase[0]) or temp_eto_index == phrase.find(names_in_phrase[0]) + len(names_in_phrase[0]) + 1)))):
            if(len(names_in_phrase) > 1):
                print("phrase_num = ", counter, " --- ", names_in_phrase[1:])
                manager_names.append(names_in_phrase[1:])
        elif(len(names_in_phrase) > 0):
            print("phrase_num = ", counter, " --- ", names_in_phrase[0])
            manager_names.append(names_in_phrase[0])
        counter += 1
manager_names


# ## В данном случае код сделал 2 ложных срабатывания: в 4-м и 5-м диалоге

# # Пункт d

# ## Как правило, когда человек говорит название своей компании, это звучит следующим образом: "Компания \название компании\ ", "Я из компании \название компании\ ", "Я представляю компанию". 

# ## Кроме того, что мы ожидаем услышать от клиента, мы так же предполагаем, что название компании, если клиент об этом говорил, поместится в двух словах
# ## Для того, чтобы сохранить именно название компании клиента, а не транспортной или любой другой компании, вводим переменную, которая меняет свое значение 1 раз, при первом нахождении компании

# In[17]:


companies = ["компания ", "я из компании", "я представляю компанию"]
find_mask = [False] * len(client_phrases)
companies_res = ["None"] * len(client_phrases)
for i in range(len(client_phrases)):
    for phrase in client_phrases[i]:
        for company in companies:
            curr_find = phrase.lower().find(company)
            if curr_find != -1 and find_mask[i] == False: 
                find_mask[i] = True
                temp = phrase[curr_find + len(company):].split()
                client_company = temp[0] + " " + temp[1]
                companies_res[i] = client_company
                print("Client number = ", i, 'company:', client_company)


# # Пункт e
# ## Так как разговор официальный, скорее всего в диалоге будет использовано одно из следующих прощаний: "До свидания", "Всего доброго", "Всего хорошего". 

# In[18]:


farewells = ["до свидания", "всего доброго", "всего хорошего"]
second_mask_for_f = [False] * len(manager_phrases)
farewell_phrase_index = [-1] * len(manager_phrases)
for i in range(len(manager_phrases)):
    counter = 0
    for phrase in manager_phrases[i]:
        for farewell in farewells:
            if phrase.lower().find(farewell) != -1: 
                print("Manager number = ", i, 'Farewell: "', phrase, '"')
                second_mask_for_f[i] = True
                farewell_phrase_index[i] = counter
        counter += 1


# ## Двое менеджеров попрощались с клиентом

# # Пункт f

# In[19]:


first_mask_for_f = np.array(first_mask_for_f)
second_mask_for_f = np.array(second_mask_for_f)
first_mask_for_f, second_mask_for_f


# In[20]:


task_f = np.array(first_mask_for_f) * np.array(second_mask_for_f)
task_f


# ## Один менеджер и поздоровался и попрощался с клиентом

# # Сохраним результаты решения каждого пункта

# In[21]:


farewell_phrase_index


# In[22]:


greeting_phrase_index


# In[23]:


for i in range(len(manager_phrases)):
    print('---')
    if(greeting_phrase_index[i] != -1):
        print(manager_phrases[i][greeting_phrase_index[i]])
    if(farewell_phrase_index[i] != -1):
        print(manager_phrases[i][farewell_phrase_index[i]])


# In[24]:


companies_res[3] = companies_res[3].split()[0]
companies_res


# In[25]:


manager_names


# In[26]:


manager_names[3] = "None"
manager_names[4] = "None"
manager_names


# In[27]:


results = pd.DataFrame(columns=['dlg_no', 'a', 'b', 'c', 'd', 'e', 'f'])
for i in range(len(manager_phrases)):
    #new_row = {'a':'', 'b':87, 'c':92, 'd':97, 'e':, 'f': }
    new_row = {}
    new_row['dlg_no'] = i
    new_row['b'] = "Менеджер не представился"
    if(greeting_phrase_index[i] != -1):
        #print(manager_phrases[i][greeting_phrase_index[i]])
        new_row['a'] = manager_phrases[i][greeting_phrase_index[i]]
    else:
        new_row['a'] = "Менеджер не поздоровался"
    if manager_names[i] != "None":
        new_row['c'] = manager_names[i]
    else:
        new_row['c'] = "Имя менеджера не называлось"
    if companies_res[i] != "None":
        new_row['d'] = companies_res[i]
    else:
        new_row['d'] = "Компания клиента не упоминалась"
    if(farewell_phrase_index[i] != -1):
        #print(manager_phrases[i][greeting_phrase_index[i]])
        new_row['e'] = manager_phrases[i][farewell_phrase_index[i]]
    else:
        new_row['e'] = "Менеджер не попрощался"
    if task_f[i] == True:
        new_row['f'] = "Менеджер поздоровался и попрощался"
    else:
        new_row['f'] = "Условие не выполнено"
        #results = results.append(['None'])
    results = pd.concat([results, pd.DataFrame.from_dict(new_row, orient = 'index').T], axis=0, ignore_index=True)


# In[28]:


results


# In[29]:


results.to_csv("parsing_results.csv", encoding='cp1251', index = False)


# In[ ]:




