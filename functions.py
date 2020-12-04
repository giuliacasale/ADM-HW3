import numpy as np
import pandas as pd
import math

import nltk
import itertools
import os

from collections import Counter
from collections import defaultdict
from collections import OrderedDict
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

import seaborn as sns
import matplotlib.pyplot as plt




'''Function needed to read all the books stored in separate .tsv files and combine them in a single dataframe that will be used in the search engine'''

## This function is used only once to create the dataframe that will be then stored in a csv file to ease the access


def create_dataframe():
    
    df = pd.read_csv('/Users/Giulia/Downloads/tsv/book_0.tsv',sep='\t',index_col=False)
    
    books_to_skip = [192,2464,3011,3056,10141,11338,19728,20585,25426,25453,25902,26045,28476,29403]
    
    for i in range(1,30000):
        if os.path.getsize(f'/Users/Giulia/Downloads/tsv/book_{i}.tsv') != 0 and i not in books_to_skip:
            x = pd.read_csv(f'/Users/Giulia/Downloads/tsv/book_{i}.tsv',sep='\t',index_col=False)
            df = df.append(x)
   
    df.reset_index(drop=True, inplace=True)

    return df



    

'''Function that preprocess text by removing punctuation, double spaces, stopwords and  by applying Stemming'''

def pre_processing_data(text):           
    
    stop_words = set(stopwords.words('english'))
    text = (text.lower()).replace('\\n',' ')
    punctuation = RegexpTokenizer(r'\w+')               #identifies punctuation
    tokens = punctuation.tokenize(text)               #create a list of all words

    ps = PorterStemmer()

    filtered_text = []
    for word in tokens:
        if word not in stop_words:
            filtered_text.append(ps.stem(word)) 
    
    return filtered_text


'''Function that first creates a list of unique words that can be found from the union of ALL the unique words of each book, and then creates a dictionary that maps each word to an integer'''

def build_dictionary(df):                            

    vocabulary_list = []
    vocabs_in_books = defaultdict(list)

    for i in range(len(df)):
        x = df.loc[i]
        plot = x['plot']
        plot_filtered = pre_processing_data(plot)

        vocabs_book_i=set()
        for word in plot_filtered:
            vocabs_book_i.add(word)

        vocabulary_list.append(vocabs_book_i)
        vocabs_in_books[i]=list(vocabs_book_i)

    bag_of_words = set.union(*vocabulary_list)
    
    dictionary = {}
    for num, word in enumerate(bag_of_words):
        dictionary[word] = num
    
    return dictionary



#################################################### CONJUNCTIVE QUERY ########################################################


''' Function that for each document, calculates the frequency of the words in the document '''

def frequency_of_words_per_book(df,dictionary):      
    
    frequency_of_word = []

    #each dictionary has for keys all the words in the dictionary file we created above
    df1 = df['plot']

    for i in range(len(df)):

        plot_filtered = pre_processing_data(df1.loc[i])
        frequency_of_words = Counter(plot_filtered)

        frequency_of_word.append(frequency_of_words)
    
    return frequency_of_word


''' Function used to create the first inverted index: for each token (unique word in file vocabulary) associates a list of the documents that contain that token at leas once'''

def inverted_index1(df,dictionary,frequency_of_words):  
    
    inverted_index1 = dict.fromkeys(dictionary.values(),[])
    for key, value in dictionary.items():
        elem = []
        for i in range(len(df)):
            if frequency_of_words[i][key] > 0:
                elem.append(f'document_{i}')
        inverted_index1[value] = elem 
    
    return inverted_index1
    

''' function that processes the query given in input by the user '''
def query_processed(query):
    q = pre_processing_data(query)
    return q    

    

''' function that from the query, looks through the inverted index at each document that contains ALL the words in the query and adds that specific document to a new dataframe that will be shown in output to the user'''

def search_engine1(query,df,inverted_index_1,dictionary):   
    
    output = pd.DataFrame(columns = df.columns.tolist())
    
    for i in range(len(df)):
        count = 0
        for word in query:
            if f'document_{i}' in inverted_index_1[dictionary[word]]:
                count += 1
        
        if count == len(query):
            output = output.append(df.loc[i],ignore_index=True)
    
    return output

    

############################################# CONJUNCTIVE QUERY & RANKING SCORE ##################################################

'''function that calculates the term frequency in each document and creates a list of dictionaries where each dictionary refers to a single document. It has for keys the words in the filtered plot of it and for values the tf_score = frequency term/total number of words in that plot'''

def tf(df,dictionary,frequency_of_word):      
 
    tfs = []

    for i in range(len(df)):
        x = df.loc[i]
        plot = x['plot']
        plot_filtered = pre_processing_data(plot)

        #dictionary where the keys are all the words from plot_filtered
        tf = dict.fromkeys(plot_filtered,0)
        tot_number_of_words = len(plot_filtered)

        for key,item in frequency_of_word[i].items():
            frequency = item
            tf_score = frequency / tot_number_of_words
            tf[key] = tf_score    
        tfs.append(tf)
        
    return tfs


'''function that calculates the idf score of each token. The output is a dictionary where the keys are the all the different words stored in the vocabulary created above and the values are given by log(N/n):
- N = total number of documents in the original dataframe
- n = total number of documents in which each token appears'''

def idf(df,dictionary,frequency_of_word):    
    
    N = len(df)
    idf = dict.fromkeys(dictionary.keys(), 0)

    #calculate df by looking at the number of documents each token appears
    for i in range(len(df)):
        for key,item in frequency_of_word[i].items():
            if item > 0:
                idf[key] += 1
                
    #calculate idf 
    for key,item in idf.items():
        idf[key] = np.log(N/(item))
        
    return idf


'''function that puts together the tf and idf score for each document and for each token. The output is a list of dictionaries where each of them refers to a specific document. The keys are given by the keys in the tf_score dictionary created above and the values are given by multiplying the two scores (tf and idf)'''

def tf_idf_score(df,dictionary,tf_score,idf_score):    

    tf_idf_scores = []
    for i in range(len(df)):
        tf_idf_score = {}
        for key,item in tf_score[i].items():
            tf_idf_score[key] = round(item*idf_score[key],5) 
        tf_idf_scores.append(tf_idf_score)

    return tf_idf_scores


'''second inverted index where, for each token, we have a list of documents where the token appeared at least once, followed by their tf-idf scores calculated above'''

def inverted_index2(df,dictionary,tf_idf_scores,frequency_of_word):    
    
    inverted_idx = dict.fromkeys(dictionary.values(),[])
    
    for key,value in dictionary.items():
        elem = []
        for i in range(len(df)):
            if frequency_of_word[i][key] > 0:
                elem.append((f'document_{i}', tf_idf_scores[i][key]))
        inverted_idx[value] = elem   

    return inverted_idx


'''function that calculates the cosine similairty score for each document on respect of the query given in input by the user'''

def cosine_similarity(query,df,tf_idf_scores):
    
    scores = []
    for i in range(len(df)):
        d = {k: v**2 for k, v in tf_idf_scores[i].items()}
        norm_d = np.sqrt(sum(d.values()))
        somma = 0
        for token in query:
            if token in tf_idf_scores[i]: 
                if tf_idf_scores[i][token] > 0:
                    somma += tf_idf_scores[i][token]
        cosine_similarity_score = round(somma/norm_d,4)
        scores.append(cosine_similarity_score)
        
    return scores


'''this function first appends a new column to the original dataframe given by the cosine similarity scores calculate before, then creates a new dataframe where the rows are given by the documents that contain in the plot ALL the words given in input by the user. The rows are also sorted by their similarity score in descending order and only the top-10 are shown'''

def search_engine2(df,query,inverted_index_1,dictionary,scores):
    
    df1 = df[['bookTitle','plot','url']]
    df1['Similarity'] = scores
    
    output = pd.DataFrame(columns = df1.columns.tolist())

    for i in range(len(df)):
        count = 0
        for word in query:
            if f'document_{i}' in inverted_index_1[dictionary[word]]:
                count += 1

        if count == len(query):
            output = output.append(df1.loc[i],ignore_index=True)
    
    output = output.sort_values(by = ['Similarity'],ascending=False).head(10)

    return output 


################################################## Q3: define a new score ########################################################


def rating_score(m):
    list_element=m["ratingCount"].to_list()
    
    rate_number=[]
    for lista in list_element:
        li=""
        for valore in str(lista):
            if valore!=",":
                li=li+valore
        rate_number.append(int(li))
    
    rate_value=m["ratingValue"].to_list()
    
    max_rate_number=max(rate_number)
    max_rate_value=max(rate_value)
    rate_score=[]
    for i in range(len(m)):
        numerator=(rate_value[i])*(np.log(1+rate_number[i]))
        denominator=(max_rate_value)*(np.log(1+max_rate_number))
        score=numerator/denominator
        rate_score.append(score)
    
    return rate_score
    


    
def lenght_score(m,input_value):
    
    m["numberOfPages"]=m["numberOfPages"].fillna(100000)
    lenght_list=m["numberOfPages"].to_list()
    counter_na=lenght_list.count(100000)
    
    title_list=list(x for x in range(0,len(lenght_list)))
    difference_list=[]
    for book in lenght_list:
        difference=book-input_value
        difference_list.append(abs(difference))
    d=zip(title_list,difference_list)
    d=dict(d)
    d=sorted(d.items(),key=lambda x:x[1])
    d=dict(d)
    coefficient=1/(len(d)-counter_na)

    score_list=list(d.values())
    keys_list=list(d.keys())
    not_na=len(lenght_list)-counter_na

    score=1
    new_score_list=[]
    new_score_list.append(score)
    
    for x in range(1,len(score_list)):
    
        if x<not_na:
            
            if score_list[x]==score_list[x-1]:
                new_score_list.append(new_score_list[x-1])
                
            else:
                value=round(score-(x*coefficient),3)
                new_score_list.append(value)
        else:
            
            new_score_list.append(0)
    
    new_d=zip(keys_list,new_score_list)
    new_d=dict(new_d)
    new_d=sorted(new_d.items(),key=lambda x:x[0])
    new_d=dict(new_d)
    
    book_lenght_score=list(new_d.values())
    return book_lenght_score



def publish_score(m,input_value):
    
    m["published"]=m["published"].fillna("0000")
    date=m["published"].to_list()
    list_years=[]

    for i in range(len(date)):
        date[i]=str(date[i])
        year=(date[i][-4:])
        if year[:].isdigit():
   
            list_years.append(int(year))
        else:
            list_years.append(0)

    title_list=list(x for x in range(0,len(list_years)))
    difference_list=[]
    for book in list_years:
        difference=book-input_value
        difference_list.append(abs(difference))
    d=zip(title_list,difference_list)
    d=dict(d)
    d=sorted(d.items(),key=lambda x:x[1])
    d=dict(d)
    coefficient=1/(len(d))

    score_list=list(d.values())
    keys_list=list(d.keys())
   

    score=1
    new_score_list=[]
    new_score_list.append(score)
    
    for x in range(1,len(score_list)):
    
        
            if score_list[x]==score_list[x-1]:
                new_score_list.append(new_score_list[x-1])
            
            else:
                value=round(score-(x*coefficient),3)
                new_score_list.append(value)
        
        

    new_d=zip(keys_list,new_score_list)
    new_d=dict(new_d)
    new_d=sorted(new_d.items(),key=lambda x:x[0])
    new_d=dict(new_d)
    book_date_score=list(new_d.values())
    return book_date_score



def input_vote():
    
    c=False
    list_=[0,1,2,3,4,5]
    while c!=True:
        
        x=input("insert a value between 0 and 5\n")
        
        if x.isdigit():
            x=int(x)
            if x in list_:
                c=True
            else:
                print("wrong value inserted, please insert a correct vote\n")
        else:
            print("wrong value inserted, please insert a correct vote\n")
    
    return x



def similarity_score(m,cosine_similarity_score,rate_score,lenght_score,publish_score,k_list):
    
    weights_sum=k_list[0]+k_list[1]+k_list[2]+k_list[3]
    
    user_score=[]
    for i in range(len(cosine_similarity_score)):
        score=(k_list[0]*cosine_similarity_score[i]+k_list[1]*rate_score[i]\
               +k_list[2]*lenght_score[i]+k_list[3]*publish_score[i])/weights_sum
        user_score.append(score)
    
    matches=pd.DataFrame()
    
    matches["book_title"]=m["bookTitle"]
    matches["url"]=m["url"]
    matches["plot"]=m["plot"]
    matches["cosine_similarity"]=cosine_similarity_score
    matches["rate_score"]=rate_score
    matches["lenght_score"]=lenght_score
    matches["date_release_score"]=publish_score
    matches["user_score"]=user_score
    
    matches=matches.sort_values(by=["user_score"],ascending=False).head(10)
    

    return matches


################################################## Q4: make nice visualization ###################################################

def split_series_and_book_series(df):
    
    Series = []
    bookInSeries = []
    
    for item in df['bookSeries']:
        item = item.split('#')
        Series.append(item[0])
        if len(item)> 1:
            bookInSeries.append(item[1])
        else:
            bookInSeries.append('none')
            
    df['Series'] = Series
    df['bookInSeries'] = bookInSeries
    df = df.drop('bookSeries',axis='columns')
    df = df.reset_index()
    
    return df


def series_to_analyze(df):
    series_to_analyze = []

    for i in range(len(df)):
        if (df.iloc[i].Series not in series_to_analyze) and (df.iloc[i].bookInSeries != 'none') and (len(df.iloc[i].bookInSeries)==1):
            if len(series_to_analyze)<=10:
                series_to_analyze.append(df.iloc[i].Series)
            else:
                break
    
    return series_to_analyze



def create_new_dataframe(df,series_to_analyze):

    df1 = pd.DataFrame(columns = df.columns.tolist())

    for i in range(len(df)):
        x = df.iloc[i]
        if x.Series in series_to_analyze and len(df.iloc[i].bookInSeries) == 1:
            df1 = df1.append(df.loc[i],ignore_index=True)
    
    df1 = df1.drop({'level_0','index'},axis = 1)

    return df1



def plot_series(series_to_analyze, df1):

    for serie in series_to_analyze:

        books = df1[df1['Series']==serie].sort_values(by = ['published'])

        years = books['published']

        x = [years[years.index[i]].year -  years[years.index[0]].year for i in range(len(books))]
        y = list(map(int,books['numberOfPages'].cumsum()))

        ax = sns.barplot(x,y,palette='Blues')

        for p in ax.patches:
            ax.annotate(format(p.get_height()), 
                           (p.get_x() + p.get_width() / 2, p.get_height()), 
                           ha = 'center', va = 'center', 
                           xytext = (0, 5), 
                           textcoords = 'offset points')

        ax.set(xlabel="years since the first book of the series", ylabel = "cumulative series page count",title=serie)
        plt.show()
    
    return 

