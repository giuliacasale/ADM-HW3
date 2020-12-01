import numpy as np
import pandas as pd



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