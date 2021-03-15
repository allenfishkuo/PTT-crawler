# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:12:46 2020

@author: allen
"""
from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
import time
import sys
import os
import urllib.request
import cv2

start_time = time.time()

def crawl_one_page(href,index):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36"
    }
    payload = {'from':'/bbs/Beauty/M.1514740613.A.FF1.html', 'yes': 'yes' }
    rs = requests.session()
    res = rs.post("https://www.ptt.cc/ask/over18", data=payload)
    req = rs.get(href, headers=headers)
    req.encoding = 'utf-8'
    soup = BeautifulSoup(req.text, 'html.parser')
    article = soup.find_all('div', {'class': 'title'})
    date = soup.find_all('div', {'class': 'date'})
    boo_like = soup.find_all('div', {'class': 'nrec'})
    all_articles = open('all_articles.txt','a+',encoding="utf-8")
    all_popular = open('all_popular.txt','a+',encoding="utf-8")
    #print(len(article))
    for count in range(len(article)):
        if len(article[count]) > 1:
            article_url = article[count].find_all('a')[0]['href']
            if article[count].text.rstrip().strip()[0:4] != '[公告]' and article[count].text.rstrip().strip()[4:8] != '[公告]':                
                line = date[count].text.replace('/','') +','+ article[count].text.rstrip().strip() +','+'https://www.ptt.cc' +article_url+ '\n'
                if index == 3143 and date[count].text.replace('/','') == " 101":
                    break
                else :
                    all_articles.write(line)
                    if boo_like[count].text == '爆':
                        all_popular.write(line)
    all_articles.close()
    all_popular.close()
         
        
    
    
     

def Crawl(year):
    #index 2749 - 3143
    for index in range(2749,3144):
        #print(index)
        href = 'https://www.ptt.cc/bbs/Beauty/index'+str(index)+'.html'
        print(href)
        crawl_one_page(href,index)
        time.sleep(0.5)
    


def ptt_time_boo_like(df, start, end):
    file = 'push['+str(start)+'-'+str(end)+'].txt'
    if os.path.exists(file):
        os.remove(file)
    push_like_hate =open(file,'a+',encoding="utf-8")
    total_like ,total_hate  = 0,0
    df_like , df_hate = {}, {}
    for i in range(df.shape[0]):
        #print(df.iloc[i,1])
        if int(df.iloc[i,0]) >= int(start) and int(df.iloc[i,0]) <= int(end):
            headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36"
            }
            payload = {'from':'/bbs/Beauty/M.1514740613.A.FF1.html', 'yes': 'yes' }
            rs = requests.session()
            res = rs.post("https://www.ptt.cc/ask/over18", data=payload)
            req = rs.get(df.iloc[i,1], headers=headers)
            req.encoding = 'utf-8'
            soup = BeautifulSoup(req.text, 'html.parser')
            push = soup.find_all('div',{'class':'push'})
            for i in range(len(push)):
                #print(push[i].text)
                interval_content =push[i].text.replace(':',' ').replace(',','').rstrip('\n').split(' ')
                #print(interval_content[0],interval_content[1])
                if interval_content[0] == '推':
                    total_like += 1
                    #print({ interval_content[1] : 1})
                    if interval_content[1] not in  df_like :
                        df_like.update({ interval_content[1] : 1})
                    else :
                        df_like[interval_content[1]] +=1
                elif interval_content[0] == '噓':
                    total_hate += 1
                    #print({ interval_content[1] : 1})
                    if interval_content[1] not in  df_hate :
                        df_hate.update({ interval_content[1] : 1})
                    else :
                        df_hate[interval_content[1]] +=1
                        
    print(total_like,total_hate)
    top_10_like, top_10_hate = dict(sorted(sorted(df_like.items(), key = lambda d : d[0]),reverse = True,key = lambda d : d[1])) , dict(sorted(sorted(df_hate.items(), key = lambda d : d[0]),reverse =True,key = lambda d : d[1]))   
    push_like_hate.write('all like: '+str(total_like)+'\nall boo: '+str(total_hate)+'\n')
    for i,(k , v) in enumerate(top_10_like.items()):
        if i == 10 : break
        else :
            push_like_hate.write('like #'+str(i+1)+': '+str(k)+' '+str(v)+'\n')
    for i,(k , v) in enumerate(top_10_hate.items()):
        if i == 10 : break
        else :
            push_like_hate.write('boo #'+str(i+1)+': '+str(k)+' '+str(v)+'\n')
    push_like_hate.close()

                  

                    
                
            #print(soup.text)
            
    return

def ptt_get_popular_img(df, start, end):
    file = 'popular['+str(start)+'-'+str(end)+'].txt'
    if os.path.exists(file):
        os.remove(file)
    popular_articles =open(file,'a+',encoding="utf-8")
    total_popular = 0
    for count in range(df.shape[0]):
        if int(df.iloc[count,0]) >= int(start) and int(df.iloc[count,0]) <= int(end):
            total_popular += 1
    popular_articles.write('number of popular articles: '+str(total_popular)+'\n')
    for i in range(df.shape[0]):
        #print(df.iloc[i,1])
        if int(df.iloc[i,0]) >= int(start) and int(df.iloc[i,0]) <= int(end):
            headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36"
            }
            payload = {'from':'/bbs/Beauty/M.1514740613.A.FF1.html', 'yes': 'yes' }
            rs = requests.session()
            res = rs.post("https://www.ptt.cc/ask/over18", data=payload)
            req = rs.get(df.iloc[i,1], headers=headers)
            req.encoding = 'utf-8'
            soup = BeautifulSoup(req.text, 'html.parser')
            push = soup.find_all('a')
            
            for i in range(len(push)):
                 if push[i].text[-4:] == '.jpg' or push[i].text[-4:] == '.gif' or push[i].text[-5:] == '.jpeg' or push[i].text[-4:] == '.png':
                     print(push[i].text)
                     popular_articles.write(push[i].text+'\n')

    popular_articles.close()

def ptt_get_keyword_img(df,keyword, start, end):
    
    file = 'keyward('+str(keyword)+')['+str(start)+'-'+str(end)+'].txt'
    if os.path.exists(file):
        os.remove(file)
    keyword_articles =open(file,'a+',encoding="utf-8")
    
    word =''
    for i in range(df.shape[0]):
        
        if int(df.iloc[i,0]) >= int(start) and int(df.iloc[i,0]) <= int(end):
            headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36"
            }
            payload = {'from':'/bbs/Beauty/M.1514740613.A.FF1.html', 'yes': 'yes' }
            rs = requests.session()
            res = rs.post("https://www.ptt.cc/ask/over18", data=payload)
            req = rs.get(df.iloc[i,1], headers=headers)
            req.encoding = 'utf-8'
            soup = BeautifulSoup(req.text, 'html.parser')
            total = soup.find_all('div',{'id':'main-container'})
            push = soup.find_all('a')
            for k in range(len(total)):
                word += total[k].text
            #print(word)
            word = word.split("※ 發信站")[0]
            if keyword in word :               
                
                for j in range(len(push)):
                    print(push[j].text)
                    if push[j].text[-4:] == '.jpg' or push[j].text[-4:] == '.gif' or push[j].text[-5:] == '.jpeg' or push[j].text[-4:] == '.png':
                        
                        keyword_articles.write(push[j].text+'\n')
    keyword_articles.close()
 


def read_file(file_name):
    try :
        articles = open(file_name,'r',encoding = 'utf8')
    except :
        print("you should crawl first")
    
    line = articles.readline()
    
    dataframe = pd.DataFrame(columns = ['date','href'])
    while line:
       #print(line)
       content = line.rstrip('\n').split(',')
       #print(content[0],content[2])
       dataframe = dataframe.append({'date':content[0],'href':content[len(content)-1]},ignore_index=True)       
       line = articles.readline()
    articles.close() 
    print(dataframe)
    #dataframe = pd.Dataframe(['date','href'])
    return dataframe


def imgfilename():
    file = open('popular[101-1231].txt','r')
    urls = file.readlines()[1:]
  
    for i , url in enumerate(urls):    
        url = url.split('\n')[0]
        print(url)
        
        if url[-4:] == '.jpg' or url[-4:] == '.png':
            try :
                resp = urllib.request.urlopen(url)
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            except :
                continue
            try :
                cv2.imwrite('./imgfilename/'+str(i+1)+'.jpg', image)
            except :
                continue
    return  


    
def main():
    #sys.argv = ['0856720.py','push','701','731']
    sys.argv = ['0856720.py','imgfilename']
    input_len = len(sys.argv)
    if input_len == 2 :
        if sys.argv[1] == 'crawl':
            Crawl(2019)
        elif sys.argv[1] =='imgfilename':
            imgfilename()
    elif input_len == 4 :
        if sys.argv[1] == "push":
            start_date = int(sys.argv[2])
            end_date = int(sys.argv[3])
            df = read_file('all_articles.txt')
            ptt_time_boo_like(df, start_date, end_date)
        elif sys.argv[1] == 'popular':
            start_date = int(sys.argv[2])
            end_date = int(sys.argv[3])
            df = read_file('all_popular.txt')
            ptt_get_popular_img(df, start_date, end_date)
    elif input_len == 5:
        if sys.argv[1] == 'keyword':
            keyword = str(sys.argv[2])
            start_date = int(sys.argv[3])
            end_date = int(sys.argv[4])
            df = read_file('all_articles.txt')
            ptt_get_keyword_img(df, keyword, start_date, end_date)
        
            

if __name__ == "__main__":
    main()
    print("--- %s seconds ---" % (time.time() - start_time))



