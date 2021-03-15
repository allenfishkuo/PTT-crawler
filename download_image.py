# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 23:44:15 2020

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
    all_articles = open('all_articles_1.txt','a+',encoding="utf-8")
    all_popular = open('all_popular_1.txt','a+',encoding="utf-8")
    #print(len(article))
    for count in range(len(article)):
        if len(article[count]) > 1:
            article_url = article[count].find_all('a')[0]['href']
            if article[count].text.rstrip().strip()[0:4] != '[公告]' and article[count].text.rstrip().strip()[4:8] != '[公告]':                
                line = date[count].text.replace('/','') +','+ article[count].text.rstrip().strip() +','+'https://www.ptt.cc' +article_url+ ','
                if index == 3143 and date[count].text.replace('/','') == " 101":
                    break
                else :
                    all_articles.write(line)
                    if boo_like[count].text == '爆':
                        all_popular.write(line)
                        all_articles.write('1\n')
                    else :
                        all_articles.write('0\n')
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
    '''
    file = 'push['+str(start)+'-'+str(end)+'].txt'
    if os.path.exists(file):
        os.remove(file)
    push_like_hate =open(file,'a+',encoding="utf-8")
    '''
    count_0 = 7166
    count_1 = 2711
    for i in range(df.shape[0]):
        total_like = 0
        word =''
        
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
            img = soup.find_all('div',{'id':'main-content'})
            for title in img:
                word += title.text
            word = word.split('2019')[1]
            word = word.split("※ 發信站")[0]
            word = word.split('\n')
            
            print(df.iloc[i,0])

            for i in range(len(push)):
                #print(push[i].text)
                interval_content =push[i].text.replace(':',' ').replace(',','').rstrip('\n').split(' ')
                #print(interval_content[0],interval_content[1])
                if interval_content[0] == '推':
                    total_like += 1
            if total_like >= 35 :
                for line in word:
                    if line[-4:] == '.jpg':
                        try :
                            
                            resp = urllib.request.urlopen(line)
                            image = np.asarray(bytearray(resp.read()), dtype="uint8")
                            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                        except :
                            continue
                        try :
                            cv2.imwrite('./imgfilename/1/'+str(count_1+1)+'.jpg', image)
                            count_1 += 1
                        except :
                            continue
            """
            else :
                for line in word:
                    if line[-4:] == '.jpg':
                        try :
                            
                            resp = urllib.request.urlopen(line)
                            image = np.asarray(bytearray(resp.read()), dtype="uint8")
                            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                        except :
                            continue
                        try :
                            cv2.imwrite('./imgfilename/0/'+str(count_0+1)+'.jpg', image)
                            count_0 += 1
                        except :
                            print('notin')
                            continue
            """
                
                    
                
    

                  

                    
                
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
    
    dataframe = pd.DataFrame(columns = ['date','href','label'])
    while line:
       #print(line)
       content = line.rstrip('\n').split(',')
       #print(content[0],content[2])
       dataframe = dataframe.append({'date':content[0],'href':content[len(content)-2],'label':content[len(content)-2]},ignore_index=True)       
       line = articles.readline()
    articles.close() 
    print(dataframe)
    #dataframe = pd.Dataframe(['date','href'])
    return dataframe


def imgfilename():
    return  

def load_image():
    
    return 

    
def main():
    sys.argv = ['0856720.py','push','1001','1231']
    #sys.argv = ['0856720.py','crawl']
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
            df = read_file('all_articles_1.txt')
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



