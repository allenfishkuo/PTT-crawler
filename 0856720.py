# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 17:17:54 2020

@author: Allen
"""

from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
import time
import sys
import os
"""
from random import shuffle 
from tqdm import tqdm 
from tflearn.data_augmentation import ImageAugmentation
import tflearn 
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.conv import conv_2d, max_pool_2d, residual_block, batch_normalization 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression   
import tensorflow as tf 
#import cv2
"""
start_time = time.time()
TEST_DIR ='./imgfilelistname/'
IMG_SIZE = 256
LR = 1e-4
MODEL_NAME = 'image_classification-{}-{}.model'.format(LR, '6conv-basic') 
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
                line = date[count].text.strip().replace('/','') +','+ article[count].contents[1].text +','+'https://www.ptt.cc' +article_url+ '\n'
                #print(date[count].text.replace('/',''))
                if (index == 3142 and date[count].text.replace('/','') == " 101") or (index == 2748 and date[count].text.replace('/','') == "1231"):
                    continue
                else :
                    all_articles.write(line)
                    if boo_like[count].text == '爆':
                        all_popular.write(line)
    all_articles.close()
    all_popular.close()
         
        
    
    
     

def Crawl(year):
    #index 2749 - 3143
    for index in range(2748,3143):
        #print(index)
        href = 'https://www.ptt.cc/bbs/Beauty/index'+str(index)+'.html'
        #print(href)
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
            #print(len(push))
            for i in range(len(push)):
                #print(push[i].text)
                interval_content =push[i].text.replace(':',' ').replace(',','').rstrip('\n').split(' ')
               # print(interval_content[0],interval_content[1])
                if interval_content[0] == '推' :
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
                        
    #print(total_like,total_hate)
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
                     #print(push[i].text)
                     popular_articles.write(push[i].text+'\n')
    
    popular_articles.close()

def ptt_get_keyword_img(df,keyword, start, end):
    
    file = 'keyword('+str(keyword)+')['+str(start)+'-'+str(end)+'].txt'
    if os.path.exists(file):
        os.remove(file)
    keyword_articles =open(file,'a+',encoding="utf-8")
    
    
    for i in range(df.shape[0]):
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
            total = soup.find_all('div',{'id':'main-container'})
            push = soup.find_all('a')
            for k in range(len(total)):
                word += total[k].text
            #print(word)
            word = word.split("※ 發信站")[0]
            if keyword in word :               
                
                for j in range(len(push)):
                    #print(push[j].text)
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
    #dataframe = pd.Dataframe(['date','href'])
    return dataframe
"""
def process_test_data(): 
    testing_data = [] 
    for img in tqdm(os.listdir(TEST_DIR)): 
            path = os.path.join(TEST_DIR, img) 
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
            try :
                img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE)) 
            except :
                continue
            testing_data.append(np.array(img)) 
    #np.save('test_data.npy', testing_data) 
    return testing_data 

def tf_model():
    tf.reset_default_graph()
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
    conv1 = conv_2d(convnet, 32, 2, activation='relu')
    conv1 = max_pool_2d(conv1, 2)
    conv2 = conv_2d(conv1, 64, 2, activation='relu')
    conv2 = max_pool_2d(conv2, 2)
    conv3 = conv_2d(conv2, 64, 2, activation='relu')
    conv3 = max_pool_2d(conv3, 2)    
    conv4 = conv_2d(conv3, 128, 2, activation='relu')
    conv4 = max_pool_2d(conv4, 2)    
    conv5 = conv_2d(conv4, 128, 2, activation='relu')
    conv5 = max_pool_2d(conv5, 2)    
    conv6 = conv_2d(conv5, 256, 2, activation='relu')
    conv6 = max_pool_2d(conv6, 2)    
    conv7 = conv_2d(conv6, 256, 2, activation='relu')
    conv7 = max_pool_2d(conv7, 2)    
    conv8 = conv_2d(conv7, 512, 2, activation='relu')
    conv8 = max_pool_2d(conv8, 2)
    fc1 = fully_connected(conv8, 1024, activation='relu')
    fc1 = dropout(fc1, 0.8)
    fc2 = fully_connected(fc1, 128, activation='relu')
    fc2 = dropout(fc2, 0.8)
    output = fully_connected(fc2, 2, activation='softmax')
    output = regression(output, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    
    model = tflearn.DNN(output, tensorboard_dir='log')     # logs to temp file for tensorboard analysis
    return model
    

def test_classification():
    file = 'classification.txt'
    testing_data = process_test_data()
    testing_model = tf_model()
    testing_model.load('./'+str(MODEL_NAME))
    ans ='' 
    classification_file = open(file,'a+',encoding="utf-8")
    for num, data in enumerate(testing_data):      
        data = data.reshape(IMG_SIZE, IMG_SIZE, 1) 
        model_out = testing_model.predict([data])
        #print("model out:", model_out)
        ans += str(np.argmax(model_out))
    classification_file.write(ans)
    classification_file.close()
"""
def main():
    #sys.argv = ['0856720.py','push','102','102']
    #sys.argv = ['0856720.py','crawl']
    input_len = len(sys.argv)
    if input_len == 2 :
        if sys.argv[1] == 'crawl':
            Crawl(2019)
        #elif sys.argv[1] =='imgfilelistname':
           # test_classification()
            
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
    