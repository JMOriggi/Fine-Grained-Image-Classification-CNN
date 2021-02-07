#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import urllib.request

url = 'https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/13747/868529/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1605819874&Signature=IyaMRaZoc5y1OFUxN7DxuqQamxMonNrJ15q2GKdmyRVjYrSvOqGYaz8cfRTxFJGN3ydM9dmZfjTXaZhCknY7cZRESb5rayBKEYHre2KSVRemQOD5xTtnFkWA1CTdmIwRJTRaTY4XRmAWinuOs6%2B0CsGSWGKCS%2FCZAYPZDeVex3KbZhgOe7KKJcTYh%2BZ9u%2FYIFW81%2B4mli%2BAaF%2FBeWq41pPhLFpIz5piDQcu%2B3yIsnuOIu0gwYiFFlaLMH%2BBiBW3oHAjFG6n6%2B1B26DWr%2BIDMo%2BuY33xjmG2fQADQwy6%2FKE4OEAe3EM88N40UD5881OtwfwBz%2BJSBcsNZzj%2Bu0ykytA%3D%3D&response-content-disposition=attachment%3B+filename%3Dinaturalist-2019-fgvc6.zip'
urllib.request.urlretrieve(url, 'data.zip')


# In[ ]:


import zipfile
with zipfile.ZipFile('data.zip', 'r') as f:
    f.extractall('./rawData/')


# In[ ]:


# import tarfile
# with tarfile.open('./rawData/train_val2019.tar.gz') as f:
#     f.extractall('./rawData/') # specify which folder to extract to

# with tarfile.open('./rawData/test2019.tar.gz') as f:
#     f.extractall('./rawData/') # specify which folder to extract to


# # In[2]:


# import os
 
# for filename in os.listdir('./rawData/train_val2019'):
#     for f in os.listdir('./rawData/train_val2019/'+filename+'/'):
#         os.replace('./rawData/train_val2019/'+filename+'/'+f, './data/'+f)

# for i in range(1010):
#     directory = './data/' + str(i)
#     for filename in os.listdir(directory):
#         if not filename.endswith(".jpg"):
#             print('deleted')
#             os.remove('./data/'+str(i)+'/'+filename)

