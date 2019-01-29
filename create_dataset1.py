
# coding: utf-8

# In[1]:


import random


# In[2]:


lines_seen = set()
f = open("sample_train_dataset.txt","w+")
total = 0
while(total!=400):
    a = random.randint(0,7)
    string = './dataset/00' + str(a) + '_0' + str(random.randint(10,64))+'.jpg'    
    if string not in lines_seen:
        f.write("%s\t" %(string))
        f.write("%s\n" %str((a)))
        total+=1
        lines_seen.add(string)
f.close()


# In[3]:


f = open("sample_test_dataset.txt","w+")
g = open("output_of_sample_test_dataset.txt","w+")

total = 0
while(total!=100):
    a = random.randint(0,7)
    string = './dataset/00' + str(a) + '_0' + str(random.randint(10,64))+'.jpg'
    string1 = './dataset/00' + str(a) + '_00' + str(random.randint(0,9))+'.jpg'
    if string not in lines_seen:
        g.write("%s\n" %str((a)))
        f.write("%s\n" %(string))
        total+=1
    lines_seen.add(string)
    
    if string1 not in lines_seen:
        g.write("%s\n" %str((a)))
        f.write("%s\n" %(string1))
        total+=1    
    lines_seen.add(string1)
    
f.close()

