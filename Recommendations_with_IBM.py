#!/usr/bin/env python
# coding: utf-8

# # Recommendations with IBM
# 
# In this notebook, you will be putting your recommendation skills to use on real data from the IBM Watson Studio platform. 
# 
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/rubrics/2322/view).  **Please save regularly.**
# 
# By following the table of contents, you will build out a number of different methods for making recommendations that can be used for different situations. 
# 
# 
# ## Table of Contents
# 
# I. [Exploratory Data Analysis](#Exploratory-Data-Analysis)<br>
# II. [Rank Based Recommendations](#Rank)<br>
# III. [User-User Based Collaborative Filtering](#User-User)<br>
# IV. [Content Based Recommendations (EXTRA - NOT REQUIRED)](#Content-Recs)<br>
# V. [Matrix Factorization](#Matrix-Fact)<br>
# VI. [Extras & Concluding](#conclusions)
# 
# At the end of the notebook, you will find directions for how to submit your work.  Let's get started by importing the necessary libraries and reading in the data.

# In[57]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import project_tests as t
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('data/user-item-interactions.csv')
df_content = pd.read_csv('data/articles_community.csv')
del df['Unnamed: 0']
del df_content['Unnamed: 0']

# Show df to get an idea of the data
df.head(10)


# In[58]:


# Show df_content to get an idea of the data
df_content.head(10)


# In[59]:


df_content.info()


# ### <a class="anchor" id="Exploratory-Data-Analysis">Part I : Exploratory Data Analysis</a>
# 
# Use the dictionary and cells below to provide some insight into the descriptive statistics of the data.
# 
# `1.` What is the distribution of how many articles a user interacts with in the dataset?  Provide a visual and descriptive statistics to assist with giving a look at the number of times each user interacts with an article.  

# In[60]:


df.describe()


# In[61]:


df_grouped_by_user = df.groupby(['email']).count()
df_grouped_by_user.describe()


# In[62]:


dist = df.groupby('email')['article_id'].count().reset_index(name='count')
plt.hist(dist['count'], bins=20,color = "green",lw=0)
plt.xlabel('User - Articles Interaction')
plt.ylabel('Count')
plt.title('User - Articles Interaction Distribution')
plt.show()


# In[63]:


dist.describe()


# In[64]:


# Fill in the median and maximum number of user_article interactios below

median_val =3 # 50% of individuals interact with ____ number of articles or fewer.
max_views_by_user =364 # The maximum number of user-article interactions by any 1 user is ______.


# `2.` Explore and remove duplicate articles from the **df_content** dataframe.  

# In[65]:


len(df_content['article_id']) - len(np.unique(df_content['article_id']))


# In[66]:


# Find and explore duplicate articles
df_content[df_content.duplicated(subset = 'article_id', keep = False) == True]


# In[67]:


# Remove any rows that have the same article_id - only keep the first
df_content.drop_duplicates(subset = 'article_id', keep = 'first', inplace = True)
df_content.info()


# `3.` Use the cells below to find:
# 
# **a.** The number of unique articles that have an interaction with a user.  
# **b.** The number of unique articles in the dataset (whether they have any interactions or not).<br>
# **c.** The number of unique users in the dataset. (excluding null values) <br>
# **d.** The number of user-article interactions in the dataset.

# In[68]:


#a. The number of unique articles that have an interaction with a user.
unique_articles = df.drop_duplicates(subset = ['article_id']).shape[0]
print(unique_articles)

#b. The number of unique articles in the dataset (whether they have any interactions or not).
total_articles = len(np.unique(df_content['doc_full_name']))
print(total_articles)

#c. The number of unique users in the dataset. (excluding null values)
unique_users = df_grouped_by_user.article_id.count()
unique_users
print(unique_users)

#d. The number of user-article interactions in the dataset.
df.shape[0]
print(df.shape[0])


# In[69]:


unique_articles =df.drop_duplicates(subset = ['article_id']).shape[0] # The number of unique articles that have at least one interaction
total_articles =len(np.unique(df_content['doc_full_name'])) # The number of unique articles on the IBM platform
unique_users =df_grouped_by_user.article_id.count() # The number of unique users
user_article_interactions =df.shape[0] # The number of user-article interactions


# `4.` Use the cells below to find the most viewed **article_id**, as well as how often it was viewed.  After talking to the company leaders, the `email_mapper` function was deemed a reasonable way to map users to ids.  There were a small number of null values, and it was found that all of these null values likely belonged to a single user (which is how they are stored using the function below).

# In[70]:


df['article_id'].value_counts()[df['article_id'].value_counts() == df['article_id'].value_counts().max()]


# In[71]:


df['article_id'].value_counts()


# In[72]:


df['article_id'].value_counts() == df['article_id'].value_counts().max()


# In[73]:


most_viewed_article_id ='1429.0' # The most viewed article in the dataset as a string with one value following the decimal 
max_views =937 # The most viewed article in the dataset was viewed how many times?


# In[74]:


## No need to change the code here - this will be helpful for later parts of the notebook
# Run this cell to map the user email to a user_id column and remove the email column

def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

email_encoded = email_mapper()
del df['email']
df['user_id'] = email_encoded

# show header
df.head()


# In[75]:


## If you stored all your results in the variable names above, 
## you shouldn't need to change anything in this cell

sol_1_dict = {
    '`50% of individuals have _____ or fewer interactions.`': median_val,
    '`The total number of user-article interactions in the dataset is ______.`': user_article_interactions,
    '`The maximum number of user-article interactions by any 1 user is ______.`': max_views_by_user,
    '`The most viewed article in the dataset was viewed _____ times.`': max_views,
    '`The article_id of the most viewed article is ______.`': most_viewed_article_id,
    '`The number of unique articles that have at least 1 rating ______.`': unique_articles,
    '`The number of unique users in the dataset is ______`': unique_users,
    '`The number of unique articles on the IBM platform`': total_articles
}

# Test your dictionary against the solution
t.sol_1_test(sol_1_dict)


# ### <a class="anchor" id="Rank">Part II: Rank-Based Recommendations</a>
# 
# Unlike in the earlier lessons, we don't actually have ratings for whether a user liked an article or not.  We only know that a user has interacted with an article.  In these cases, the popularity of an article can really only be based on how often an article was interacted with.
# 
# `1.` Fill in the function below to return the **n** top articles ordered with most interactions as the top. Test your function using the tests below.

# In[76]:


def get_top_articles(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    # Your code here
    top_articles = []
    top_article_ids = get_top_article_ids(n, df)
    
    df_unique_articles = df.drop('user_id', axis = 1).drop_duplicates(subset=['article_id']).title
    
    for i in top_article_ids:
        top_articles.append(df_unique_articles.loc[df.article_id == float(i)].values[0])
    
    return top_articles # Return the top article titles from df (not df_content)

def get_top_article_ids(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    # Your code here
    top_articles = []
    sorted_count = df.groupby(['article_id']).count().title.sort_values(ascending = False)
    for i in range(0, n):
        top_articles.append(str(sorted_count.index[i]))
 
    return top_articles # Return the top article ids


# In[77]:


print(get_top_articles(10))
print(get_top_article_ids(10))


# In[78]:


# most interactions articles
print(get_top_articles(1))


# In[79]:


# Test your function by returning the top 5, 10, and 20 articles
top_5 = get_top_articles(5)
top_10 = get_top_articles(10)
top_20 = get_top_articles(20)

# Test each of your three lists from above
t.sol_2_test(get_top_articles)


# ### <a class="anchor" id="User-User">Part III: User-User Based Collaborative Filtering</a>
# 
# 
# `1.` Use the function below to reformat the **df** dataframe to be shaped with users as the rows and articles as the columns.  
# 
# * Each **user** should only appear in each **row** once.
# 
# 
# * Each **article** should only show up in one **column**.  
# 
# 
# * **If a user has interacted with an article, then place a 1 where the user-row meets for that article-column**.  It does not matter how many times a user has interacted with the article, all entries where a user has interacted with an article should be a 1.  
# 
# 
# * **If a user has not interacted with an item, then place a zero where the user-row meets for that article-column**. 
# 
# Use the tests to make sure the basic structure of your matrix matches what is expected by the solution.

# In[80]:


# create the user-article matrix with 1's and 0's

def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    # Fill in the function here
    df_local=df.groupby(['article_id', 'user_id']).count()
    df_local=df_local.pivot_table(index='user_id',columns='article_id',values='title')
    df_local = df_local.replace(np.nan, 0)
    user_item=df_local.applymap(lambda x: 1 if x > 0 else x)
    
    return user_item # return the user_item matrix 

user_item = create_user_item_matrix(df)
print(user_item)


# In[82]:


## Tests: You should just need to run this cell.  Don't change the code.
assert user_item.shape[0] == 5149, "Oops!  The number of users in the user-article matrix doesn't look right."
assert user_item.shape[1] == 714, "Oops!  The number of articles in the user-article matrix doesn't look right."
assert user_item.sum(axis=1)[1] == 36, "Oops!  The number of articles seen by user 1 doesn't look right."
print("You have passed our quick tests!  Please proceed!")


# `2.` Complete the function below which should take a user_id and provide an ordered list of the most similar users to that user (from most similar to least similar).  The returned result should not contain the provided user_id, as we know that each user is similar to him/herself. Because the results for each user here are binary, it (perhaps) makes sense to compute similarity as the dot product of two users. 
# 
# Use the tests to test your function.

# In[83]:


def find_similar_users(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first
    
    Description:
    Computes the similarity of every pair of users based on the dot product
    Returns an ordered
    
    '''
    user_row = user_item.iloc[user_id-1]
    # compute similarity of each user to the provided user    
    #similarity = np.dot(user_row, user_item.T)
    similarity = user_item.dot(user_item.loc[user_id])
    
    
    # sort by similarity
    #sorted_users = np.argsort(-similarity)
    similarity = similarity.sort_values(ascending=False)
    
    
    # create list of just the ids
    #most_similar_users = list(sorted_users[1:])  
    similarity = similarity.index
    
    
    #print(most_similar_users)
    # remove the own user's id
    #most_similar_users.remove(user_id)
    most_similar_users = similarity.drop(user_id)
    
    
    #print(most_similar_users)  
    
    return most_similar_users # return a list of the users in order from most to least similar
    
        


# In[84]:


# Do a spot check of your function
print("The 10 most similar users to user 1 are: {}".format(find_similar_users(1)[:10]))
print("The 5 most similar users to user 3933 are: {}".format(find_similar_users(3933)[:5]))
print("The 3 most similar users to user 46 are: {}".format(find_similar_users(46)[:3]))


# `3.` Now that you have a function that provides the most similar users to each user, you will want to use these users to find articles you can recommend.  Complete the functions below to return the articles you would recommend to each user. 

# In[85]:


def get_article_names(article_ids, df=df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''
    # Your code here
    article_names = []
    df_local = df.drop_duplicates(subset = 'article_id')
    
    for id in article_ids:
        article_names.append(df_local[df_local['article_id'] == float(id)]['title'].iloc[0])    
        
    return article_names # Return the article names associated with list of article ids


def get_user_articles(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the doc_full_name column in df_content)
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    # Your code here
    article_ids = []
    
    for article_id in user_item.columns:
        if user_item.loc[user_id, article_id] == 1:
            #print(article_id)
            article_ids.append(str(article_id))

    article_names = get_article_names(article_ids)
    
    return article_ids, article_names # return the ids and names


def user_user_recs(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    Users who are the same closeness are chosen arbitrarily as the 'next' user
    
    For the user where the number of recommended articles starts below m 
    and ends exceeding m, the last items are chosen arbitrarily
    
    '''
    # Your code here
    # get a list of articles that the given user has seen
    seen_articles = get_user_articles(user_id, user_item)[0]
    
    # retrieve similar user lists
    similar_users = find_similar_users(user_id, user_item)
    
    # loop through the users based on the closeness

    recs = np.array([])
    
    for user in similar_users:
        
        neighbor_seen_articles = get_user_articles(user, user_item)[0]
        
        # find the list of articles that are not seen by the user
        new_recs = np.setdiff1d(neighbor_seen_articles, seen_articles, assume_unique=True)
        
        # Update recs with new recs
        recs = np.unique(np.concatenate([new_recs, recs], axis=0))
        
        # If we have enough recommendations exit the loop     
        if len(recs) >= m:
            break
    
    new_recs = recs[:m].tolist()
    
    return recs # return your recommendations for this user_id    


# In[86]:


# Check Results
get_article_names(user_user_recs(1, 10)) # Return 10 recommendations for user 1


# In[87]:


# Test your functions here - No need to change this code - just run this cell
assert set(get_article_names(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])) == set(['using deep learning to reconstruct high-resolution audio', 'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model', 'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding', 'use xgboost, scikit-learn & ibm watson machine learning apis']), "Oops! Your the get_article_names function doesn't work quite how we expect."
assert set(get_article_names(['1320.0', '232.0', '844.0'])) == set(['housing (2015): united states demographic measures','self-service data preparation with ibm data refinery','use the cloudant-spark connector in python notebook']), "Oops! Your the get_article_names function doesn't work quite how we expect."
assert set(get_user_articles(20)[0]) == set(['1320.0', '232.0', '844.0'])
assert set(get_user_articles(20)[1]) == set(['housing (2015): united states demographic measures', 'self-service data preparation with ibm data refinery','use the cloudant-spark connector in python notebook'])
assert set(get_user_articles(2)[0]) == set(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])
assert set(get_user_articles(2)[1]) == set(['using deep learning to reconstruct high-resolution audio', 'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model', 'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding', 'use xgboost, scikit-learn & ibm watson machine learning apis'])
print("If this is all you see, you passed all of our tests!  Nice job!")


# `4.` Now we are going to improve the consistency of the **user_user_recs** function from above.  
# 
# * Instead of arbitrarily choosing when we obtain users who are all the same closeness to a given user - choose the users that have the most total article interactions before choosing those with fewer article interactions.
# 
# 
# * Instead of arbitrarily choosing articles from the user where the number of recommended articles starts below m and ends exceeding m, choose articles with the articles with the most total interactions before choosing those with fewer total interactions. This ranking should be  what would be obtained from the **top_articles** function you wrote earlier.

# In[88]:


def get_top_sorted_users(user_id, df=df, user_item=user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook 
    user_item - (pandas dataframe) matrix of users by articles: 
            1's when a user has interacted with an article, 0 otherwise
    
            
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u
                    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
     
    '''
    # Your code here
    user_row = user_item.iloc[user_id-1]# -1 is for switching from value to its index

    # get  similarity of the given user with others
    similarity = np.dot(user_row, user_item.T)

    # sorting by similarity
    sorted_users = np.argsort(-similarity) + 1 # +1 is to switch back to the value

    # sort the similarity scores
    sorted_scores = np.sort(similarity)[::-1]

    # save the to a Dataframe
    df_joined = list(zip(sorted_users, sorted_scores))
    neighbors_df = pd.DataFrame(df_joined, columns = ['neighbor_user', 'similarity'])

    # remove the user's similarity with themself by dropping the first record
    neighbors_df = neighbors_df.drop(neighbors_df.index[0], axis = 0)
    
    # user_id and how many articles they've interacted with
    user_interactions = df.groupby('user_id')['article_id'].count().sort_values(ascending=False).reset_index(name='count')
    
    # add num_interactions column
    neighbors_df = pd.merge(neighbors_df, user_interactions, how='left',                  left_on='neighbor_user', right_on='user_id').drop('user_id', axis=1)
    
    # then sort dataframe desc by similarity and then by number of interactions
    neighbors_df = neighbors_df.sort_values(by=['similarity', 'count'], ascending=[False, False])  
    return neighbors_df # Return the dataframe specified in the doc_string


def user_user_recs_part2(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    * Choose the users that have the most total article interactions 
    before choosing those with fewer article interactions.

    * Choose articles with the articles with the most total interactions 
    before choosing those with fewer total interactions. 
   
    '''
    # Your code here
    recs = []
    seen_article_ids = get_user_articles(user_id)[0]
    nearest_users_df = get_top_sorted_users(user_id)
    
    for near_user in nearest_users_df['neighbor_user']:
        if len(recs) > m:
                    break
        for article in get_user_articles(near_user)[0]:
            if article not in seen_article_ids:
                if len(recs) > m:
                    break
                else:
                    recs.append(article)
    rec_names = get_article_names(recs)
    
    return recs, rec_names


# In[89]:


# Quick spot check - don't change this code - just use it to test your functions
rec_ids, rec_names = user_user_recs_part2(20, 10)
print("The top 10 recommendations for user 20 are the following article ids:")
print(rec_ids)
print()
print("The top 10 recommendations for user 20 are the following article names:")
print(rec_names)


# `5.` Use your functions from above to correctly fill in the solutions to the dictionary below.  Then test your dictionary against the solution.  Provide the code you need to answer each following the comments below.

# In[90]:


get_top_sorted_users(1).head(5)


# In[91]:


get_top_sorted_users(131).head(10)


# In[92]:


user1_most_sim=get_top_sorted_users(1).loc[0,'neighbor_user']
print(user1_most_sim)
user131_10th_sim=get_top_sorted_users(131).loc[9,'neighbor_user']
print(user131_10th_sim)


# In[93]:


### Tests with a dictionary of results

user1_most_sim = 3933# Find the user that is most similar to user 1 
user131_10th_sim =242 # Find the 10th most similar user to user 131


# In[94]:


## Dictionary Test Here
sol_5_dict = {
    'The user that is most similar to user 1.': user1_most_sim, 
    'The user that is the 10th most similar to user 131': user131_10th_sim,
}

t.sol_5_test(sol_5_dict)


# `6.` If we were given a new user, which of the above functions would you be able to use to make recommendations?  Explain.  Can you think of a better way we might make recommendations?  Use the cell below to explain a better method for new users.

# **Response is.**

# **In the case of a new user (called cold start problem) the above methods are not applicable because the user_item matrix will be empty. 
# Instead, we could use the rank based recommendation method to make recommendations. For example, we can recommend the most popular articles (the most interacted articles in our case).
# When we face this kind of problem,As I mentioned above it knowns as the cold start problem, we could also use content based recommendations method that provides recommendations utilizing information about the content of articles.**

# `7.` Using your existing functions, provide the top 10 recommended articles you would provide for the a new user below.  You can test your function against our thoughts to make sure we are all on the same page with how we might make a recommendation.

# In[95]:


new_user = '0.0'

# What would your recommendations be for this new user '0.0'?  As a new user, they have no observed articles.
# Provide a list of the top 10 article ids you would give to 
new_user_recs = new_user_recs = get_top_article_ids(10)# Your recommendations here

new_user_recs


# In[96]:


assert set(new_user_recs) == set(['1314.0','1429.0','1293.0','1427.0','1162.0','1364.0','1304.0','1170.0','1431.0','1330.0']), "Oops!  It makes sense that in this case we would want to recommend the most popular articles, because we don't know anything about these users."

print("That's right!  Nice job!")


# ### <a class="anchor" id="Content-Recs">Part IV: Content Based Recommendations (EXTRA - NOT REQUIRED)</a>
# 
# Another method we might use to make recommendations is to perform a ranking of the highest ranked articles associated with some term.  You might consider content to be the **doc_body**, **doc_description**, or **doc_full_name**.  There isn't one way to create a content based recommendation, especially considering that each of these columns hold content related information.  
# 
# `1.` Use the function body below to create a content based recommender.  Since there isn't one right answer for this recommendation tactic, no test functions are provided.  Feel free to change the function inputs if you decide you want to try a method that requires more input values.  The input values are currently set with one idea in mind that you may use to make content based recommendations.  One additional idea is that you might want to choose the most popular recommendations that meet your 'content criteria', but again, there is a lot of flexibility in how you might make these recommendations.
# 
# ### This part is NOT REQUIRED to pass this project.  However, you may choose to take this on as an extra way to show off your skills.

# In[ ]:


def make_content_recs():
    '''
    INPUT:
    
    OUTPUT:
    
    '''


# `2.` Now that you have put together your content-based recommendation system, use the cell below to write a summary explaining how your content based recommender works.  Do you see any possible improvements that could be made to your function?  Is there anything novel about your content based recommender?
# 
# ### This part is NOT REQUIRED to pass this project.  However, you may choose to take this on as an extra way to show off your skills.

# **Write an explanation of your content based recommendation system here.**

# `3.` Use your content-recommendation system to make recommendations for the below scenarios based on the comments.  Again no tests are provided here, because there isn't one right answer that could be used to find these content based recommendations.
# 
# ### This part is NOT REQUIRED to pass this project.  However, you may choose to take this on as an extra way to show off your skills.

# In[ ]:


# make recommendations for a brand new user


# make a recommendations for a user who only has interacted with article id '1427.0'


# ### <a class="anchor" id="Matrix-Fact">Part V: Matrix Factorization</a>
# 
# In this part of the notebook, you will build use matrix factorization to make article recommendations to the users on the IBM Watson Studio platform.
# 
# `1.` You should have already created a **user_item** matrix above in **question 1** of **Part III** above.  This first question here will just require that you run the cells to get things set up for the rest of **Part V** of the notebook. 

# In[97]:


# Load the matrix here
user_item_matrix = pd.read_pickle('user_item_matrix.p')


# In[98]:


# quick look at the matrix
user_item_matrix.head(10)


# `2.` In this situation, you can use Singular Value Decomposition from [numpy](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.svd.html) on the user-item matrix.  Use the cell to perform SVD, and explain why this is different than in the lesson.

# In[99]:


# Perform SVD on the User-Item Matrix Here

u, s, vt = np.linalg.svd(user_item_matrix) # use the built in to get the three matrices


# In[100]:


u.shape, s.shape, vt.shape


# **Response is.**
# 
# **The missing value is the key factor when we use SVD.
# The SVD in NumPy does not work when the matrix has missing values, which was the case for the user-movie matrix in the lesson.
# Since there are no missing values in our current user_item_matrix, we can use SVD in Numpy.**

# `3.` Now for the tricky part, how do we choose the number of latent features to use?  Running the below cell, you can see that as the number of latent features increases, we obtain a lower error rate on making predictions for the 1 and 0 values in the user-item matrix.  Run the cell below to get an idea of how the accuracy improves as we increase the number of latent features.

# In[101]:


num_latent_feats = np.arange(10,700+10,20)
sum_errs = []

for k in num_latent_feats:
    # restructure with k latent features
    s_new, u_new, vt_new = np.diag(s[:k]), u[:, :k], vt[:k, :]
    
    # take dot product
    user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new))
    
    # compute error for each prediction to actual value
    diffs = np.subtract(user_item_matrix, user_item_est)
    
    # total errors and keep track of them
    err = np.sum(np.sum(np.abs(diffs)))
    sum_errs.append(err)
    
    
plt.plot(num_latent_feats, 1 - np.array(sum_errs)/df.shape[0],color='red');
plt.xlabel('Number of Latent Features');
plt.ylabel('Accuracy');
plt.title('Accuracy vs. Number of Latent Features');


# `4.` From the above, we can't really be sure how many features to use, because simply having a better way to predict the 1's and 0's of the matrix doesn't exactly give us an indication of if we are able to make good recommendations.  Instead, we might split our dataset into a training and test set of data, as shown in the cell below.  
# 
# Use the code from question 3 to understand the impact on accuracy of the training and test sets of data with different numbers of latent features. Using the split below: 
# 
# * How many users can we make predictions for in the test set?  
# * How many users are we not able to make predictions for because of the cold start problem?
# * How many articles can we make predictions for in the test set?  
# * How many articles are we not able to make predictions for because of the cold start problem?

# In[102]:


df_train = df.head(40000)
df_test = df.tail(5993)

def create_test_and_train_user_item(df_train, df_test):
    '''
    INPUT:
    df_train - training dataframe
    df_test - test dataframe
    
    OUTPUT:
    user_item_train - a user-item matrix of the training dataframe 
                      (unique users for each row and unique articles for each column)
    user_item_test - a user-item matrix of the testing dataframe 
                    (unique users for each row and unique articles for each column)
    test_idx - all of the test user ids
    test_arts - all of the test article ids
    
    '''
    # Your code here
    
    # User-item matrix of the training dataframe
    user_item_train = create_user_item_matrix(df_train)
    
    # User-item matrix of the testing dataframe 
    user_item_test = create_user_item_matrix(df_test)
    
    # Revise user_item_test to only include users and articles that are also found in user_item_train to avoid the cold start problem
    # List for train & test data - rows and columns values
    # The rows to be used for test data
    train_idx = set(user_item_train.index)
    test_idx = set(user_item_test.index)
    common_rows = train_idx.intersection(test_idx)
    
    # The columns to be used for test data
    train_arts = set(user_item_train.columns)
    test_arts = set(user_item_test.columns)
    common_cols = train_arts.intersection(test_arts)
       
    user_item_test = user_item_test.loc[common_rows, common_cols]
    
    return user_item_train, user_item_test, test_idx, test_arts

user_item_train, user_item_test, test_idx, test_arts = create_test_and_train_user_item(df_train, df_test)


# In[103]:


# Replace the values in the dictionary below
a = 662 
b = 574 
c = 20 
d = 0 


sol_4_dict = {
    'How many users can we make predictions for in the test set?': c, 
    'How many users in the test set are we not able to make predictions for because of the cold start problem?': a, 
    'How many movies can we make predictions for in the test set?': b,
    'How many movies in the test set are we not able to make predictions for because of the cold start problem?': d
}

t.sol_4_test(sol_4_dict)


# `5.` Now use the **user_item_train** dataset from above to find U, S, and V transpose using SVD. Then find the subset of rows in the **user_item_test** dataset that you can predict using this matrix decomposition with different numbers of latent features to see how many features makes sense to keep based on the accuracy on the test data. This will require combining what was done in questions `2` - `4`.
# 
# Use the cells below to explore how well SVD works towards making predictions for recommendations on the test data.  

# In[104]:


# fit SVD on the user_item_train matrix
u_train, s_train, vt_train =np.linalg.svd(user_item_train) # fit svd similar to above then use the cells below


# In[105]:


# Use these cells to see how well you can use the training 
# decomposition to predict on test data
row_idx = user_item_train.index.isin(test_idx)
col_idx = user_item_train.columns.isin(test_arts)

u_test = u_train[row_idx, :]
vt_test = vt_train[:, col_idx]


# In[54]:


u_test.shape, vt_test.shape


# In[107]:


num_latent_feats = np.arange(10,700+10,20)
sum_errs_train = []
sum_errs_test = []

for k in num_latent_feats:
    
    # restructure with k latent features
    s_train_new, u_train_new, vt_train_new = np.diag(s_train[:k]), u_train[:, :k], vt_train[:k, :]
    u_test_new, vt_test_new = u_test[:, :k], vt_test[:k, :]
    
    # take dot product
    user_item_train_est = np.around(np.dot(np.dot(u_train_new, s_train_new), vt_train_new))
    user_item_test_ets = np.around(np.dot(np.dot(u_test_new, s_train_new), vt_test_new))
    
    # Calculate the error of each prediction with the true value
    diffs_train = np.subtract(user_item_train, user_item_train_est)
    diffs_test = np.subtract(user_item_test, user_item_test_ets)
    
    # Total Error
    err_train = np.sum(np.sum(np.abs(diffs_train)))
    err_test = np.sum(np.sum(np.abs(diffs_test)))
    
    sum_errs_train.append(err_train)
    sum_errs_test.append(err_test)


# In[108]:


plt.plot(num_latent_feats, 1 - np.array(sum_errs_train)/(user_item_train.shape[0]*user_item_test.shape[1]), label='Train',color='green');
plt.plot(num_latent_feats, 1 - np.array(sum_errs_test)/(user_item_test.shape[0]*user_item_test.shape[1]), label='Test',color='orange');

plt.xlabel('Number of Latent Features');
plt.ylabel('Accuracy');
plt.title('Accuracy vs. Number of Latent Features');
plt.legend();


# `6.` Use the cell below to comment on the results you found in the previous question. Given the circumstances of your results, discuss what you might do to determine if the recommendations you make with any of the above recommendation systems are an improvement to how users currently find articles? 

# **Response is.**
# 
# 
# - **From the above graph we can infer that the accuracy of training data set increases with number of latent features while the accuracy of test data gradually decreases this is because our more latent features will just act as a noise and result in poor accuracy.This could be because of the imbalanced dataset. It seems like 100 latent features best describe the variability of the dataset and best suited for building recommendations.**
# 
# - **We might combine content-based recommendation with other recommendation methods. We can define categories of each article using information from df_content and combine it with rank-based recommendation, and use it as filter for user-user based collaborative filtering.**
# 
# - **Another explanation is the binary nature (ones and zeros) of the interaction between users and articles, which is imbalanced compared to the values associated to a rating system.**
# 
# - **An alternative to the offline method previously used is an online approach to determine the usefulness of different recommendation systems. For example, we could use an A/B test to measure the effectiveness of a rank based recommendation system against the matrix recommendation system. We would separate users using cookies into equal number groups. Group A would use the rank based recommendations while group B would use the matrix based recommendations.**
# 

# <a id='conclusions'></a>
# ### Extras
# Using your workbook, you could now save your recommendations for each user, develop a class to make new predictions and update your results, and make a flask app to deploy your results.  These tasks are beyond what is required for this project.  However, from what you learned in the lessons, you certainly capable of taking these tasks on to improve upon your work here!
# 
# 
# ## Conclusion
# 
# > Congratulations!  You have reached the end of the Recommendations with IBM project! 
# 
# > **Tip**: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the [rubric](https://review.udacity.com/#!/rubrics/2322/view). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
# 
# 
# ## Directions to Submit
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations! 

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Recommendations_with_IBM.ipynb'])

