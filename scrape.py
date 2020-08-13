'''
This file is to scrape webpages from the r/complaints subreddit using the PRAW API.
'''

import praw
import pandas as pd

# authenticate
reddit = praw.Reddit(client_id='REDACTED',
                     client_secret='REDACTED',
                     password='reddit',
                     user_agent='complainer by /u/REDACTED_USERNAME',
                     username='REDACTED')

# id - id of submission
# score - number of upvotes on submission
# title - title of the post
# selftext - body text of the post
# created_utc - Time the submission was created, represented in Unix time


# get top 1000 newest posts from the 'complaints' subreddit
new_posts = reddit.subreddit('complaints').new(limit=1000)

# turn relevant information from 'complaints' posts into a Pandas dataframe
posts = []
for post in new_posts:
    posts.append([post.id, post.title, post.selftext, post.created_utc, post.score])
posts = pd.DataFrame(posts,columns=['id', 'title', 'body', 'created_utc', 'score'])
print(posts)
posts.to_csv('complaints_new_13082020.csv') #note that file name is date in UTC

# import pprint
# for post in hot_posts:
#     print(post.title)
#     pprint.pprint(vars(post))
