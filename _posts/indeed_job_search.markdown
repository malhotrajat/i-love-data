---
title:  "Scraping Indeed.com to search for relevant jobs"
date:   2018-02-21
tags: [scraping]

header:
  image: "i-love-data/assets/images/indeed_job_search/jobsearch.jpg"

excerpt: "Indeed.com, Scraping, Jobs"
---

# Motivation

I am currently in the search for a job and I see that one of the most time-consuming processes is actually searching for a, well, job. Finding all the relevant positions in all the companies across all cities sounds impossible. So, the motivation for creating a tool like this was simple. Search for job roles being posted daily and evaluate (at least very basically) whether the job descriptions match with my skillset. For my analysis, I used Indeed.com, which is a major job aggregator and is used by many people daily.

# Overview

This tool would go through all the jobs (type of jobs should be mentioned by the user) in a particular city or cities and add those ones to a list that require particular skills that match with the user's skillset. All of the code is written in the form of functions in order to change parameters,search terms or the number of pages we want to search.

# Scoring a job
For my analysis, I would be using the "data scientist" position since that is what I am interested in. Evaluating any data science job can't be simple and every company has a different definition for who a "data scientist" is. But, we can evaluate it superficially and get to know whether a few keywords exist or not. For my purposes, I'd be using the following keywords:

#### R

#### SQL

#### Python

#### Hadoop

#### Tableau

These are the skills that I possess and therefore if any job description contains any of these words, I want to know about it. Obviously, I won't be applying to every job that contains any of these keywords, but a consolidated list of jobs I COULD apply to is a good start.

```python
#importing the necessary libraries
import requests
import bs4
import re
import time
import smtplib

#Defining a function that would score a job based on the specific keywords you want the job description to contain
def job_score(url):
    
    #obtaining the html script
    htmlcomplete = requests.get(url)
    htmlcontent = bs4.BeautifulSoup(htmlcomplete.content, 'lxml')
    htmlbody = htmlcontent('body')[0]
    
    #findin all the keywords
    r = len(re.findall('R[\,\.]', htmlbody.text))
    sql = htmlbody.text.count('sql')+htmlbody.text.count('Sql')+htmlbody.text.count('SQL')
    python = htmlbody.text.count('python')+htmlbody.text.count('Python')
    hadoop = htmlbody.text.count('hadoop')+htmlbody.text.count('Hadoop')+htmlbody.text.count('HADOOP')
    tableau = htmlbody.text.count('tableau')+htmlbody.text.count('Tableau')
    total=r+python+sql+hadoop+tableau
    print ('R count:', r, ',','Python count:', python, ',','SQL count:', sql, ',','Hadoop count:', hadoop, ',','Tableau count:', tableau, ',',)
    return total
```

# Evaluating an example job

Let's evaluate this "Data Insights Analyst" job from Homeaway.

