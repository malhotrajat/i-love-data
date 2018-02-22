---
title:  "Scraping Indeed.com for jobs"
date:   2018-02-22
tags: [scraping]

header:
  image: "assets/images/indeed_scraping/indeed.jpg"
 
excerpt: "Indeed.com, scraping, job descriptions"
---

# **Motivation**

I am currently in the search for a job and I see that one of the most time-consuming processes is actually searching for a, well, job. Finding all the relevant positions in all the companies across all cities sounds impossible. So, the motivation for creating a tool like this was simple. Search for job roles being posted daily and evaluate (at least very basically) whether the job descriptions match with my skillset. For my analysis, I used Indeed.com, which is a major job aggregator and is used by many people daily.

# **Overview**

This tool would go through all the jobs (type of jobs should be mentioned by the user) in a particular city or cities and add those ones to a list that require particular skills that match with the user's skillset.
All of the code is written in the form of functions in order to change parameters,search terms or the number of pages we want to search.

# **Scoring a job**

For my analysis, I would be using the "data scientist" position since that is what I am interested in.
Evaluating any data science job can't be simple and every company has a different definition for who a "data scientist" is. But, we can evaluate it superficially and get to know whether a few keywords exist or not. 
For my purposes, I'd be using the following keywords:

**R**

**SQL**

**Python**

**Hadoop**

**Tableau**

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

# **Evaluating an example job**

Let's evaluate this "Data Insights Analyst" job from Homeaway.


```python
job_score('https://www.indeed.com/viewjob?jk=29d57706cae9885e&tk=1c6l78ddmafhgf15&from=serp&vjs=3')
```

    R count: 1 , Python count: 1 , SQL count: 2 , Hadoop count: 1 , Tableau count: 1 ,
    




    6



# **Looking at the HTML script behind the scenes**

To extract any kind of information using the HTML script, we need to know how it is structured and where is the relevant information (needed by us) located in the script)


```python
#This section of the code lets you see the html script so that you can understand the structure and what information can be extracted from which part of the script 
URL = 'https://www.indeed.com/jobs?q=data&l=Austin%2C+TX&sort=date'

#conducting a request of the stated URL above:
complete = requests.get(URL)

#specifying a desired format of “page” using the html parser - this allows python to read the various components of the page, rather than treating it as one long string.
content = bs4.BeautifulSoup(complete.text, 'html.parser')

#printing soup in a more structured tree format that makes for easier reading
print(content.prettify())
```

    <!DOCTYPE html>
    <html lang="en">
     <head>
      <meta content="text/html;charset=utf-8" http-equiv="content-type"/>
      <script src="/s/044574d/en_US.js" type="text/javascript">
      </script>
      <link href="/s/ecdfb5e/jobsearch_all.css" rel="stylesheet" type="text/css"/>
      <link href="http://rss.indeed.com/rss?q=data&amp;l=Austin%2C+TX&amp;sort=date" rel="alternate" title="Data Jobs, Employment in Austin, TX" type="application/rss+xml"/>
      <link href="/m/jobs?q=data&amp;l=Austin%2C+TX&amp;sort=date" media="only screen and (max-width: 640px)" rel="alternate"/>
      <link href="/m/jobs?q=data&amp;l=Austin%2C+TX&amp;sort=date" media="handheld" rel="alternate"/>
      <script type="text/javascript">
       if (typeof window['closureReadyCallbacks'] == 'undefined') {
            window['closureReadyCallbacks'] = [];
        }
.
.
.
.
.
 
            </a>
            <div class=" row result" data-jk="0829198f649e9c08" data-tn-component="organicJob" data-tu="" id="p_0829198f649e9c08">
             <h2 class="jobtitle" id="jl_0829198f649e9c08">
              <a class="turnstileLink" data-tn-element="jobTitle" href="/rc/clk?jk=0829198f649e9c08&amp;fccid=7c30762e902763ee&amp;vjs=3" onclick="setRefineByCookie([]); return rclk(this,jobmap[0],true,0);" onmousedown="return rclk(this,jobmap[0],0);" rel="noopener nofollow" target="_blank" title="Data Entry Associate">
               <b>
                Data
               </b>
               Entry Associate
              </a>
             </h2>
.
.
.
.

# **Extracting Job Data**

The next step after defining a job scoring function, is to define a function that gets you all the relevant information from the HTML script for all the jobs on a single page.
We look for non-sponsored or organic jobs and extract the attributes from those. These attributes contain a lot of information that I don't need but we will just let them be. What we do need are the following things:

**Name of the company**

**Date when the job was posted**

**Title**

**Hyperlink to the job**


```python
def jobdata(url):
    htmlcomplete2 = requests.get(url)
    htmlcontent2 = bs4.BeautifulSoup(htmlcomplete2.content, 'lxml')
    #only getting the tags for organic job postings and not the ones thatare sponsored
    tags = htmlcontent2.find_all('div', {'data-tn-component' : "organicJob"})
    #getting the list of companies that have the organic job posting tags
    companies = [x.span.text for x in tags]
    #extracting the features like the company name, complete link, date, etc.
    attributes = [x.h2.a.attrs for x in tags]
    dates = [x.find_all('span', {'class':'date'}) for x in tags]
    
    # update attributes dictionaries with company name and date posted
    [attributes[i].update({'company': companies[i].strip()}) for i, x in enumerate(attributes)]
    [attributes[i].update({'date posted': dates[i][0].text.strip()}) for i, x in enumerate(attributes)]
    return attributes
```

Now we can look at a sample of the attribute dictionary for the first job on the page I have specified.


```python
jobdata('https://www.indeed.com/jobs?q=data&l=Austin%2C+TX&sort=date')[0]
```




    {'class': ['turnstileLink'],
     'company': 'Absolute Software',
     'data-tn-element': 'jobTitle',
     'date posted': 'Just posted',
     'href': '/rc/clk?jk=0829198f649e9c08&fccid=7c30762e902763ee&vjs=3',
     'onclick': 'setRefineByCookie([]); return rclk(this,jobmap[0],true,0);',
     'onmousedown': 'return rclk(this,jobmap[0],0);',
     'rel': ['noopener', 'nofollow'],
     'target': '_blank',
     'title': 'Data Entry Associate'}



# **Defining a list of cities**

We define a list of cities that we want to search for jobs in


```python
#defining a list of cities you want to search jobs in
citylist = ['New+York','Chicago', 'Austin']#, 'San+Francisco', 'Seattle', 'Los+Angeles', 'Philadelphia', 'Atlanta', 'Dallas', 'Pittsburgh', 'Portland', 'Phoenix', 'Denver', 'Houston', 'Miami', 'Washington+DC', 'Boulder']
```

# **Searching for and Scoring all new jobs**

I can now loop through Indeed.com and apply the functions defined above to every page. 


```python
#defining a list to store all the relevant jobs
newjobslist = []

#defining a new function to go through all the jobs posted in the last 'n' days for a specific role
#essentially looping over 2 
def newjobs(daysago = 1, startingpage = 0, pagelimit = 20, position = 'data+scientist'):
    for city in citylist:
        indeed_url = 'http://www.indeed.com/jobs?q={0}&l={1}&sort=date&start='.format(position, city)
        
        
        for i in range(startingpage, startingpage + pagelimit):
            print ('URL:', str(indeed_url + str(i*10)), '\n')
        
            attributes = jobdata(indeed_url + str(i*10))
            
            for j in range(0, len(attributes)):
                href = attributes[j]['href']
                title = attributes[j]['title']
                company = attributes[j]['company']
                date_posted = attributes[j]['date posted']
                
                print (repr(company),',', repr(title),',', repr(date_posted))
                
                evaluation = job_score('http://indeed.com' + href)
                
                if evaluation >= 1:
                    newjobslist.append('{0}, {1}, {2}, {3}'.format(company, title, city, 'http://indeed.com' + href))
                    
                print ('\n')
                
            time.sleep(1)
           
    newjobsstring = '\n\n'.join(newjobslist)
    return newjobsstring
```

# **Sending an email to myself**

I can now send an email to myself using the smtplib library. 


```python
def emailme(from_addr = '****', to_addr = '****', subject = 'Daily Data Science Jobs Update Scraped from Indeed', text = None):
    
    message = 'Subject: {0}\n\nJobs: {1}'.format(subject, text)

    # login information
    username = '****'
    password = '****'
    
    # send the message
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()
    server.login(username, password)
    server.sendmail(from_addr, to_addr, message)
    server.quit()
    print ('Please check your mail')
```


```python
def main():
    print ('Searching for jobs...')

    starting_page = 0
    page_limit = 2
    datascientist = newjobs(position = 'data+scientist', startingpage = starting_page, pagelimit = page_limit)
    emailme(text = datascientist)
```


```python
main()
```

    Searching for jobs...
    URL: http://www.indeed.com/jobs?q=data+scientist&l=New+York&sort=date&start=0 
    
    'J.Crew Group, Inc.' , 'Customer Analytics Manager' , 'Just posted'
    R count: 0 , Python count: 1 , SQL count: 1 , Hadoop count: 0 , Tableau count: 0 ,
    .
    .
    .
    .
    
    'Invenio Marketing Solutions' , 'Inside Sales Representative - Mediacom' , '4 days ago'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    Please check your mail.
    
Here's a snapshot of the email you would receive:
![png](/assets/images/indeed_scraping/indeed_scraping.PNG?raw=True)
    

# **Ending Remarks**

This was a pretty interesting project to complete and a lot of fun too. I am sure there are many improvements that can be made and it can give more information too. More changes may be made in future.
