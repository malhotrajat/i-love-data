
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
    
        function call_when_jsall_loaded(cb) {
            if (window['closureReady']) {
                cb();
            } else {
                window['closureReadyCallbacks'].push(cb);
            }
        }
      </script>
      <meta content="1" name="ppstriptst"/>
      <script src="/s/7ef6be0/jobsearch-all-compiled.js" type="text/javascript">
      </script>
      <script type="text/javascript">
       var searchUID = '1c6u39dhta24qbca';
    var tk = '1c6u39dhta24qbca';
    
    var loggedIn = false;
    var dcmPayload = 'jobse0;jobal0;viewj0;savej0;6927552';
    var myindeed = true;
    var userEmail = '';
    var tellFriendEmail = '';
    var globalLoginURL = 'https:\/\/www.indeed.com\/account\/login?dest=%2Fjobs%3Fq%3Ddata%26l%3DAustin%2C%2BTX%26sort%3Ddate';
    var globalRegisterURL = 'https:\/\/www.indeed.com\/account\/register?dest=%2Fjobs%3Fq%3Ddata%26l%3DAustin%2C%2BTX%26sort%3Ddate';
    var searchKey = 'c8f52df9be67ff24';
    var searchState = 'q=data&amp;l=Austin%2C+TX&amp;sort=date';
    var searchQS = 'q=data&l=Austin,+TX&sort=date';
    var eventType = 'jobsearch';
    var locale = 'en_US';
    function clk(id) { var a = document.getElementById(id); var hr = a.href; var si = a.href.indexOf('&jsa='); if (si > 0) return; var jsh = hr + '&tk=1c6u39dhta24qbca&jsa=6621'; a.href = jsh; }
    function sjomd(id) { var a = document.getElementById(id); var hr = a.href; var ocs = hr.indexOf('&oc=1'); if (ocs < 0) return; var oce = ocs + 5; a.href = hr.substring(0, ocs) + hr.substring(oce); }
    function sjoc(id, sal) { var a = document.getElementById(id); a.href = a.href + '&oc=1&sal='+sal; }
    function ptk(st,p) {document.cookie = 'PTK="tk=1c6u39dhta24qbca&type=jobsearch&subtype='+st+(p?'&'+p:'')+'"; path=/';}
    function rbptk(st, c, p) { ptk(st, 'cat='+c+(p?'&p='+p:''));}
      </script>
      <script type="text/javascript">
       function loadJSAsync( ) {
    		for ( var i = 0; i < arguments.length; i++ ) {
    			var url = arguments[i];
    			(function() {
    				var s = document.createElement("script"), el = document.getElementsByTagName("script")[0];
    				s.async = true;
    				s.src = url;
    				el.parentNode.insertBefore(s, el);
    			})();
    		}
    	}
      </script>
      <meta content="5,474 Data jobs available in Austin, TX on Indeed.com. Court Clerk, Management Analyst, Market Researcher and more!" name="description"/>
      <meta content="Data Jobs, Employment in Austin, TX, Austin, TX careers, Austin, TX employment, Austin, TX job listings, Austin, TX job search, Austin, TX search engine, work in Austin, TX" name="keywords"/>
      <meta content="origin-when-cross-origin" name="referrer"/>
      <link href="/q-Data-l-Austin,-TX-jobs.html" rel="canonical"/>
      <link href="/jobs?q=data&amp;l=Austin%2C+TX&amp;sort=date&amp;start=10" rel="next">
       <style type="text/css">
        #recPromoDisplay { margin-bottom: 3em;margin-left: 0.5em; }
            #recPromoDisplayPageLast { font-size: 16px; margin: 1.5em 0; }
       </style>
       <script type="text/javascript">
        var dcmPayload = 'jobse0;jobal0;viewj0;savej0;6927552';
        var indeedCsrfToken = 'oSaIDJrvvXmUCZda0NfpXIEXDVc6bpGb';
        var hashedCsrfToken = 'fe8488f22fdc0b8f4a571f572ebdc656';
       </script>
       <style type="text/css">
        .jasxcustomfonttst-useCustomHostedFontFullPage *{font-family:"Open Sans", sans-serif !important}.jasxcustomfonttst-useLato *{font-family:"Lato", sans-serif !important}.jasxcustomfonttst-useFira *{font-family:"Fira Sans", sans-serif !important}.jasxcustomfonttst-useGibson *{font-family:"Gibson", sans-serif !important}.jasxcustomfonttst-useAvenir *{font-family:"Avenir Next", sans-serif !important}#accessibilityBanner{position:absolute;left:-10000%;height:50px;width:100%;color:#000;font-size:13px;font-family:Arial;background-color:#F2F2F2;border-width:1px;border-color:#E6E6E6;line-height:50px}#accessibilityText{position:relative;left:12px;white-space:nowrap}#accessibilityClose{position:absolute;right:12px}.icl-Button{box-sizing:border-box;display:inline-block;vertical-align:middle;font-family:Avenir Next,Arial,Roboto,Noto,sans-serif;font-size:1.125rem;line-height:1.8rem;font-weight:700;text-decoration:none;text-overflow:ellipsis;white-space:nowrap;filter:progid:DXImageTransform.Microsoft.dropshadow(OffX=0,OffY=-1,Color=#e80f2299,Positive=true);-webkit-appearance:none;-moz-appearance:none;appearance:none;overflow:hidden;-webkit-user-select:none;-moz-user-select:none;-ms-user-select:none;user-select:none;-webkit-touch-callout:none;-webkit-highlight:none;-webkit-tap-highlight-color:transparent;color:#f8f8f9;filter:progid:DXImageTransform.Microsoft.gradient(startColorstr="#ff6598fe",endColorstr="#ff3c69e0",GradientType=0)}[dir] .icl-Button{padding:0.5rem 1.125rem;text-align:center;background-repeat:repeat-x;border:1px solid;border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,0.2);cursor:pointer;text-shadow:0 -1px #0f2299;background-color:#5585f2;background-image:linear-gradient(180deg, #6598ff, #2e5ad7);border-color:#1642bb;border-bottom-color:#1642bb}[dir=ltr] .icl-Button{margin:12px 12px 12px 0}[dir=rtl] .icl-Button{margin:12px 0 12px 12px}[dir] .icl-Button::-moz-focus-inner{border:0}[dir] .icl-Button:hover{background-image:none;box-shadow:0 1px 5px rgba(0,0,0,0.4)}.icl-Button:active{outline:none}[dir] .icl-Button:active{background-image:none;box-shadow:inset 0 2px 4px rgba(0,0,0,0.15),0 1px 2px rgba(0,0,0,0.05)}.icl-Button:disabled{opacity:.65}[dir] .icl-Button:disabled{background-image:none;box-shadow:none;cursor:default}[dir] .icl-Button:disabled:hover{box-shadow:none}.icl-Button:hover{text-decoration:none}[dir] .icl-Button:active,[dir] .icl-Button:disabled,[dir] .icl-Button:hover{background-color:#2e5ad7}.icl-Button--primary{display:inline-block;vertical-align:middle;font-family:Avenir Next,Arial,Roboto,Noto,sans-serif;font-size:1.125rem;line-height:1.8rem;font-weight:700;text-decoration:none;text-overflow:ellipsis;white-space:nowrap;filter:progid:DXImageTransform.Microsoft.dropshadow(OffX=0,OffY=-1,Color=#e80f2299,Positive=true);-webkit-appearance:none;-moz-appearance:none;appearance:none;overflow:hidden;-webkit-user-select:none;-moz-user-select:none;-ms-user-select:none;user-select:none;-webkit-touch-callout:none;-webkit-highlight:none;-webkit-tap-highlight-color:transparent;color:#f8f8f9;filter:progid:DXImageTransform.Microsoft.gradient(startColorstr="#ff6598fe",endColorstr="#ff3c69e0",GradientType=0)}[dir] .icl-Button--primary{padding:0.5rem 1.125rem;text-align:center;background-repeat:repeat-x;border:1px solid;border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,0.2);cursor:pointer;text-shadow:0 -1px #0f2299;background-color:#5585f2;background-image:linear-gradient(180deg, #6598ff, #2e5ad7);border-color:#1642bb;border-bottom-color:#1642bb}[dir=ltr] .icl-Button--primary{margin:12px 12px 12px 0}[dir=rtl] .icl-Button--primary{margin:12px 0 12px 12px}[dir] .icl-Button--primary::-moz-focus-inner{border:0}[dir] .icl-Button--primary:hover{background-image:none;box-shadow:0 1px 5px rgba(0,0,0,0.4)}.icl-Button--primary:active{outline:none}[dir] .icl-Button--primary:active{background-image:none;box-shadow:inset 0 2px 4px rgba(0,0,0,0.15),0 1px 2px rgba(0,0,0,0.05)}.icl-Button--primary:disabled{opacity:.65}[dir] .icl-Button--primary:disabled{background-image:none;box-shadow:none;cursor:default}[dir] .icl-Button--primary:disabled:hover{box-shadow:none}.icl-Button--primary:hover{text-decoration:none}[dir] .icl-Button--primary:active,[dir] .icl-Button--primary:disabled,[dir] .icl-Button--primary:hover{background-color:#2e5ad7}.icl-Button--secondary{display:inline-block;vertical-align:middle;font-family:Avenir Next,Arial,Roboto,Noto,sans-serif;font-size:1.125rem;line-height:1.8rem;font-weight:700;text-decoration:none;text-overflow:ellipsis;white-space:nowrap;filter:progid:DXImageTransform.Microsoft.dropshadow(OffX=0,OffY=-1,Color=#e80f2299,Positive=true);-webkit-appearance:none;-moz-appearance:none;appearance:none;overflow:hidden;-webkit-user-select:none;-moz-user-select:none;-ms-user-select:none;user-select:none;-webkit-touch-callout:none;-webkit-highlight:none;-webkit-tap-highlight-color:transparent;color:#333;filter:progid:DXImageTransform.Microsoft.gradient(startColorstr="#fff8f8f9",endColorstr="#ffe6e6e6",GradientType=0)}[dir] .icl-Button--secondary{padding:0.5rem 1.125rem;text-align:center;background-repeat:repeat-x;border:1px solid;border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,0.2);cursor:pointer;text-shadow:0 1px #fff;background-color:#d9d9e2;background-image:linear-gradient(180deg, #f8f8f9, #d9d9e2);border-color:#9a99ac;border-bottom-color:#a2a2a2}[dir=ltr] .icl-Button--secondary{margin:12px 12px 12px 0}[dir=rtl] .icl-Button--secondary{margin:12px 0 12px 12px}[dir] .icl-Button--secondary::-moz-focus-inner{border:0}[dir] .icl-Button--secondary:hover{background-image:none;box-shadow:0 1px 5px rgba(0,0,0,0.4)}.icl-Button--secondary:active{outline:none}[dir] .icl-Button--secondary:active{background-image:none;box-shadow:inset 0 2px 4px rgba(0,0,0,0.15),0 1px 2px rgba(0,0,0,0.05)}.icl-Button--secondary:disabled{opacity:.65}[dir] .icl-Button--secondary:disabled{background-image:none;box-shadow:none;cursor:default}[dir] .icl-Button--secondary:disabled:hover{box-shadow:none}.icl-Button--secondary:hover{text-decoration:none}[dir] .icl-Button--secondary:active,[dir] .icl-Button--secondary:disabled,[dir] .icl-Button--secondary:hover{background-color:#f8f8f9}.icl-Button--special{display:inline-block;vertical-align:middle;font-family:Avenir Next,Arial,Roboto,Noto,sans-serif;font-size:1.125rem;line-height:1.8rem;font-weight:700;text-decoration:none;text-overflow:ellipsis;white-space:nowrap;filter:progid:DXImageTransform.Microsoft.dropshadow(OffX=0,OffY=-1,Color=#e80f2299,Positive=true);-webkit-appearance:none;-moz-appearance:none;appearance:none;overflow:hidden;-webkit-user-select:none;-moz-user-select:none;-ms-user-select:none;user-select:none;-webkit-touch-callout:none;-webkit-highlight:none;-webkit-tap-highlight-color:transparent;color:#f8f8f9;filter:progid:DXImageTransform.Microsoft.gradient(startColorstr="#ff6598fe",endColorstr="#ff3c69e0",GradientType=0)}[dir] .icl-Button--special{padding:0.5rem 1.125rem;text-align:center;background-repeat:repeat-x;border:1px solid;border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,0.2);cursor:pointer;text-shadow:0 -1px #000;background-color:#f14200;background-image:linear-gradient(180deg, #f60, #f14200);border-color:#ba3200;border-bottom-color:#ba3200}[dir=ltr] .icl-Button--special{margin:12px 12px 12px 0}[dir=rtl] .icl-Button--special{margin:12px 0 12px 12px}[dir] .icl-Button--special::-moz-focus-inner{border:0}[dir] .icl-Button--special:hover{background-image:none;box-shadow:0 1px 5px rgba(0,0,0,0.4)}.icl-Button--special:active{outline:none}[dir] .icl-Button--special:active{background-image:none;box-shadow:inset 0 2px 4px rgba(0,0,0,0.15),0 1px 2px rgba(0,0,0,0.05)}.icl-Button--special:disabled{opacity:.65}[dir] .icl-Button--special:disabled{background-image:none;box-shadow:none;cursor:default}[dir] .icl-Button--special:disabled:hover{box-shadow:none}.icl-Button--special:hover{text-decoration:none}[dir] .icl-Button--special:active,[dir] .icl-Button--special:disabled,[dir] .icl-Button--special:hover{background-color:#f14200}.icl-Button--danger{display:inline-block;vertical-align:middle;font-family:Avenir Next,Arial,Roboto,Noto,sans-serif;font-size:1.125rem;line-height:1.8rem;font-weight:700;text-decoration:none;text-overflow:ellipsis;white-space:nowrap;filter:progid:DXImageTransform.Microsoft.dropshadow(OffX=0,OffY=-1,Color=#e80f2299,Positive=true);-webkit-appearance:none;-moz-appearance:none;appearance:none;overflow:hidden;-webkit-user-select:none;-moz-user-select:none;-ms-user-select:none;user-select:none;-webkit-touch-callout:none;-webkit-highlight:none;-webkit-tap-highlight-color:transparent;color:#f8f8f9;filter:progid:DXImageTransform.Microsoft.gradient(startColorstr="#ff6598fe",endColorstr="#ff3c69e0",GradientType=0)}[dir] .icl-Button--danger{padding:0.5rem 1.125rem;text-align:center;background-repeat:repeat-x;border:1px solid;border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,0.2);cursor:pointer;text-shadow:0 -1px #000;background-color:#b01825;background-image:linear-gradient(180deg, #d1787f, #b01825);border-color:#83121b;border-bottom-color:#83121b}[dir=ltr] .icl-Button--danger{margin:12px 12px 12px 0}[dir=rtl] .icl-Button--danger{margin:12px 0 12px 12px}[dir] .icl-Button--danger::-moz-focus-inner{border:0}[dir] .icl-Button--danger:hover{background-image:none;box-shadow:0 1px 5px rgba(0,0,0,0.4)}.icl-Button--danger:active{outline:none}[dir] .icl-Button--danger:active{background-image:none;box-shadow:inset 0 2px 4px rgba(0,0,0,0.15),0 1px 2px rgba(0,0,0,0.05)}.icl-Button--danger:disabled{opacity:.65}[dir] .icl-Button--danger:disabled{background-image:none;box-shadow:none;cursor:default}[dir] .icl-Button--danger:disabled:hover{box-shadow:none}.icl-Button--danger:hover{text-decoration:none}[dir] .icl-Button--danger:active,[dir] .icl-Button--danger:disabled,[dir] .icl-Button--danger:hover{background-color:#b01825}.icl-Button--working{display:inline-block;vertical-align:middle;font-family:Avenir Next,Arial,Roboto,Noto,sans-serif;font-size:1.125rem;line-height:1.8rem;font-weight:700;text-decoration:none;text-overflow:ellipsis;white-space:nowrap;filter:progid:DXImageTransform.Microsoft.dropshadow(OffX=0,OffY=-1,Color=#e80f2299,Positive=true);-webkit-appearance:none;-moz-appearance:none;appearance:none;overflow:hidden;-webkit-user-select:none;-moz-user-select:none;-ms-user-select:none;user-select:none;-webkit-touch-callout:none;-webkit-highlight:none;-webkit-tap-highlight-color:transparent}[dir] .icl-Button--working{padding:0.5rem 1.125rem;text-align:center;background-repeat:repeat-x;border:1px solid;border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,0.2);cursor:pointer;border-color:#9a99ac;border-bottom-color:#a2a2a2}[dir=ltr] .icl-Button--working{margin:12px 12px 12px 0}[dir=rtl] .icl-Button--working{margin:12px 0 12px 12px}[dir] .icl-Button--working::-moz-focus-inner{border:0}[dir] .icl-Button--working:hover{background-image:none;box-shadow:0 1px 5px rgba(0,0,0,0.4)}.icl-Button--working:active{outline:none}[dir] .icl-Button--working:active{background-image:none;box-shadow:inset 0 2px 4px rgba(0,0,0,0.15),0 1px 2px rgba(0,0,0,0.05)}.icl-Button--working:disabled{opacity:.65}[dir] .icl-Button--working:disabled{background-image:none;box-shadow:none;cursor:default}[dir] .icl-Button--working:disabled:hover{box-shadow:none}.icl-Button--working:hover{text-decoration:none}[dir] .icl-Button--working:disabled{background-color:#f8f8f9}.icl-Button--transparent{display:inline-block;vertical-align:middle;font-family:Avenir Next,Arial,Roboto,Noto,sans-serif;font-size:1.125rem;line-height:1.8rem;font-weight:700;text-decoration:none;text-overflow:ellipsis;white-space:nowrap;filter:progid:DXImageTransform.Microsoft.dropshadow(OffX=0,OffY=-1,Color=#e80f2299,Positive=true);-webkit-appearance:none;-moz-appearance:none;appearance:none;overflow:hidden;-webkit-user-select:none;-moz-user-select:none;-ms-user-select:none;user-select:none;-webkit-touch-callout:none;-webkit-highlight:none;-webkit-tap-highlight-color:transparent;color:#00c}[dir] .icl-Button--transparent{padding:0.5rem 1.125rem;text-align:center;background-repeat:repeat-x;border:1px solid;border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,0.2);cursor:pointer;text-shadow:none;background:transparent none;border:none;box-shadow:none}[dir=ltr] .icl-Button--transparent{margin:12px 12px 12px 0}[dir=rtl] .icl-Button--transparent{margin:12px 0 12px 12px}[dir] .icl-Button--transparent::-moz-focus-inner{border:0}[dir] .icl-Button--transparent:hover{background-image:none;box-shadow:0 1px 5px rgba(0,0,0,0.4)}.icl-Button--transparent:active{outline:none}[dir] .icl-Button--transparent:active{background-image:none;box-shadow:inset 0 2px 4px rgba(0,0,0,0.15),0 1px 2px rgba(0,0,0,0.05)}.icl-Button--transparent:disabled{opacity:.65}[dir] .icl-Button--transparent:disabled{background-image:none;box-shadow:none;cursor:default}[dir] .icl-Button--transparent:disabled:hover{box-shadow:none}.icl-Button--transparent:hover{color:#00c;text-decoration:underline}[dir] .icl-Button--transparent:active,[dir] .icl-Button--transparent:hover{box-shadow:0 0 0 transparent}.icl-Button--transparent:disabled:hover{color:#00c}.icl-Button--block{display:block;width:100%;max-width:351px}[dir] .icl-Button--block{margin:12px auto}.icl-Button--md{font-size:.9375rem;font-weight:600}[dir] .icl-Button--md{padding:0.5rem 1rem;border-radius:5px}.icl-Button--sm{font-size:.8125rem;font-weight:500}[dir] .icl-Button--sm{padding:0.325rem 0.8125rem;border-radius:4px}[dir] .icl-Button--group{border-radius:0;box-shadow:0 0 0 transparent}[dir=ltr] .icl-Button--group{float:left;margin:0 0 0 -1px}[dir=rtl] .icl-Button--group{float:right;margin:0 -1px 0 0}[dir] .icl-Button--group:hover{box-shadow:0 0 0 transparent}[dir=ltr] .icl-Button--group:first-child{margin-left:0;border-bottom-left-radius:6px;border-top-left-radius:6px}[dir=rtl] .icl-Button--group:first-child{margin-right:0}[dir=ltr] .icl-Button--group:last-child,[dir=rtl] .icl-Button--group:first-child{border-bottom-right-radius:6px;border-top-right-radius:6px}[dir=rtl] .icl-Button--group:last-child{border-bottom-left-radius:6px;border-top-left-radius:6px}[dir=ltr] .icl-Button--icon,[dir=rtl] .icl-Button--icon{padding-left:10px;padding-right:10px}.icl-Button--responsive{max-width:351px;width:100%}[dir] .icl-Button--responsive{margin:12px 0 0}[dir] .icl-Button--responsive:first-child{margin-top:0}@media only screen and (min-width: 768px){.icl-Button--responsive{width:auto}}
       </style>
       <style type="text/css">
        #resultsCol { padding-top: 0; }
        .searchCount { margin-top: 6px; }
        .showing { padding-top: 9px; padding-bottom: 9px; }
    
        .brdr { height: 1px; overflow: hidden; background-color: #ccc; }
    
        /* Tall window sizes */
        @media only screen and (min-height:780px){
            .showing { padding-bottom: 0; }
        }
    
        /* Wide window sizes */
        @media only screen and (min-width:1125px){
            .brdr  { margin-left: 12px; margin-right: 12px; }
        }
    
        a, a:link, .link, .btn, .btn:hover { text-decoration:none; }
    a:hover, .link:hover { text-decoration:underline; }
    .dya-container a { text-decoration: underline!important; }
       </style>
       <script>
        function onLoadHandler() {
                
                    document.js.reset();
                    jobSeenInit('1c6u39dhta24qbca', [{
                        'jobClassName': 'result',
                        'scanIta': true,
                        'containsSponsored': true,
                        'context': ''
                    }]);
                
                if ( document.radius_update ) { document.radius_update.reset(); }
                
    
                initJobsearchUnloadBeacon('1c6u39dhta24qbca');
            }
    
            initLogInitialUserInteraction('1c6u39dhta24qbca', 'serp');
    
            window.onload = onLoadHandler;
       </script>
       <link href="android-app://com.indeed.android.jobsearch/https/www.indeed.com/m/jobs?q=data&amp;l=Austin%2C+TX&amp;sort=date" rel="alternate"/>
       <title>
        Data Jobs, Employment in Austin, TX | Indeed.com
       </title>
       <style type="text/css">
        .btn,.sg-btn{display:inline-block;padding:9px 15px;border:1px solid #9a99ac;border-bottom-color:#a2a2a2;-webkit-border-radius:6px;-moz-border-radius:6px;-ms-border-radius:6px;-o-border-radius:6px;border-radius:6px;background-color:#D9D9E2;background-image:-moz-linear-gradient(top, #f8f8f9, #D9D9E2);background-image:-webkit-gradient(linear, 0 0, 0 100%, from(#f8f8f9), to(#D9D9E2));background-image:-webkit-linear-gradient(top, #f8f8f9, #D9D9E2);background-image:linear-gradient(to bottom, #f8f8f9, #D9D9E2);background-repeat:repeat-x;-webkit-box-shadow:0 1px 5px rgba(0,0,0,0.2);-moz-box-shadow:0 1px 5px rgba(0,0,0,0.2);-ms-box-shadow:0 1px 5px rgba(0,0,0,0.2);-o-box-shadow:0 1px 5px rgba(0,0,0,0.2);box-shadow:0 1px 5px rgba(0,0,0,0.2);color:#333;vertical-align:middle;text-align:center;text-decoration:none;text-shadow:0 1px #fff;font-weight:700;font-size:16px;font-family:"Helvetica Neue",Helvetica,Arial,"Lucida Grande",sans-serif;line-height:22px;filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#fff8f8f9', endColorstr='#ffe6e6e6', GradientType=0);cursor:pointer;-webkit-user-select:none;-moz-user-select:none;-ms-user-select:none;-o-user-select:none;user-select:none;-webkit-touch-callout:none;-webkit-highlight:none;-webkit-tap-highlight-color:transparent;text-overflow:ellipsis;white-space:nowrap;overflow:hidden}.btn.active,.btn.sg-active,.btn:active,.btn.disabled,.btn.sg-disabled,.btn[disabled],.sg-btn.active,.sg-btn.sg-active,.sg-btn:active,.sg-btn.disabled,.sg-btn.sg-disabled,.sg-btn[disabled]{outline:none;background-color:#f8f8f9;color:#333}.btn:focus,.sg-btn:focus{outline:0;box-shadow:0 0 1px 0 #1642bb;-webkit-transition:box-shadow 0.2s linear;-moz-transition:box-shadow 0.2s linear;transition:box-shadow 0.2s linear}.btn.active,.btn.sg-active,.btn:active,.sg-btn.active,.sg-btn.sg-active,.sg-btn:active{background-color:#f8f8f9;background-image:none;-webkit-box-shadow:inset 0 2px 4px rgba(0,0,0,0.15),0 1px 2px rgba(0,0,0,0.05);-moz-box-shadow:inset 0 2px 4px rgba(0,0,0,0.15),0 1px 2px rgba(0,0,0,0.05);box-shadow:inset 0 2px 4px rgba(0,0,0,0.15),0 1px 2px rgba(0,0,0,0.05)}.btn.disabled,.btn.sg-disabled,.btn[disabled],.sg-btn.disabled,.sg-btn.sg-disabled,.sg-btn[disabled]{background-color:#f8f8f9;background-image:none;-webkit-box-shadow:none;-moz-box-shadow:none;box-shadow:none;opacity:.65;filter:alpha(opacity=65);cursor:default}.btn-primary,.sg-btn-primary{border-color:#1642bb;background-color:#5585f2;background-image:-moz-linear-gradient(top, #6598ff, #2e5ad7);background-image:-webkit-gradient(linear, 0 0, 0 100%, from(#6598ff), to(#2e5ad7));background-image:-webkit-linear-gradient(top, #6598ff, #2e5ad7);background-image:linear-gradient(to bottom, #6598ff, #2e5ad7);background-repeat:repeat-x;color:#F8F8F9;text-shadow:0 -1px #0f2299;-ms-filter:progid:DXImageTransform.Microsoft.dropshadow(OffX=0, OffY=-1, Color=#e80f2299, Positive=true);filter:progid:DXImageTransform.Microsoft.dropshadow(OffX=0, OffY=-1, Color=#e80f2299, Positive=true);filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#ff6598fe', endColorstr='#ff3c69e0', GradientType=0);zoom:1}.btn-primary.active,.btn-primary.sg-active,.btn-primary:active,.btn-primary.disabled,.btn-primary.sg-disabled,.btn-primary[disabled],.sg-btn-primary.active,.sg-btn-primary.sg-active,.sg-btn-primary:active,.sg-btn-primary.disabled,.sg-btn-primary.sg-disabled,.sg-btn-primary[disabled]{background-color:#2e5ad7;color:#F8F8F9}.btn-primary:focus,.sg-btn-primary:focus{box-shadow:0 0 1px 0 #000}.btn-special,.sg-btn-special{border-color:#ba3200;background-color:#5585f2;background-image:-moz-linear-gradient(top, #f60, #f14200);background-image:-webkit-gradient(linear, 0 0, 0 100%, from(#f60), to(#f14200));background-image:-webkit-linear-gradient(top, #f60, #f14200);background-image:linear-gradient(to bottom, #f60, #f14200);background-repeat:repeat-x;color:#F8F8F9;text-shadow:0 -1px #000;-ms-filter:progid:DXImageTransform.Microsoft.dropshadow(OffX=0, OffY=-1, Color=#e80f2299, Positive=true);filter:progid:DXImageTransform.Microsoft.dropshadow(OffX=0, OffY=-1, Color=#e80f2299, Positive=true);filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#ff6598fe', endColorstr='#ff3c69e0', GradientType=0);zoom:1}.btn-special.active,.btn-special.sg-active,.btn-special:active,.btn-special.disabled,.btn-special.sg-disabled,.btn-special[disabled],.sg-btn-special.active,.sg-btn-special.sg-active,.sg-btn-special:active,.sg-btn-special.disabled,.sg-btn-special.sg-disabled,.sg-btn-special[disabled]{background-color:#f14200;color:#F8F8F9}.btn-special:focus,.sg-btn-special:focus{box-shadow:0 0 1px 0 #000}.btn-danger,.sg-btn-danger{border-color:#83121b;background-color:#5585f2;background-image:-moz-linear-gradient(top, #d1787f, #b01825);background-image:-webkit-gradient(linear, 0 0, 0 100%, from(#d1787f), to(#b01825));background-image:-webkit-linear-gradient(top, #d1787f, #b01825);background-image:linear-gradient(to bottom, #d1787f, #b01825);background-repeat:repeat-x;color:#F8F8F9;text-shadow:0 -1px #000;-ms-filter:progid:DXImageTransform.Microsoft.dropshadow(OffX=0, OffY=-1, Color=#e80f2299, Positive=true);filter:progid:DXImageTransform.Microsoft.dropshadow(OffX=0, OffY=-1, Color=#e80f2299, Positive=true);filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#ff6598fe', endColorstr='#ff3c69e0', GradientType=0);zoom:1}.btn-danger.active,.btn-danger.sg-active,.btn-danger:active,.btn-danger.disabled,.btn-danger.sg-disabled,.btn-danger[disabled],.sg-btn-danger.active,.sg-btn-danger.sg-active,.sg-btn-danger:active,.sg-btn-danger.disabled,.sg-btn-danger.sg-disabled,.sg-btn-danger[disabled]{background-color:#b01825;color:#F8F8F9}.btn-danger:focus,.sg-btn-danger:focus{box-shadow:0 0 1px 0 #000}input.btn,input.sg-btn{-webkit-appearance:none}button.btn::-moz-focus-inner,button.sg-btn::-moz-focus-inner{border:0}.btn-sm,.sg-btn-sm,.btn-xs,.sg-bt-xs{padding:6px 12px}.btn-xs,.sg-btn-xs{padding:3px 6px;line-height:15px}.btn-md,.sg-btn-md{padding:6px 6px}.btn-lg,.sg-btn-lg{padding:9px 18px;border-radius:6px;font-size:18px}.btn-block,.sg-btn-block{display:block;margin:9px auto;-webkit-box-sizing:border-box !important;-moz-box-sizing:border-box !important;box-sizing:border-box !important;max-width:352px}.btn-block-compact,.sg-btn-block-compact{margin:2px auto}input.btn-block,input.sg-btn-block,button.btn-block,button.sg-btn-block{width:100%;max-width:351px}.btn-block+.btn-block,.sg-btn-block+.sg-btn-block{margin-top:5px}#buttonContainer .btn,#buttonContainer .sg-btn{margin:0}.btn-icon .cssImage{margin-bottom:-40px;position:relative;top:-27px}.btn-pair{-webkit-box-sizing:border-box !important;-moz-box-sizing:border-box !important;box-sizing:border-box !important;width:49%}#refineresultscol{width:184px}#refineresults{width:180px}#branding-td{width:186px;text-align:center}.ltr #searchCount{margin-right:4px;margin-left:4px}.ltr #resultsCol .sorting{padding-right:4px}.ltr #resultsCol .showing{padding-left:4px}.ltr #jobsearch{padding-left:4px}.rtl #searchCount{margin-left:4px;margin-right:4px}.rtl #resultsCol .sorting{padding-left:4px}.rtl #resultsCol .showing{padding-right:4px}.rtl #jobsearch{padding-right:4px}.jaui,.hasu .gasc,.row,.message,.oocs{padding-left:4px;padding-right:4px}#resumePromo{padding-left:4px;padding-right:4px}#primePromo{padding-left:4px;padding-right:4px}@media only screen and (min-width: 1125px){.ltr #refineresults{padding-left:15px}.ltr #branding img{margin-left:16px;margin-right:28px}.rtl #refineresults{padding-right:15px}.rtl #branding img{margin-right:16px;margin-left:28px}.ltr #searchCount{margin-right:12px;margin-left:12px}.ltr #resultsCol .sorting{padding-right:12px}.ltr #resultsCol .showing{padding-left:12px}.ltr #jobsearch{padding-left:12px}.rtl #searchCount{margin-left:12px;margin-right:12px}.rtl #resultsCol .sorting{padding-left:12px}.rtl #resultsCol .showing{padding-right:12px}.rtl #jobsearch{padding-right:12px}.jaui,.hasu .gasc,.row,.message,.oocs{padding-left:12px;padding-right:12px}#resumePromo{padding-left:12px;padding-right:12px}#primePromo{padding-left:12px;padding-right:12px}#resultsBody{width:1125px}#refineresultscol,#branding-td{width:225px}#refineresults{width:210px}}@media only screen and (min-width: 1250px){#resultsBody{width:1250px}#auxCol{width:315px}#refineresultscol,#branding-td{width:275px}#refineresults{width:260px}.ltr #refineresults{padding-left:15px}.ltr #branding img{margin-left:16px;margin-right:28px}.rtl #refineresults{padding-right:15px}.rtl #branding img{margin-right:16px;margin-left:28px}.ltr #searchCount{margin-right:12px;margin-left:12px}.ltr #resultsCol .sorting{padding-right:12px}.ltr #resultsCol .showing{padding-left:12px}.ltr #jobsearch{padding-left:12px}.rtl #searchCount{margin-left:12px;margin-right:12px}.rtl #resultsCol .sorting{padding-left:12px}.rtl #resultsCol .showing{padding-right:12px}.rtl #jobsearch{padding-right:12px}.jaui,.hasu .gasc,.row,.message,.oocs{padding-left:12px;padding-right:12px}#resumePromo{padding-left:12px;padding-right:12px}#primePromo{padding-left:12px;padding-right:12px}}#branding-td{padding:5px 0 5px 0px}.resultsTop{padding-top:9px}
       </style>
      </link>
     </head>
     <body class="ltr jasxcustomfonttst-inactive " data-tn-application="jasx" data-tn-olth="41be357fa1c7dc26c5ee98836f8950b3" data-tn-originlogid="1c6u39dhta24qbca" data-tn-originlogtype="jobsearch">
      <div id="accessibilityBanner">
       <span id="accessibilityText">
        Skip to
        <!-- This is translated before reaching this state -->
        <a class="accessibilityMenu" href="#jobPostingsAnchor" id="skipToJobs">
         Job Postings
        </a>
        ,
        <!-- This is translated before reaching this state -->
        <a class="accessibilityMenu" href="#what" id="skipToSearch">
         Search
        </a>
       </span>
       <a id="accessibilityClose">
        Close
       </a>
      </div>
      <script type="text/javascript">
       createTabBar('1c6u39dhta24qbca');
      </script>
      <script type="text/javascript">
       var vjsStable = false;
      var vjsExp = true;
      var viewJobOnSerp = new indeed.vjs('1c6u39dhta24qbca', vjsExp);
      var vjk = viewJobOnSerp.getVJK();
      var vjFrom = viewJobOnSerp.getFrom();
      var vjTk = viewJobOnSerp.getTk();
      var vjUrl = viewJobOnSerp.vjUrl(vjk, vjFrom, vjTk);
      var showVjOnSerp = vjsStable || vjsExp;
      var zrp = false;
      if ((zrp || !showVjOnSerp || window.innerWidth < 1280) && vjUrl) {
        window.location.replace(vjUrl);
      } else if (showVjOnSerp) {
        var jobKeysWithInfo = {};
        
        jobKeysWithInfo['bbb3d7f08e6073ee'] = true;
        
        jobKeysWithInfo['0d9a0d1520d456d6'] = true;
        
        jobKeysWithInfo['5c07dc6626afe971'] = true;
        
        jobKeysWithInfo['5923aa20581d8796'] = true;
        
        jobKeysWithInfo['4ae29996c0cf2693'] = true;
        
        jobKeysWithInfo['0cd229ee6b711c05'] = true;
        
        jobKeysWithInfo['0829198f649e9c08'] = true;
        
        jobKeysWithInfo['ed6c0f013d00569e'] = true;
        
        jobKeysWithInfo['d3651049ccc706d2'] = true;
        
        jobKeysWithInfo['10b5a33e6be90586'] = true;
        
        jobKeysWithInfo['7cc26370ceeb26a7'] = true;
        
        jobKeysWithInfo['d42da7a52cf52852'] = true;
        
        jobKeysWithInfo['d04f631427a29323'] = true;
        
        if (vjk && !jobKeysWithInfo.hasOwnProperty(vjk)) {
          jobKeysWithInfo[vjk] = true;
        }
        viewJobOnSerp.preloadDescs(jobKeysWithInfo);
      }
      </script>
      <style type="text/css">
       body { margin-top: 0; margin-left: 0; margin-right: 0; padding-top: 0; padding-right: 0; padding-left: 0; }
    
        #g_nav { border-bottom:1px solid #ccc; margin-bottom:9px; }
    
        #g_nav a,
        #g_nav a:visited { color: #00c; }
    
        .navBi { display: -moz-inline-box; display: inline-block; padding: 9px 12px; margin: 0; list-style-type: none; }
      </style>
      <div class="left" data-tn-section="globalNav" id="g_nav" role="navigation">
       <table cellpadding="0" cellspacing="0" width="100%">
        <tr>
         <td nowrap="">
          <style type="text/css">
           #p_nav a.selected { font-weight: bold; text-decoration:none; color: #000 !important; }
          </style>
          <div id="p_nav">
           <span class="navBi">
            <a class="selected" href="/" id="jobsLink" title="Jobs">
             Find Jobs
            </a>
           </span>
           <span class="navBi">
            <a href="/companies" onmousedown="this.href = appendParamsOnce(this.href, '?from=headercmplink&amp;attributionid=jobsearch')">
             Company Reviews
            </a>
           </span>
           <span class="navBi">
            <a href="/salaries" onmousedown="this.href = appendParamsOnce(this.href, '?from=headercmplink&amp;attributionid=jobsearch')">
             Find Salaries
            </a>
           </span>
           <span class="navBi">
            <a href="/resumes?isid=find-resumes&amp;ikw=SERPtop&amp;co=US&amp;hl=en" id="rezLink">
             Find Resumes
            </a>
           </span>
           <span class="navBi">
            <a href="/hire?hl=en&amp;cc=US" id="empLink" onclick="if ( this.href.match('&amp;isid=employerlink-US-control&amp;ikw=SERPtop') == null ) { this.href += '&amp;isid=employerlink-US-control&amp;ikw=SERPtop' };">
             Employers / Post Job
            </a>
           </span>
          </div>
         </td>
         <td align="right" nowrap="">
          <style type="text/css">
           #navpromo a,
        #navpromo a:visited {
            color: #f60;
        }
    
        
        #u_nav .login_unconfirmed,
        #u_nav .login_unconfirmed a,
        #u_nav .login_unconfirmed a:visited {
            color: #c00
        }
    
        #u_nav .resume_pending,
        #u_nav .resume_pending a,
        #u_nav .resume_pending a:visited {
            color: #c00
        }
    
        #userOptionsLabel {
            position: relative;
            z-index: 5;
        }
    
        #userOptionsLabel b {
            cursor: pointer;
            text-decoration: underline;
            position: relative;
            z-index: 5;
        }
    
        #userOptionsLabel:active {
            outline: none;
        }
    
        #userOptionsLabel.active {
            padding: 9px 11px;
            margin-bottom: -1px;
            _margin-bottom: 0px;
            border: 1px solid #ccc;
            border-top: 0;
        }
    
        #userOptionsLabel.active .arrowStub {
            border-width: 0 3px 3px;
            _border-width: 0px 3px 4px;
            border-color: transparent;
            border-bottom-color: #666;
            top: -2px;
            border-style: dashed dashed solid;
        }
    
        #userOptionsLabel.active .halfPxlFix {
            background: #fff;
            bottom: -3px;
            height: 6px;
            left: 0;
            position: absolute;
            right: 0;
            border: 1px solid #fff;
        }
    
        .arrowStub {
            position: relative;
            border-style: solid dashed dashed;
            border-color: transparent;
            border-top-color: #666;
            display: -moz-inline-box;
            display: inline-block;
            font-size: 0;
            height: 0;
            line-height: 0;
            width: 0;
        left: 4px;
            border-top-width: 3px;
            border-bottom-width: 0;
            border-right-width: 3px;
            padding-top: 1px;
            top: -1px;
        }
    
        #userOptions {
            z-index: 2;
            visibility: hidden;
            position: absolute;
        right: 0;
            x_right: -1px;
            top: 100%;
            padding: 9px 15px;
            border: 1px solid #ccc;
            background: #fff;
            min-width: 150px;
            _width: 150px;
            text-align: left;
        }
    
        #userOptions.open {
            visibility: visible;
        }
    
        .userOptionItem {
            margin: 6px 0;
        }
    
        .userOptionItem a {
            white-space: nowrap;
        }
    
        .userOptionGroup {
            border-top: 1px solid #e8e8e8;
            margin-top: 12px;
        }
    
        .userNameRepeat {
            color: #a8a8a8;
            padding-right: 48px;
            font-weight: bold;
        }
    
        .userOptionGroupHeader {
            font-weight: bold;
            margin: 6px 0;
        }
    
        #g_nav {
        position: relative;
    }
    
    #g_nav td {
        height: 58px;
        font-size:13px;
    }
    
    .resumeCTAOrangeOutlineGreyBackgound {
        border: #f60 solid 1px;
        border-radius: 3px;
    }
    
    .resumeCTAOrangeOutlineGreyBackgound:hover {
        background-color: #f7f7f7;
    }
    
    .resumeCTAOrangeOutlineGreyBackgound span {
        margin-left: -13px;
        margin-right: -13px;
    }
    
    .resumeCTAOrangeOutlineGreyBackgound span a {
        text-decoration: none;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        padding: 9px 12px;
    }
    
    .resumeCTAOrangeOutlineGreyBackgound span a:hover {
        text-decoration: none;
    }
    
    .resumeCTAOrangeOutlineOrangeBackgound {
        border: #f60 solid 1px;
        border-radius: 3px;
    }
    
    .resumeCTAOrangeOutlineOrangeBackgound:hover {
        background-color: #fffaf7;
    }
    
    .resumeCTAOrangeOutlineOrangeBackgound span {
        margin-left: -12px;
        margin-right: -12px;
    }
    
    .resumeCTAOrangeOutlineOrangeBackgound span a {
        text-decoration: none;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        padding: 9px 12px;
    }
    
    .resumeCTAOrangeOutlineOrangeBackgound span a:hover {
        text-decoration: none;
    }
          </style>
          <div id="u_nav">
           <script>
            function regExpEscape(s) {
        return String(s).replace(/([-()\[\]{}+?*.$\^|,:#<!\\])/g, '\\$1').
                replace(/\x08/g, '\\x08');
    }
    
    
    function appendParamsOnce(url, params) {
        var useParams = params.replace(/^(\?|\&)/, '');
        if (url.match(new RegExp('[\\?|\\&]' + regExpEscape(useParams))) == null) {
            return url += (url.indexOf('?') > 0 ? '&' : '?' ) + useParams;
        }
        return url;
    }
           </script>
           <div id="user_actions">
            <span class="navBi resumeCTAWhiteOutline resumeCTAOrangeOutlineGreyBackgound">
             <span class="resume-promo" id="navpromo">
              <a href="/promo/resume" onclick="window.location=this.href + '?from=nav&amp;subfrom=rezprmstd&amp;trk.origin=jobsearch&amp;trk.variant=rezprmstd&amp;trk.pos=nav&amp;trk.tk=1c6u39dhta24qbca'; return false;">
               Upload your resume
              </a>
             </span>
            </span>
            <span class="navBi">
             <a href="https://www.indeed.com/account/login?dest=%2Fjobs%3Fq%3Ddata%26l%3DAustin%2C%2BTX%26sort%3Ddate" id="userOptionsLabel" rel="nofollow">
              Sign in
             </a>
            </span>
           </div>
          </div>
         </td>
        </tr>
       </table>
      </div>
      <style type="text/css">
       .indeedLogo {
            margin: 8px 0 0 9px;
            border: 0;
            width: 166px;
            height: 64px;
        }
        
                .indeedLogo {
                    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 155 43'%3E%3Cstyle%3E.st0%7Bfill:%232164F3;%7D%3C/style%3E%3Cpath class='st0' d='M147.2 34.1c-.6 1.2-1.3 2.1-2.2 2.6-.9.6-2 .8-3.1.8s-2.2-.3-3.1-.9c-.9-.6-1.7-1.5-2.2-2.7-.5-1.2-.8-2.7-.8-4.4 0-1.6.3-3.1.8-4.3.5-1.2 1.2-2.2 2.2-2.8.9-.7 2-1 3.2-1h.1c1.1 0 2.1.3 3.1 1 .9.6 1.7 1.6 2.3 2.7.6 1.2.8 2.7.8 4.3-.3 2-.5 3.5-1.1 4.7m6.5-25.7c-.5-.6-1.3-.9-2.3-.9-1 0-1.7.3-2.3 1-.5.7-.8 1.6-.8 2.9v9.4c-1.2-1.3-2.5-2.3-3.7-2.9-.8-.4-1.7-.7-2.8-.8-.6-.1-1.2-.1-1.9-.1-3.2 0-5.8 1.1-7.8 3.3-2 2.2-3 5.3-3 9.2 0 1.9.3 3.6.8 5.2s1.2 3 2.2 4.2 2.1 2.1 3.4 2.7c1.3.6 2.7 1 4.3 1 .7 0 1.4-.1 2-.2l1.2-.3c1-.3 1.9-.8 2.7-1.4.8-.6 1.7-1.4 2.6-2.4v.6c0 1.2.3 2 .8 2.7.6.6 1.3.9 2.2.9s1.6-.3 2.2-.9c.6-.6.8-1.5.8-2.7V11.1c.2-1.2-.1-2.1-.6-2.7m-41.5 14.4c1-1.1 2.3-1.6 3.9-1.6 1.7 0 3 .5 4.1 1.6 1 1.1 1.6 2.6 1.8 4.8h-11.7c.2-2.1.8-3.7 1.9-4.8m15 11.6c-.4-.4-.9-.5-1.6-.5-.6 0-1.1.1-1.4.4-.8.7-1.5 1.3-2 1.8-.5.4-1.1.8-1.7 1.2-.6.4-1.2.7-1.8.8-.7.2-1.3.3-2.1.3h-.5c-1-.1-1.9-.3-2.7-.8-1-.5-1.7-1.4-2.3-2.4-.6-1.1-.8-2.5-.9-3.9h12.6c1.7 0 3-.1 3.9-.6.9-.5 1.4-1.5 1.4-3.1 0-1.7-.5-3.4-1.4-5.1-.9-1.6-2.2-3-4.1-4-1.8-1-3.9-1.6-6.5-1.6h-.2c-1.9 0-3.6.3-5.1.9-1.6.6-2.9 1.5-4 2.6-1.1 1.2-1.9 2.5-2.5 4.1-.6 1.6-.9 3.3-.9 5.2 0 4 1.2 7.1 3.5 9.4 2.2 2.2 5.2 3.3 9 3.5h.7c1.8 0 3.4-.2 4.8-.7 1.4-.5 2.6-1 3.5-1.7.9-.7 1.6-1.5 2.1-2.2.5-.8.7-1.4.7-2 .1-.8-.1-1.3-.5-1.6M86.3 22.8c1-1.1 2.3-1.6 3.9-1.6 1.7 0 3 .5 4.1 1.6 1 1.1 1.6 2.6 1.8 4.8H84.4c.3-2.1.9-3.7 1.9-4.8m13.5 11c-.6 0-1.1.1-1.4.4-.8.7-1.5 1.3-2 1.8-.5.4-1 .8-1.6 1.2-.6.4-1.2.7-1.9.8-.6.2-1.3.3-2.1.3h-.5c-1-.1-1.9-.3-2.7-.8-.9-.5-1.7-1.4-2.3-2.4-.6-1.1-.9-2.5-.9-3.9H97c1.7 0 3-.1 3.9-.6.9-.5 1.4-1.5 1.4-3.1 0-1.7-.4-3.4-1.3-5.1-.9-1.6-2.3-3-4.1-4-1.8-1-4-1.6-6.5-1.6h-.2c-1.9 0-3.5.3-5.1.9-1.6.6-2.9 1.5-4 2.6-1.1 1.2-1.9 2.5-2.5 4.1s-.9 3.3-.9 5.2c0 4 1.2 7.1 3.5 9.4 2.2 2.2 5.2 3.3 9 3.5h.7c1.8 0 3.4-.2 4.8-.7 1.4-.5 2.6-1 3.5-1.7.9-.7 1.6-1.5 2.1-2.2.5-.8.7-1.4.7-2 0-.6-.2-1.2-.6-1.5-.4-.4-1-.6-1.6-.6M14 38.3c0 1.4.3 2.4 1 3.1.7.7 1.5 1.1 2.5 1.1 1.1 0 1.9-.4 2.6-1 .7-.7 1-1.7 1-3.1V21.8c-1.7 1-3.6 1.5-5.7 1.5-.4 0-.9 0-1.3-.1L14 38.3m54.2-4.2c-.6 1.2-1.3 2.1-2.3 2.6-.9.6-2 .8-3.1.8s-2.2-.3-3.1-.9c-1-.6-1.7-1.5-2.2-2.7s-.8-2.7-.8-4.4c0-1.6.3-3.1.8-4.3.5-1.2 1.2-2.2 2.2-2.8.9-.7 2-1 3.1-1h.1c1.1 0 2.1.3 3.1 1 1 .6 1.7 1.6 2.3 2.7.5 1.2.8 2.7.8 4.3 0 2-.3 3.5-.9 4.7m6.5-25.7c-.5-.6-1.3-.9-2.2-.9-1 0-1.7.3-2.3 1-.5.7-.8 1.6-.8 2.9v9.3c-1.2-1.3-2.5-2.3-3.8-2.9-.8-.4-1.7-.7-2.8-.8-.6-.1-1.2-.1-1.9-.1-3.2 0-5.8 1.1-7.8 3.3-2 2.2-3 5.3-3 9.2 0 1.9.3 3.6.7 5.2.5 1.6 1.2 3 2.2 4.2 1 1.2 2.1 2.1 3.4 2.7 1.3.6 2.7 1 4.3 1 .7 0 1.4-.1 2-.2l1.2-.3c1-.3 1.9-.8 2.7-1.4.8-.6 1.7-1.4 2.6-2.4v.6c0 1.2.3 2 .9 2.7.6.6 1.3.9 2.2.9.8 0 1.6-.3 2.2-.9.6-.6.8-1.5.8-2.7V11.1c.2-1.2 0-2.1-.6-2.7M31.3 20.3c0-.7-.1-1.4-.4-1.9-.3-.5-.7-.9-1.1-1.2-.5-.3-1-.4-1.5-.4-.9 0-1.7.3-2.2.9-.5.6-.8 1.5-.8 2.7v18.2c0 1.2.3 2.2.8 2.8.6.7 1.4 1 2.3 1 1 0 1.7-.3 2.3-1 .6-.7.8-1.6.8-2.9v-7.9c0-2.6.2-4.3.5-5.3.4-1.2 1.2-2.1 2.1-2.8.9-.7 2-1 3.1-1 1.8 0 3 .6 3.5 1.6.6 1.1.8 2.7.8 4.8v10.5c0 1.2.4 2.2 1 2.8.6.7 1.4 1 2.3 1 .9 0 1.7-.3 2.3-1 .6-.6.9-1.6.9-2.9V26.9c0-1.4-.1-2.5-.2-3.4-.1-.9-.4-1.7-.8-2.5-.7-1.3-1.6-2.3-3-3.1s-2.9-1.1-4.6-1.1c-1.8 0-3.3.4-4.7 1-1.3.7-2.6 1.8-3.7 3.2M14 2.2c-6.3 2.6-10.7 8.2-12.7 15C1 18.5.7 19.7.5 21c0 0-.1 1.3.1 1 .2-.3.3-.8.4-1.1 1-3.2 2.1-6 3.9-8.8 4.3-6.2 11.2-10.2 18.6-8 1.3.5 2.5 1.2 3.8 1.9.2.2 1.9 1.5 1.6.3-.3-.9-1.1-1.7-1.8-2.4C23.3.8 18.3.7 14 2.2M19.6 19c2.6-1.3 3.7-4.4 2.3-7-1.3-2.6-4.5-3.6-7.1-2.3-2.6 1.3-3.6 4.4-2.3 7 1.3 2.6 4.5 3.6 7.1 2.3'/%3E%3C/svg%3E") no-repeat;
                    background-size: 155px 43px;
                }
            
    #branding img { border: 0; }
    #jobsearch { margin: 0 }
    .inwrap { border-right: 1px solid #e8e8e8;border-bottom: 1px solid #e8e8e8;display:inline-block; }
    .inwrap input { box-sizing: border-box; margin:0; height: 30px; font-family:Arial,sans-serif;border:1px solid #ccc; border-bottom-color:#aaa;border-right-color:#aaa; -webkit-border-radius: 0; -webkit-appearance: none; }
    .inwrap .input_text { font-size:18px;padding:3px 6px;_margin: -1px 0; }
    .inwrap .input_submit {color:#614041;font-size:15px;height:30px;background: #e8e8e8; padding:3px 9px;cursor:pointer;_padding:3px;}
    .inwrap .input_submit:active { background: #ccc; }
    .lnav  {width:100%;line-height:1;;font-size:10pt;}
    .jsf .label {font-size:12px; line-height:1.2;padding-top:0;color:#aaa;font-weight:normal;white-space:nowrap;padding-right:1.5em}
    .jsf .label label {font-weight:normal}
    .jsf .sl { font-size: 11px; color: #77c; white-space: nowrap; }
    .npb { padding-bottom: 0; color: #f60; text-transform: lowercase;font-weight:bold; }
    .npl { padding-left: 0 }
    iframe { display:block; }
    
    .acd { border: 1px solid #333; background: #fff; position:absolute; width:100%; z-index: 1; }
    .aci { font-size: 18px; padding:1px 6px; cursor:pointer; }
    .acis { background:#36c; color:#fff; }
            /* This css contains styles for company links in what autocomplete (DISC-81) */
    
    /* make the whole div with autocomplete suggestion to be a link */
    #what_acdiv .what_ac_link {
        display: inline-block;
        text-decoration: none;
        width: 100%;
        height: 100%;
        color: black;
    }
    
    /* show icon that link will be open in a new window */
    #what_acdiv .what_ac_link:after {
        background: url(/images/icon-open-in-new-tab.png) no-repeat;
        background-position-x: right;
        background-position-y: center;
        background-origin: border-box;
        height: 24px;
        display: inline-block;
        position: absolute;
        padding-left: 5px;
        padding-right: 20px;
        color: #666666;
    }
    
    #what_acdiv .what_ac_reviews_link:after {
        content: " reviews";
    }
    #what_acdiv .what_ac_salaries_link:after {
        content: " salaries";
    }
    
    /* if item is chosen then text should be white */
    #what_acdiv .acis .what_ac_link,
    #what_acdiv .acis .what_ac_link:after {
        color: #fff;
    }
    
    
    
    #jobalerts .ws_label,
    #jobalerts .member { z-index: 1; }
    #acr td { padding-top:0; padding-bottom:0; }
    #acr td .h { display:none; }
    
    #what { width: 280px; }
    #where { width: 260px; }
    .inwrapBorder{border:1px solid #1C4ED9;border-top-color:#2F62F1;border-bottom-color:#133FBB;display:inline-block;width:auto;}
    .inwrapBorderTop{border-top:1px solid #69F;display:inline-block;background-color:#3163F2;filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#3163F2',endColorstr='#2B57D5');background:-webkit-gradient(linear,left top,left bottom,from(#3163F2),to(#2B57D5));background:-moz-linear-gradient(top,#3163F2,#2B57D5);background:linear-gradient(top,#3163F2,#2B57D5);}
    .inwrapBorder .input_submit{background: transparent;border:0;color:#fff;font-family:Arial;font-size:15px;margin:0;padding:4px 9px;cursor:pointer;_padding:3px;}
    .inwrapBorder a.input_submit{text-decoration: none; display: block;}
    
    .inwrapBorder:hover{border-color:#235af6;border-top-color:#4072ff;border-bottom-color:#1e4fd9;}
    .inwrapBorderTop:hover{border-top-color:#7ba7ff;background-color:#4273ff;filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#4273ff',endColorstr='#3364f1');background:-webkit-gradient(linear,left top,left bottom,from(#4273ff),to(#3364f1));background:-moz-linear-gradient(top,#4273ff,#3364f1);background:linear-gradient(top,#4273ff,#3364f1);}
    
    .inwrapBorder:active{border-color:#536db7;border-top-color:#4b69c1;border-bottom-color:#3753a6;}
    .inwrapBorder:active .inwrapBorderTop{border-top-color:#6c82c1;background-color:#4b69c1;filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#4b69c1',endColorstr='#3753a6');background:-webkit-gradient(linear,left top,left bottom,from(#4b69c1),to(#3753a6));background:-moz-linear-gradient(top,#4b69c1,#3753a6);background:linear-gradient(top,#4b69c1,#3753a6);}
    
    .roundedCorner {
        display: inline-block;
        zoom: 1; /* zoom and *display = ie7 hack for display:inline-block */
        *display: inline;
        vertical-align: baseline;
        margin: 0 2px;
        outline: none;
        cursor: pointer;
        text-align: center;
        text-decoration: none;
        font: 15px/100% Arial, Helvetica, sans-serif;
        padding: .5em 2em .55em;
        text-shadow: 0 1px 1px rgba(0,0,0,.3);
        -webkit-border-radius: .5em;
        -moz-border-radius: .5em;
        border-radius: .5em;
        -webkit-box-shadow: 0 1px 2px rgba(0,0,0,.2);
        -moz-box-shadow: 0 1px 2px rgba(0,0,0,.2);
        box-shadow: 0 1px 2px rgba(0,0,0,.2);
    }
    .roundedCorner:hover {
        text-decoration: none;
    }
    .roundedCorner:active {
        position: relative;
        top: 1px;
    }
    
    .bigrounded {
        -webkit-border-radius: 2em;
        -moz-border-radius: 2em;
        border-radius: 2em;
    }
    .medium {
        font-size: 12px;
        padding: .4em 1.5em .42em;
    }
    .small {
        font-size: 11px;
        padding: .2em 1em .275em;
    }
    
    .indeedblue {
        color: #d9eef7;
        border: solid 1px #1C4ED9;
        background: #3163F2;
        background: -webkit-gradient(linear, left top, left bottom, from(#2F62F1), to(#133FBB));
        background: -moz-linear-gradient(top,  #2F62F1,  #133FBB);
        filter:  progid:DXImageTransform.Microsoft.gradient(startColorstr='#2F62F1', endColorstr='#133FBB');
    }
    .indeedblue:hover, .indeedblue:focus {
        background: #235af6;
        background: -webkit-gradient(linear, left top, left bottom, from(#4072ff), to(#1e4fd9));
        background: -moz-linear-gradient(top,  #4072ff,  #1e4fd9);
        filter:  progid:DXImageTransform.Microsoft.gradient(startColorstr='#4072ff', endColorstr='#1e4fd9');
    }
    .indeedblue:active {
        color: #d9eef7;
        background: -webkit-gradient(linear, left top, left bottom, from(#4b69c1), to(#3753a6));
        background: -moz-linear-gradient(top,  #4b69c1,  #3753a6);
        filter:  progid:DXImageTransform.Microsoft.gradient(startColorstr='#4b69c1', endColorstr='#3753a6');
    }
      </style>
      <span id="hidden_colon" style="display:none">
       :
      </span>
      <table border="0" cellpadding="0" cellspacing="0" role="banner">
       <tr>
        <td width="1125">
         <table cellpadding="0" cellspacing="0" class="lnav">
          <tr>
           <td id="branding-td" style="vertical-align:top;">
            <a href="/" id="branding" onmousedown="ptk('logo');">
             <img alt="one search. all jobs. Indeed" class="indeedLogo" src="data:image/gif;base64,R0lGODlhAQABAJEAAAAAAP///////wAAACH5BAEHAAIALAAAAAABAAEAAAICVAEAOw==" style="margin-bottom:6px;display:block;" title="one search. all jobs. Indeed"/>
            </a>
           </td>
           <td style="padding-top:3px;" valign="top">
            <form action="/jobs" class="jsf" id="jobsearch" method="get" name="js" onsubmit="formptk('topsearch','where_ac','',['where_ac','what_ac'], ptk);formptk('topsearch','what_ac','w',['where_ac','what_ac'], ptk);">
             <table align="left" cellpadding="3" cellspacing="0">
              <tr>
               <td class="npb">
                <label for="what" id="what_label_top">
                 What
                </label>
               </td>
               <td class="npb" colspan="3">
                <label for="where" id="where_label_top">
                 Where
                </label>
               </td>
              </tr>
              <tr role="search">
               <td class="npl epr">
                <span class="inwrap">
                 <input aria-labelledby="what_label_top hidden_colon what_label" class="input_text" id="what" maxlength="512" name="q" size="31" value="data"/>
                </span>
                <div style="width:250px">
                 <!-- -->
                </div>
               </td>
               <td class="npl epr">
                <span class="inwrap">
                 <input aria-labelledby="where_label_top hidden_colon where_label" class="input_text" id="where" maxlength="45" name="l" size="27" value="Austin, TX"/>
                </span>
                <div style="width:200px">
                 <!-- -->
                </div>
               </td>
               <td class="npl" style="width:1px">
                <span class="inwrapBorder" style="width:auto;padding-right:0;">
                 <span class="inwrapBorderTop">
                  <input class="input_submit" id="fj" type="submit" value="Find Jobs"/>
                 </span>
                </span>
               </td>
               <td class="npl advanced-search" style="width:240px;">
                <div style="margin-left:12px; display:flex;">
                 <a class="sl" href="/advanced_search?q=data&amp;l=Austin%2C+TX&amp;sort=date">
                  Advanced Job Search
                 </a>
                </div>
               </td>
              </tr>
              <tr id="acr">
               <td>
                <span class="h">
                </span>
               </td>
               <td class="npl" colspan="2">
                <div style="position:relative;z-index:2">
                 <div class="acd" id="acdiv" style="display:none;">
                 </div>
                </div>
               </td>
               <td>
                <span class="h">
                </span>
               </td>
              </tr>
              <tr id="acr">
               <td class="npl" colspan="3">
                <div style="position:relative;z-index:2">
                 <div class="acd" id="what_acdiv" style="display:none;">
                 </div>
                </div>
               </td>
               <td>
                <span class="h">
                </span>
               </td>
              </tr>
              <tr valign="baseline">
               <td class="label" id="what_label_cell">
                <label aria-hidden="true" for="what" id="what_label">
                 job title, keywords or company
                </label>
               </td>
               <td class="label" colspan="3" id="where_label_cell">
                <label aria-hidden="true" for="where" id="where_label">
                 city, state, or zip
                </label>
               </td>
              </tr>
             </table>
             <input name="sort" type="hidden" value="date"/>
            </form>
           </td>
          </tr>
         </table>
        </td>
       </tr>
      </table>
      <script type="text/javascript">
       initAutocomplete('where_ac', gbid('where'), gbid('acdiv'), '/rpc/suggest?from=serp&tk=1c6u39dhta24qbca', function() { formptk('topsearch','where_ac', '', ['where_ac','what_ac'], ptk); }, gbid('where'));
            
            initAutocomplete('what_ac', gbid('what'), gbid('what_acdiv'), '/rpc/suggest?what=true&tk=1c6u39dhta24qbca', function () {formptk('topsearch', 'what_ac', 'w', ['where_ac','what_ac'], ptk);}, gbid('what'));
      </script>
      <script type="text/javascript">
       function rclk(el,jobdata,oc,sal) { var ocstr = oc ? '&onclick=1' : ''; document.cookie='RCLK="jk='+jobdata.jk+'&tk=1c6u39dhta24qbca&from=web&rd='+jobdata.rd+'&qd=7tdTJLF8oc4dPpT7T_zGvOT24PAHysyotbVU1nJkvToW-5fGtbavxSQ3uaIYTsR3Kr6JUjN7qTt0azi8Svj-fwT57RfWA3K-GG7XV8VBSdSWTS_-MFJWMIamgd2yMiuEax6Fy1aha7m-zaHuiQJrVA&ts=1519281026621&sal='+sal+ocstr+'"; path=/'; return true;}
    
    function vjrclk(jk, qd, rd, oc, vjk, vjtk) {
      var ocstr = oc ? '&onclick=1' : '';
      document.cookie = 'RCLK="jk=' + jk + '&vjtk=' + vjtk + '&rd=' + rd + '&qd=' + qd + '&ts=' + new Date().getTime() + ocstr + '"; path=/';
      return true;
    }
    function zrprclk(el,jobdata,oc) { var ocstr = oc ? '&onclick=1' : ''; document.cookie='RCLK="jk='+jobdata.jk+'&tk=1c6u39dhta24qbca&from=reconzrp&rd='+jobdata.rd+'&qd=7tdTJLF8oc4dPpT7T_zGvOT24PAHysyotbVU1nJkvToW-5fGtbavxSQ3uaIYTsR3Kr6JUjN7qTt0azi8Svj-fwT57RfWA3K-GG7XV8VBSdSWTS_-MFJWMIamgd2yMiuEax6Fy1aha7m-zaHuiQJrVA&ts=1519281026621'+ocstr+'"; path=/'; return true;}
    function prjbottomclk(el,jobdata,oc) { var ocstr = oc ? '&onclick=1' : ''; document.cookie='RCLK="jk='+jobdata.jk+'&tk=1c6u39dhta24qbca&from=reconserp&rd='+jobdata.rd+'&qd=7tdTJLF8oc4dPpT7T_zGvOT24PAHysyotbVU1nJkvToW-5fGtbavxSQ3uaIYTsR3Kr6JUjN7qTt0azi8Svj-fwT57RfWA3K-GG7XV8VBSdSWTS_-MFJWMIamgd2yMiuEax6Fy1aha7m-zaHuiQJrVA&ts=1519281026621'+ocstr+'"; path=/'; return true;}
    
    
    var jobmap = {};
    
    jobmap[0]= {jk:'0829198f649e9c08',efccid: 'e5d38cda544df340',srcid:'f3df9bbac1540a21',cmpid:'7c30762e902763ee',num:'0',srcname:'Absolute Software',cmp:'Absolute Software',cmpesc:'Absolute Software',cmplnk:'/q-Absolute-Software-l-Austin,-TX-jobs.html',loc:'Austin, TX 78758',country:'US',zip:'',city:'Austin',title:'Data Entry Associate',locid:'d2a39b6d57d82344',rd:'ifLLha17ynEBlt7hho3JcfEjTYFBCydOHIJo0jSLt2vjnlmvqxomkjTMlrwPbnNo'};
    
    jobmap[1]= {jk:'ed6c0f013d00569e',efccid: '5538e76313d07ba1',srcid:'1e64c0200d18f5ca',cmpid:'920fd18472f1d3fe',num:'1',srcname:'Keller Williams',cmp:'Keller Williams',cmpesc:'Keller Williams',cmplnk:'/q-Keller-Williams-l-Austin,-TX-jobs.html',loc:'Austin, TX 78746',country:'US',zip:'78746',city:'Austin',title:'Business Intelligence Developer\/ Analyst',locid:'d2a39b6d57d82344',rd:'KMmgpPP_N8Ma8_Z5coeA6vEjTYFBCydOHIJo0jSLt2sWsH2Q85l9-2BS3E1N9GiI'};
    
    jobmap[2]= {jk:'5c07dc6626afe971',efccid: 'ba979d6fb0c7a936',srcid:'1c248d1533c507ca',cmpid:'a3f737e511d9fc8c',num:'2',srcname:'Visa',cmp:'Visa',cmpesc:'Visa',cmplnk:'/q-Visa-l-Austin,-TX-jobs.html',loc:'Austin, TX',country:'US',zip:'',city:'Austin',title:'HR Reporting &amp; Analytics Manager',locid:'d2a39b6d57d82344',rd:'0nbXxlm8yujnNeH6qUX3_PEjTYFBCydOHIJo0jSLt2v45dE2Xc7f0LxoyV9j6I_6'};
    
    jobmap[3]= {jk:'d42da7a52cf52852',efccid: 'f2a447244b103d3c',srcid:'8f3390d328804748',cmpid:'e62aa8d5bb020acb',num:'3',srcname:'RateGenius, Inc.',cmp:'RateGenius, Inc.',cmpesc:'RateGenius, Inc.',cmplnk:'/q-RateGenius-l-Austin,-TX-jobs.html',loc:'Austin, TX 78758',country:'US',zip:'78758',city:'Austin',title:'Marketing Data Analyst',locid:'d2a39b6d57d82344',rd:'yiFpXy0V5VnR7pRG_LDt7_EjTYFBCydOHIJo0jSLt2s6Y-yqBPyGuGJJ2wHIo2IB'};
    
    jobmap[4]= {jk:'0cd229ee6b711c05',efccid: '4989b90f9dfc5cf3',srcid:'7e51802b6d314dcb',cmpid:'0ab4fcb6780213bd',num:'4',srcname:'Texas Health and Human Services Commission',cmp:'Dept of State Health Services',cmpesc:'Dept of State Health Services',cmplnk:'/q-Dept-of-State-Health-Services-l-Austin,-TX-jobs.html',loc:'Austin, TX',country:'US',zip:'',city:'Austin',title:'Management Analyst III',locid:'d2a39b6d57d82344',rd:'UoSLGNYeoaCEbrc4zUQx7_EjTYFBCydOHIJo0jSLt2v643S2KwFAze-YU5kL_RQ0'};
    
    jobmap[5]= {jk:'bbb3d7f08e6073ee',efccid: 'd38502ddb1d78a1f',srcid:'3209cb17a2b4f98a',cmpid:'12042485d1cec3f9',num:'5',srcname:'IGT',cmp:'IGT',cmpesc:'IGT',cmplnk:'/q-IGT-l-Austin,-TX-jobs.html',loc:'Austin, TX 78727',country:'US',zip:'78727',city:'Austin',title:'Market Research Analyst I',locid:'d2a39b6d57d82344',rd:'ypQSWt1Kpk7SEQoWpXENyvEjTYFBCydOHIJo0jSLt2v643S2KwFAze-YU5kL_RQ0'};
    
    jobmap[6]= {jk:'5923aa20581d8796',efccid: 'a0cb924c65b1ea60',srcid:'1eb1e4492716ffd9',cmpid:'075b49fe6eeb6013',num:'6',srcname:'Texas Department of Public Safety',cmp:'DEPARTMENT OF INFORMATION RESOURCES',cmpesc:'DEPARTMENT OF INFORMATION RESOURCES',cmplnk:'/q-DEPARTMENT-OF-INFORMATION-RESOURCES-l-Austin,-TX-jobs.html',loc:'Austin, TX',country:'US',zip:'',city:'Austin',title:'Systems Analyst VI',locid:'d2a39b6d57d82344',rd:'A1kmjcdO9UbIpSbViw6K6vEjTYFBCydOHIJo0jSLt2vhZQCZZPRCz0hxOGFCekI-'};
    
    jobmap[7]= {jk:'7cc26370ceeb26a7',efccid: '07f9fe570766a4c0',srcid:'7e51802b6d314dcb',cmpid:'ec038939be4318c8',num:'7',srcname:'Texas Health and Human Services Commission',cmp:'Health & Human Services Comm',cmpesc:'Health &amp; Human Services Comm',cmplnk:'/q-Health-&-Human-Services-Comm-l-Austin,-TX-jobs.html',loc:'Austin, TX',country:'US',zip:'',city:'Austin',title:'Systems Analyst IV',locid:'d2a39b6d57d82344',rd:'gIBhyxjX88CBheNzlvzgWPEjTYFBCydOHIJo0jSLt2vznZRC5vNCJgR7tGAMSCH-'};
    
    jobmap[8]= {jk:'0d9a0d1520d456d6',efccid: '858bbe77a1be41df',srcid:'db99e6ff1d35a8f9',cmpid:'00347ecfd5bc04ed',num:'8',srcname:'Excell',cmp:'eXcell',cmpesc:'eXcell',cmplnk:'/q-eXcell-l-Austin,-TX-jobs.html',loc:'Austin, TX 78746',country:'US',zip:'78746',city:'Austin',title:'Level 4 Server \/ Data Center Support Windows \/ Linux',locid:'d2a39b6d57d82344',rd:'4END045ptKKmHngt72SuiPEjTYFBCydOHIJo0jSLt2t_C-e3hXJWU7IbVU9qfeAS'};
    
    jobmap[9]= {jk:'10b5a33e6be90586',efccid: '69cbbbd8cf4de3ed',srcid:'618f0d66967a54d9',cmpid:'651f65cb69970e95',num:'9',srcname:'City of Austin',cmp:'City of Austin',cmpesc:'City of Austin',cmplnk:'/q-City-of-Austin-l-Austin,-TX-jobs.html',loc:'Austin, TX 78702',country:'US',zip:'',city:'Austin',title:'Court Clerk Assistant',locid:'d2a39b6d57d82344',rd:'HgqJEbKk5cQW1h3qe3-fi_EjTYFBCydOHIJo0jSLt2uuIEyVEsFa1GnfHxCZMExg'};
      </script>
      <style type="text/css">
       .jobtitle {
                    font-weight: bold;
                }
                td.snip b, span.company b, #femp_list .jobtitle, #cmpinfo_list .jobtitle, .jobtitle .new {
                    font-weight: normal;
                }
                div.result-link-bar b {
                    font-weight: bold;
                }
      </style>
      <style type="text/css">
       div.row table tr td.snip { line-height: 1.4; }
      </style>
      <table border="0" cellpadding="0" cellspacing="0" id="resultsBody" role="main">
       <tr>
        <td>
         <script type="text/javascript">
          window['ree'] = "pdsssps";
        window['jas'] = "xpWMf6wUe";
         </script>
         <style type="text/css">
          /* Promos Generic Styling */
    
    /* Single Link with text */
    .basePromo {
        margin-top: 8px; margin-bottom: 13px; padding-left: 12px; padding-right: 12px;
    }
    
    .redText {
        color: #FF0000;
    }
    
    .bold {
        font-weight: bold;
    }
    
    .basePromo.resume {
        font-size: 14px;
        margin-top: 5px;
    }
    
    .basePromo.resume > img {
        height: 20px;
        margin-right: 5px;
        margin-bottom: 3px;
        width: 16px;
    }
         </style>
         <style type="text/css">
          #jobalertswrapper,
    #picard-profile-completeness-widget,
    #tjobalertswrapper,
    #femp_list,
    #univsrch-salary-v3,
    .rightRailAd {
        margin-left: inherit;
        width: inherit;
    }
    
    .resultsTop::after {
        content: '';
        clear: both;
        display: block;
    }
    
    #vjs-container {
        background-color: #fff;
        border: 1px solid #CCC;
        position: fixed;
        transition: opacity 150ms ease-in-out;
        width: 440px;
        top: 0;
        overflow: hidden;
        z-index: 1;
    }
    
    @media only screen and (min-width: 1280px) {
        #vjs-container {
            width: 504px;
        }
    
        .row .jobtitle {
            white-space: normal !important;
        }
    }
    
    @media only screen and (min-width: 1360px) {
        #vjs-container {
            width: 530px;
        }
    }
    
    @media only screen and (min-width: 1440px) {
        #vjs-container {
            width: 600px;
        }
    }
    
    @media only screen and (min-width: 1740px) {
        #vjs-container {
            width: 790px;
        }
    }
    
    #vjs-header {
        background: #FFFFFF;
        height: auto;
        border-bottom: 1px solid #CCC;
        box-shadow: 0 1px 2px rgba(0, 0, 0, .05);
        position: relative;
        padding: 16px 24px;
    }
    
    #vjs-content {
        overflow-x: auto;
        height: calc(100% - 86px);
        padding: 0 24px 10px 24px;
    }
    
    #vjs-footer #apply-button-container {
        margin-bottom: 10px;
    }
    
    #vjs-desc {
        padding: 16px 0 24px 0;
        line-height: 1.5;
    }
    
    #vjs-jobtitle {
        color: #000000;
        line-height: 20px;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 3px;
    }
    
    #vjs-jobinfo {
        font-size: 11pt;
        line-height: 20px;
    }
    
    #vjs-x-button {
        border: 0;
        background: 0;
        cursor: pointer;
        margin: 0;
        padding: 0;
    }
    
    #vjs-x-span {
        height: 24px;
        width: 24px;
        display: inline-block;
        background-size: 24px;
        color: #666;
        font-weight: bold;
        font-size: 18px;
        cursor: pointer;
    }
    
    #vjs-x {
        position: absolute;
        top: 10px;
    }
    
    .ltr #vjs-x {
        right: 10px;
    }
    
    .rtl #vjs-x {
        left: 10px;
    }
    
    #vjs-footer {
        margin-bottom: 24px;
    }
    
    #vjs-footer #apply-state-picker-container {
        margin-bottom: 0;
    }
    
    .row.vjs-highlight {
        background: #F5F5F5;
        border: 1px solid #CCC !important;
        position: relative;
    }
    
    .vjs-highlight:before,
    .vjs-highlight:after {
        content: " ";
        border-style: solid;
        position: absolute;
        z-index: 101;
    }
    
    .vjs-highlight:before {
        border-color: transparent transparent transparent #ccc;
        border-width: 13px 0 13px 13px;
        right: -13px;
        top: calc(50% - 13px);
    }
    
    .vjs-highlight:after {
        border-color: transparent transparent transparent #f5f5f5;
        border-width: 12px 0 12px 12px;
        right: -12px;
        top: calc(50% - 12px);
    }
    
    .rtl .vjs-highlight:before {
        border-color: transparent #ccc transparent transparent !important;
        border-width: 13px 13px 13px 0 !important;
        left: -13px;
        right: auto;
    }
    
    .rtl .vjs-highlight:after {
        border-color: transparent #f5f5f5 transparent transparent;
        border-width: 12px 12px 12px 0;
        left: -12px;
        right: auto;
    }
    
    .vjs-expired {
        color: #900;
        font-size: 13.5pt;
    }
    
    #footerWrapper {
        position: absolute;
        left: 0;
        right: 0;
        width: 100%;
        border-top: 1px solid #CCC;
        padding-top: 10px;
    }
    
    .no-scroll {
        overflow-x: auto;
        overflow-y: hidden;
    }
    
    #PlaceholderContainer {
        display: block;
        margin: 8px;
    }
    
    .PlaceholderCopy-line,
    .PlaceholderHeader-jobTitle {
        display: block;
        background-color: #e8e8e8;
    }
    
    .PlaceholderHeader-company,
    .PlaceholderHeader-location {
        display: inline-block;
        height: 12px;
        background-color: #e8e8e8;
    }
    
    .PlaceholderHeader-company {
        width: 18%;
    }
    
    .PlaceholderHeader-location {
        width: 40%;
    }
    
    .PlaceholderHeader-stars {
        display: inline-block;
        height: 12px;
        width: 65px;
        background-image: url('/images/stars.png');
        background-size: contain;
    }
    
    .PlaceholderHeader {
        padding: 20px 0;
        margin-bottom: 20px;
        border-bottom: 1px solid #ccc;
        min-width: 200px;
    }
    
    .PlaceholderHeader-jobTitle {
        height: 16px;
        margin-bottom: 12px;
        width: 55%;
    }
    
    .PlaceholderHeader-companyLocation {
        display: block;
    }
    
    .PlaceHolderCopy {
        display: block;
        margin-bottom: 28px;
        max-width: 400px;
    }
    
    .PlaceholderCopy-line {
        height: 12px;
        margin-bottom: 8px;
    }
    
    .PlaceholderCopy-line--1 {
        width: 24%;
    }
    
    .PlaceholderCopy-line--2 {
        width: 100%;
    }
    
    .PlaceholderCopy-line--3 {
        width: 90%;
    }
    
    .PlaceholderCopy-line--4 {
        width: 95%;
    }
    
    .PlaceholderCopy-line--5 {
        width: 80%;
    }
    
    .indeedLogo {
        width: 205px !important;
    }
    
    #refineresults {
        box-sizing: border-box;
        width: 260px;
        margin-right: 20px;
        margin-left: 5px;
    }
    
    #resultsCol {
        width: 410px;
        min-width: 410px;
        padding: 6px 28px 9px 0 !important;
    }
    
    #branding-td {
        min-width: 290px;
    }
    
    @media only screen and (min-width: 1125px) {
        #branding-td {
            min-width: 282px;
        }
    }
    
    @media screen and (min-width: 1366px) {
        #resultsCol {
            min-width: 470px;
        }
    }
    
    @media only screen and (min-width: 1740px) {
        #resultsCol {
            min-width: 586px;
        }
    }
    
    table#pageContent {
        max-width: 1024px;
        min-height: 432px;
    }
    
    #auxCol {
        position: relative !important;
        width: 320px;
        padding-left: 0;
    }
    
    #serpRecommendations .row,
    .row.result {
        padding: 10px;
        border: 1px solid transparent;
        position: relative;
    }
    
    .ltr #serpRecommendations .row,
    .ltr .row.result {
        padding-right: 28px;
    }
    
    .rtl #serpRecommendations .row,
    .rtl .row.result {
        padding-left: 28px;
    }
    
    .ltr #resultsCol .showing {
        padding-left: 10px;
    }
    
    @media only screen and (min-width: 1125px) {
        .ltr #branding img {
            margin-left: 16px;
            margin-right: 0;
        }
    }
    
    @media only screen and (min-width: 1250px) {
        .ltr #branding img {
            margin-left: 16px;
            margin-right: 0;
        }
    }
    
    #resumePromo {
        padding-left: 12px;
    }
    
    .ita-base-container {
        min-height: 600px;
        width: 750px;
        text-align: center;
        z-index: 1000;
        font-size: 40px;
        position: absolute;
        top: 0;
        box-sizing: border-box;
    }
    
    .popover {
        position: fixed;
        display: none;
        -webkit-box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.28);
        -moz-box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.28);
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.28);
    }
    
    #popover-background {
        position: fixed;
        z-index: 999;
        background: #000000;
        width: 100%;
        height: 100%;
        opacity: .30;
        filter: alpha(opacity=30);
        top: 0;
        left: 0;
    }
    
    .ita-base-container > #popover-foreground {
        max-height: 700px;
        width: 750px;
        text-align: left;
        z-index: 1000;
        font-size: 40px;
        position: fixed;
        box-sizing: border-box;
        padding: 25px;
        background: white;
        margin: 0 auto;
        overflow-y: auto;
    }
    
    #popover-x {
        float: right;
        margin: -12px -12px 0 0;
    }
    
    .popover-x-span {
        font-weight: bold;
        font-size: 18px;
        cursor: pointer;
    }
    
    .popover-heading-div {
        text-align: center;
        font-size: 14px;
        padding-bottom: 15px;
    }
    
    #popover-ita-container .btn-primary {
        color: #F8F8F9 !important;
    }
    
    .ita-base-container > #fullpage-div {
        margin: 0 !important;
    }
    
    .itaHeader { width: 100%;  background: white; }
    .itaLogo {display: block; margin-top: 9px; margin-left: auto; margin-right: auto; max-height: 40px; max-width: 159px;}
    .itaClose {text-align:center;font-weight:bold;}
    .itaText {text-align:left; color:#6f6f6f; font-size:90%;}
    .itaLocation {text-align:left; color:#6f6f6f;}
    .itaBold {font-weight:bold;}
    .itaNoFormatting {text-decoration: none}
    .ita-ctas {text-align: center;}.icl-Radio,.icl-Radio-errorText,.icl-Radio-helpText,.icl-Radio-label,.icl-Radio-legend{box-sizing:border-box}.icl-Radio{font-family:Avenir Next,Arial,Roboto,Noto,sans-serif}[dir] .icl-Radio{margin-bottom:12px;border:none}.icl-Radio-errorText{font-size:.8125rem;font-weight:700;color:#ba363f}[dir] .icl-Radio-errorText{margin-top:2px}.icl-Radio-helpText{font-size:.8125rem;display:block;color:#666}[dir] .icl-Radio-helpText{margin-top:2px;margin-bottom:.1875rem}.icl-Radio-label{display:inline-block;font-size:.875rem}[dir] .icl-Radio-label{margin-bottom:4px;padding:0}.icl-Radio-legend{display:inline-block;font-size:.875rem}[dir] .icl-Radio-legend{margin-bottom:4px;padding:0}
    /*# sourceMappingURL=Radio.css.map*//*Need to update this file when there is a update on frontend-icl library, details see ITAD-358*/
    .icl-Alert,.icl-Alert-body,.icl-Alert-close,.icl-Alert-headline,.icl-Alert-icon,.icl-Alert-iconContainer,.icl-Alert-text,.icl-Alert-textLink{box-sizing:border-box}.icl-Alert{display:table;position:relative;width:100%}[dir] .icl-Alert{margin-bottom:12px;padding:.875rem;text-align:center}[dir] .icl-Alert-body{margin-top:.28125rem}.icl-Alert-headline{display:block;font-size:.9375rem;font-weight:700;line-height:1.2}.icl-Alert-icon{vertical-align:middle}.icl-Alert-icon--success{height:1.375rem;width:1.375rem;fill:#008040}.icl-Alert-icon--info{height:1.375rem;width:1.375rem;fill:#2164f3}.icl-Alert-icon--warning{height:1.4375rem;width:1.625rem;fill:#ffb100}.icl-Alert-icon--danger{height:1.375rem;width:1.375rem;fill:#ba363f}.icl-Alert-iconContainer{display:inline-block;width:2.25rem;vertical-align:middle}[dir=ltr] .icl-Alert-iconContainer{text-align:left}[dir=rtl] .icl-Alert-iconContainer{text-align:right}.icl-Alert-text{display:inline-block;max-width:80%;font-family:Avenir Next,Arial,Roboto,Noto,sans-serif;font-size:.8125rem;line-height:1.5;vertical-align:middle}[dir=ltr] .icl-Alert-text{text-align:left}[dir=rtl] .icl-Alert-text{text-align:right}.icl-Alert-textLink{color:#00c;text-decoration:none}.icl-Alert-textLink:hover{text-decoration:underline}.icl-Alert-close{position:absolute;top:0}[dir] .icl-Alert-close{padding:.875rem}[dir=ltr] .icl-Alert-close{right:0}[dir=rtl] .icl-Alert-close{left:0}[dir] .icl-Alert--success{background-color:#e8ffe8;border:1px solid #d5dcd5}.icl-Alert--success .icl-Alert-headline{color:#008040}[dir] .icl-Alert--info{background-color:#e8f4ff;border:1px solid #d5d9dc}[dir] .icl-Alert--warning{background-color:#fff8e7;border:1px solid #dcdad5}[dir] .icl-Alert--danger{background-color:#ffe7e8;border:1px solid #dcd5d6}.icl-Alert--danger .icl-Alert-headline{color:#ba363f}
    /*# sourceMappingURL=Alert.css.map*/
    
    /*used by vjshowrezmatchtexttst*/
    .icl-u-xs-inlineBlock{display:inline-block!important}
    .icl-u-xs-my--md{margin-top:.9375rem!important;margin-bottom:.9375rem!important}
    .icl-u-xs-p--sm{padding:.875rem!important}
    #vjs-header #apply-button-container {
        margin-top: 0;
        margin-bottom: 0;
    }
    
    #vjs-header #apply-button-container .job-footer-button-row {
        margin-top: 10px;
    }
    
    #vjs-content {
        height: calc(100% - 140px);
    }
    
    #vjs-footer {
        height: 90px;
    }.job-footer-button-row {
        margin-top: 30px;
    }
    
    .state-picker-info-dismiss, .state-picker-info-undo {
        float: right;
        margin: 0 10px;
    }
    
    .job-footer-button-row .indeed-apply-button {
        margin-top: 10px;
        margin-right: 15px;
    }
    
    .view-apply-button {
        margin-right: 15px;
    }
    
    #apply-state-picker-container {
        margin-top: 10px;
        margin-bottom: 40px;
    }
    
    #apply-button-container {
        margin-top: 10px;
        margin-bottom: 15px;
    }
    
    #apply-button-container .job-footer-button-row{
        margin-top: 40px;
    }
    .dd-target {
        padding: 0 15px;
        background: #f6f6f6; /* Old browsers */
        background: -moz-linear-gradient(top,  #f6f6f6 0%, #e0e0e0 100%); /* FF3.6+ */
        background: -webkit-gradient(linear, left top, left bottom, color-stop(0%,#f6f6f6), color-stop(100%,#e0e0e0)); /* Chrome,Safari4+ */
        background: -webkit-linear-gradient(top,  #f6f6f6 0%,#e0e0e0 100%); /* Chrome10+,Safari5.1+ */
        background: -o-linear-gradient(top,  #f6f6f6 0%,#e0e0e0 100%); /* Opera 11.10+ */
        background: -ms-linear-gradient(top,  #f6f6f6 0%,#e0e0e0 100%); /* IE10+ */
        background: linear-gradient(to bottom,  #f6f6f6 0%,#e0e0e0 100%); /* W3C */
        filter: progid:DXImageTransform.Microsoft.gradient( startColorstr='#f6f6f6', endColorstr='#e0e0e0',GradientType=0 ); /* IE6-9 */
        color: black;
        -moz-border-radius: 6px;
        -webkit-border-radius: 6px;
        border-radius: 6px;
        border: 1px solid #999;
        font-size: 18px;
        line-height: 31px;
        text-align: center;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        cursor: pointer;
        font-weight: 200;
        text-decoration: none;
        text-shadow: rgb(255, 255, 255) 0 1px 0;
        box-shadow: rgba(0, 0, 0, 0.2) 0 1px 2px 0;
        min-width: 100px;
        outline: none;
    }
    
    .dd-target::-moz-focus-inner, .state-picker-unsaved::-moz-focus-inner {
        border: 0; /* disable dashed selection box for firefox */
    }
    
    .dd-target.dd-active {
        background: #e0e0e0; /* Old browsers */
        background: -moz-linear-gradient(top,  #e0e0e0 0%, #f6f6f6 100%); /* FF3.6+ */
        background: -webkit-gradient(linear, left top, left bottom, color-stop(0%,#e0e0e0), color-stop(100%,#f6f6f6)); /* Chrome,Safari4+ */
        background: -webkit-linear-gradient(top,  #e0e0e0 0%,#f6f6f6 100%); /* Chrome10+,Safari5.1+ */
        background: -o-linear-gradient(top,  #e0e0e0 0%,#f6f6f6 100%); /* Opera 11.10+ */
        background: -ms-linear-gradient(top,  #e0e0e0 0%,#f6f6f6 100%); /* IE10+ */
        background: linear-gradient(to bottom,  #e0e0e0 0%,#f6f6f6 100%); /* W3C */
        filter: progid:DXImageTransform.Microsoft.gradient( startColorstr='#e0e0e0', endColorstr='#f6f6f6',GradientType=0 ); /* IE6-9 */
        border-radius: 6px 6px 0 0;
        -moz-border-radius: 6px 6px 0 0;
        -webkit-border-radius: 6px 6px 0 0;
    }
    
    .dd-target:hover {
        background: #f7f7f7; /* Old browsers */
        background: -moz-linear-gradient(top,  #f7f7f7 0%, #ededed 100%); /* FF3.6+ */
        background: -webkit-gradient(linear, left top, left bottom, color-stop(0%,#f7f7f7), color-stop(100%,#ededed)); /* Chrome,Safari4+ */
        background: -webkit-linear-gradient(top,  #f7f7f7 0%,#ededed 100%); /* Chrome10+,Safari5.1+ */
        background: -o-linear-gradient(top,  #f7f7f7 0%,#ededed 100%); /* Opera 11.10+ */
        background: -ms-linear-gradient(top,  #f7f7f7 0%,#ededed 100%); /* IE10+ */
        background: linear-gradient(to bottom,  #f7f7f7 0%,#ededed 100%); /* W3C */
        filter: progid:DXImageTransform.Microsoft.gradient( startColorstr='#f7f7f7', endColorstr='#ededed',GradientType=0 ); /* IE6-9 */
    }
    
    .dd-wrapper {
        position: relative;
        display: inline-block;
    }
    
    .dd-hidden {
        display: none;
    }
    
    .dd-menu {
        position: absolute;
        left: 0;
        right: 0;
        font-weight: normal;
        list-style: none;
        text-decoration: none;
        padding: 0;
        font-size: 14px;
        z-index: 1000;
        border: 1px solid #999;
        margin: -1px 0 0;
        background-color: #f6f6f6;
    }
    
    .dd-menu-option {
        cursor: pointer;
        text-align: center;
        padding: 8px 0;
        border-right: 20px solid rgba(221, 221, 221, 0);
        border-left: 20px solid rgba(221, 221, 221, 0);
        background-color: #f6f6f6;
    }
    
    .dd-menu-option:last-child, .dd-menu {
        border-radius: 0 0 6px 6px;
        -moz-border-radius: 0 0 6px 6px;
        -webkit-border-radius: 0 0 6px 6px;
    }
    
    .dd-menu-option+.dd-menu-option {
        border-top: 1px solid #ccc;
    }
    
    .dd-menu-option:hover {
        background-color: #e0e0e0 !important;
    }
    
    .dd-button-arrow {
        position: relative;
        width: 0;
        height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid #333;
        top: 12px;
        margin: 5px;
    }
    
    .dd-active .dd-button-arrow {
        border-top: none;
        border-bottom: 5px solid #333;
        top: auto;
        bottom: 12px;
    }.blue-button, a.blue-button, a.blue-button:visited {
        display: inline-block;
        padding: 0 15px;
        background: #2f62f1; /* Old browsers */
        background: -moz-linear-gradient(top,  #2f62f1 0%, #133fbb 100%); /* FF3.6+ */
        background: -webkit-gradient(linear, left top, left bottom, color-stop(0%,#2f62f1), color-stop(100%,#133fbb)); /* Chrome,Safari4+ */
        background: -webkit-linear-gradient(top,  #2f62f1 0%,#133fbb 100%); /* Chrome10+,Safari5.1+ */
        background: -o-linear-gradient(top,  #2f62f1 0%,#133fbb 100%); /* Opera 11.10+ */
        background: -ms-linear-gradient(top,  #2f62f1 0%,#133fbb 100%); /* IE10+ */
        background: linear-gradient(to bottom,  #2f62f1 0%,#133fbb 100%); /* W3C */
        filter: progid:DXImageTransform.Microsoft.gradient( startColorstr='#2f62f1', endColorstr='#133fbb',GradientType=0 ); /* IE6-9 */
        color: white;
        text-shadow: rgb(0, 0, 0) 0 -1px 0;
        border: 1px solid #000C97;
        -moz-border-radius: 6px;
        -webkit-border-radius: 6px;
        border-radius: 6px;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 18px;
        line-height: 31px;
        text-align: center;
        cursor: pointer;
        font-weight: 200;
        outline: none;
        text-decoration: none;
        box-shadow: rgba(0, 0, 0, 0.2) 0px 1px 2px 0px;
    }
    
    .blue-button:hover, a.blue-button:hover {
        background: #4a76ef; /* Old browsers */
        background: -moz-linear-gradient(top,  #4a76ef 0%, #375aba 100%); /* FF3.6+ */
        background: -webkit-gradient(linear, left top, left bottom, color-stop(0%,#4a76ef), color-stop(100%,#375aba)); /* Chrome,Safari4+ */
        background: -webkit-linear-gradient(top,  #4a76ef 0%,#375aba 100%); /* Chrome10+,Safari5.1+ */
        background: -o-linear-gradient(top,  #4a76ef 0%,#375aba 100%); /* Opera 11.10+ */
        background: -ms-linear-gradient(top,  #4a76ef 0%,#375aba 100%); /* IE10+ */
        background: linear-gradient(to bottom,  #4a76ef 0%,#375aba 100%); /* W3C */
        filter: progid:DXImageTransform.Microsoft.gradient( startColorstr='#4a76ef', endColorstr='#375aba',GradientType=0 ); /* IE6-9 */
    }.state-picker-button {
        margin-top: 10px;
    }
    
    .state-picker-info-body {
        padding: 10px;
        background-color: #eee;
    }
    
    .state-picker-info-body a {
        margin-right: 10px;
        display: inline-block;
    }
    
    .state-picker-info-arrow {
        position: relative;
        top: 100%;
        right: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-left: 10px solid transparent;
        border-right: 10px solid transparent;
        border-bottom: 10px solid #eee;
        margin-top: 3px;
    }
    
    .state-picker-info-error {
        color: #c00;
    }
            .aquo {
                font-size: 140%;
                font-weight: bold;
                line-height: 0.5;
                color: #f60;
            }
    
            #bvjl {
                font-size: 16px;
                margin: 1em 0 1.5em;
            }
    
            .snip .sl,
            .snip .view_job_link {
                white-space: nowrap;
            }
         </style>
         <table border="0" cellpadding="0" cellspacing="0" id="pageContent" width="100%">
          <tr valign="top">
           <td data-tn-section="refineBy" id="refineresultscol">
            <div id="refineresults">
             <h1>
              <font size="+1">
               data jobs in Austin, TX
              </font>
             </h1>
             <div id="recPromoDisplay" style="display:none;">
             </div>
             <script type="text/javascript">
              call_when_jsall_loaded(function() {
                var recJobLink = new RecJobLink("Recommended Jobs", "recPromoDisplay", "1c6u39dg0a24qd40", "1c6u39dhta24qbca",
                        "US", "en", "",
                        "", null, true);
                recJobLink.onLoad();
            });
             </script>
             <span aria-level="2" role="heading" style="height: 0; overflow: hidden; position: absolute;">
              Filter results by:
             </span>
             <div style="margin-left: 6px; margin-bottom: 1em;">
              Sort by:
              <span class="no-wrap">
               <a href="/jobs?q=data&amp;l=Austin%2C+TX" rel="nofollow">
                relevance
               </a>
               -
               <b>
                date
               </b>
              </span>
             </div>
             <form action="/jobs" id="radius_update" method="get" name="radius_update">
              <input name="q" type="hidden" value="data"/>
              <input name="sort" type="hidden" value="date"/>
              <input name="l" type="hidden" value="Austin, TX"/>
              <label for="distance_selector" onclick="this.form.radius.focus();return false;">
               Distance:
              </label>
              <select id="distance_selector" name="radius" onchange="ptk('radius'); this.form.submit();">
               <option value="0">
                Exact location only
               </option>
               <option value="5">
                within 5 miles
               </option>
               <option value="10">
                within 10 miles
               </option>
               <option value="15">
                within 15 miles
               </option>
               <option selected="" value="25">
                within 25 miles
               </option>
               <option value="50">
                within 50 miles
               </option>
               <option value="100">
                within 100 miles
               </option>
              </select>
              <noscript>
               <input id="r_up" name="r_up" type="submit" value="Go"/>
              </noscript>
             </form>
             <div class="rbSection rbOpen" id="rb_Salary Estimate">
              <div class="rbHeader">
               <span aria-level="3" class="ws_bold" role="heading">
                Salary Estimate
               </span>
              </div>
              <div class="rbsrbo" id="SALARY_rbo">
               <ul class="rbList">
                <li onmousedown="rbptk('rb', 'salest', '1');">
                 <a href="/jobs?q=data+$30,000&amp;l=Austin,+TX&amp;sort=date" rel="nofollow" title="$30,000 (4619)">
                  $30,000
                 </a>
                 (4619)
                </li>
                <li onmousedown="rbptk('rb', 'salest', '2');">
                 <a href="/jobs?q=data+$45,000&amp;l=Austin,+TX&amp;sort=date" rel="nofollow" title="$45,000 (3498)">
                  $45,000
                 </a>
                 (3498)
                </li>
                <li onmousedown="rbptk('rb', 'salest', '3');">
                 <a href="/jobs?q=data+$55,000&amp;l=Austin,+TX&amp;sort=date" rel="nofollow" title="$55,000 (2867)">
                  $55,000
                 </a>
                 (2867)
                </li>
                <li onmousedown="rbptk('rb', 'salest', '4');">
                 <a href="/jobs?q=data+$75,000&amp;l=Austin,+TX&amp;sort=date" rel="nofollow" title="$75,000 (1769)">
                  $75,000
                 </a>
                 (1769)
                </li>
                <li onmousedown="rbptk('rb', 'salest', '5');">
                 <a href="/jobs?q=data+$95,000&amp;l=Austin,+TX&amp;sort=date" rel="nofollow" title="$95,000 (948)">
                  $95,000
                 </a>
                 (948)
                </li>
               </ul>
              </div>
             </div>
             <div class="rbSection rbOpen" id="rb_Job Type">
              <div class="rbHeader">
               <span aria-level="3" class="ws_bold" role="heading">
                Job Type
               </span>
              </div>
              <div class="rbsrbo" id="JOB_TYPE_rbo">
               <ul class="rbList">
                <li onmousedown="rbptk('rb', 'jobtype', '1');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;jt=fulltime&amp;sort=date" rel="nofollow" title="Full-time (4910)">
                  Full-time
                 </a>
                 (4910)
                </li>
                <li onmousedown="rbptk('rb', 'jobtype', '2');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;jt=parttime&amp;sort=date" rel="nofollow" title="Part-time (328)">
                  Part-time
                 </a>
                 (328)
                </li>
                <li onmousedown="rbptk('rb', 'jobtype', '3');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;jt=contract&amp;sort=date" rel="nofollow" title="Contract (307)">
                  Contract
                 </a>
                 (307)
                </li>
                <li onmousedown="rbptk('rb', 'jobtype', '4');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;jt=temporary&amp;sort=date" rel="nofollow" title="Temporary (183)">
                  Temporary
                 </a>
                 (183)
                </li>
                <li onmousedown="rbptk('rb', 'jobtype', '5');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;jt=internship&amp;sort=date" rel="nofollow" title="Internship (171)">
                  Internship
                 </a>
                 (171)
                </li>
                <li onmousedown="rbptk('rb', 'jobtype', '6');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;jt=commission&amp;sort=date" rel="nofollow" title="Commission (162)">
                  Commission
                 </a>
                 (162)
                </li>
               </ul>
              </div>
             </div>
             <div class="rbSection rbOpen" id="rb_Location">
              <div class="rbHeader">
               <span aria-level="3" class="ws_bold" role="heading">
                Location
               </span>
              </div>
              <div class="rbsrbo" id="LOCATION_rbo">
               <ul class="rbList">
                <li onmousedown="rbptk('rb', 'loc', '1');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbl=Austin,+TX&amp;jlid=d2a39b6d57d82344&amp;sort=date" rel="nofollow" title="Austin, TX (5017)">
                  Austin, TX
                 </a>
                 (5017)
                </li>
                <li onmousedown="rbptk('rb', 'loc', '2');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbl=Round+Rock,+TX&amp;jlid=6e1c0638d6049487&amp;sort=date" rel="nofollow" title="Round Rock, TX (244)">
                  Round Rock, TX
                 </a>
                 (244)
                </li>
                <li onmousedown="rbptk('rb', 'loc', '3');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbl=Cedar+Park,+TX&amp;jlid=d9f52643c6302381&amp;sort=date" rel="nofollow" title="Cedar Park, TX (41)">
                  Cedar Park, TX
                 </a>
                 (41)
                </li>
                <li onmousedown="rbptk('rb', 'loc', '4');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbl=Pflugerville,+TX&amp;jlid=0059bed8953e6e4b&amp;sort=date" rel="nofollow" title="Pflugerville, TX (30)">
                  Pflugerville, TX
                 </a>
                 (30)
                </li>
                <li onmousedown="rbptk('rb', 'loc', '5');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbl=Leander,+TX&amp;jlid=d40ca9e2511e294d&amp;sort=date" rel="nofollow" title="Leander, TX (29)">
                  Leander, TX
                 </a>
                 (29)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'loc', '6');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbl=Kyle,+TX&amp;jlid=751e09d1357ead4d&amp;sort=date" rel="nofollow" title="Kyle, TX (23)">
                  Kyle, TX
                 </a>
                 (23)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'loc', '7');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbl=Buda,+TX&amp;jlid=ab7a19d4d802e089&amp;sort=date" rel="nofollow" title="Buda, TX (22)">
                  Buda, TX
                 </a>
                 (22)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'loc', '8');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbl=Lakeway,+TX&amp;jlid=626e1c172da5040a&amp;sort=date" rel="nofollow" title="Lakeway, TX (13)">
                  Lakeway, TX
                 </a>
                 (13)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'loc', '9');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbl=South+Austin,+TX&amp;jlid=e2a2a5c0f4f84192&amp;sort=date" rel="nofollow" title="South Austin, TX (8)">
                  South Austin, TX
                 </a>
                 (8)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'loc', '10');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbl=Manor,+TX&amp;jlid=9f81d3f269ba08d3&amp;sort=date" rel="nofollow" title="Manor, TX (8)">
                  Manor, TX
                 </a>
                 (8)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'loc', '11');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbl=Hutto,+TX&amp;jlid=a4d150883db37ccf&amp;sort=date" rel="nofollow" title="Hutto, TX (7)">
                  Hutto, TX
                 </a>
                 (7)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'loc', '12');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbl=Bee+Cave,+TX&amp;jlid=5a21fea281b46dbf&amp;sort=date" rel="nofollow" title="Bee Cave, TX (6)">
                  Bee Cave, TX
                 </a>
                 (6)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'loc', '13');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbl=Georgetown,+TX&amp;jlid=9859371db9b7e630&amp;sort=date" rel="nofollow" title="Georgetown, TX (4)">
                  Georgetown, TX
                 </a>
                 (4)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'loc', '14');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbl=Dripping+Springs,+TX&amp;jlid=1fd0e53df7763808&amp;sort=date" rel="nofollow" title="Dripping Springs, TX (4)">
                  Dripping Springs, TX
                 </a>
                 (4)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'loc', '15');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbl=North+Austin,+TX&amp;jlid=6ec52361b934d33d&amp;sort=date" rel="nofollow" title="North Austin, TX (4)">
                  North Austin, TX
                 </a>
                 (4)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb',
                'loc',
                '16');">
                 <a href="/q-Data-jobs.html">
                  Data jobs
                 </a>
                 nationwide
                </li>
               </ul>
               <div class="more_link">
                <span onclick="showAllRefinements('rb_Location'); return false;" onkeyup="if (event.keyCode == 13) showAllRefinements('rb_Location'); return false;" tabindex="0">
                 more »
                </span>
               </div>
              </div>
             </div>
             <div class="rbSection rbOpen" id="rb_Company">
              <div class="rbHeader">
               <span aria-level="3" class="ws_bold" role="heading">
                Company
               </span>
              </div>
              <div class="rbsrbo" id="COMPANY_rbo">
               <ul class="rbList">
                <li onmousedown="rbptk('rb', 'cmp', '1');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbc=Health+%26+Human+Services+Comm&amp;jcid=ec038939be4318c8&amp;sort=date" rel="nofollow" title="Health &amp; Human Services Comm (179)">
                  Health &amp; Human Services Comm
                 </a>
                 (179)
                </li>
                <li onmousedown="rbptk('rb', 'cmp', '2');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbc=DELL&amp;jcid=0918a251e6902f97&amp;sort=date" rel="nofollow" title="DELL (102)">
                  DELL
                 </a>
                 (102)
                </li>
                <li onmousedown="rbptk('rb', 'cmp', '3');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbc=University+of+Texas+at+Austin&amp;jcid=f7282ad3490137c7&amp;sort=date" rel="nofollow" title="University of Texas at Austin (94)">
                  University of Texas at Austin
                 </a>
                 (94)
                </li>
                <li onmousedown="rbptk('rb', 'cmp', '4');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbc=Seton+Family+of+Hospitals,+TX&amp;jcid=62bd2af080f2c7ac&amp;sort=date" rel="nofollow" title="Seton Family of Hospitals, TX (78)">
                  Seton Family of Hospitals, TX
                 </a>
                 (78)
                </li>
                <li onmousedown="rbptk('rb', 'cmp', '5');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbc=Indeed&amp;jcid=d6ef41e202aa2c0b&amp;sort=date" rel="nofollow" title="Indeed (76)">
                  Indeed
                 </a>
                 (76)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'cmp', '6');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbc=City+of+Austin&amp;jcid=651f65cb69970e95&amp;sort=date" rel="nofollow" title="City of Austin (67)">
                  City of Austin
                 </a>
                 (67)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'cmp', '7');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbc=Advanced+Micro+Devices,+Inc.&amp;jcid=14872b5be04c1bd6&amp;sort=date" rel="nofollow" title="Advanced Micro Devices, Inc. (55)">
                  Advanced Micro Devices, Inc.
                 </a>
                 (55)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'cmp', '8');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbc=The+HT+Group&amp;jcid=f8760d24ddfe853a&amp;sort=date" rel="nofollow" title="The HT Group (54)">
                  The HT Group
                 </a>
                 (54)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'cmp', '9');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbc=Dept+of+State+Health+Services&amp;jcid=0ab4fcb6780213bd&amp;sort=date" rel="nofollow" title="Dept of State Health Services (48)">
                  Dept of State Health Services
                 </a>
                 (48)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'cmp', '10');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbc=Facebook&amp;jcid=1639254ea84748b5&amp;sort=date" rel="nofollow" title="Facebook (48)">
                  Facebook
                 </a>
                 (48)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'cmp', '11');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbc=Apple&amp;jcid=c1099851e9794854&amp;sort=date" rel="nofollow" title="Apple (47)">
                  Apple
                 </a>
                 (47)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'cmp', '12');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbc=Whole+Foods+Market&amp;jcid=ea10be8bb95dc939&amp;sort=date" rel="nofollow" title="Whole Foods Market (38)">
                  Whole Foods Market
                 </a>
                 (38)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'cmp', '13');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbc=Visa&amp;jcid=a3f737e511d9fc8c&amp;sort=date" rel="nofollow" title="Visa (35)">
                  Visa
                 </a>
                 (35)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'cmp', '14');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbc=Forcepoint&amp;jcid=7edcf937e2e69aed&amp;sort=date" rel="nofollow" title="Forcepoint (34)">
                  Forcepoint
                 </a>
                 (34)
                </li>
                <li class="moreLi" onmousedown="rbptk('rb', 'cmp', '15');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;rbc=Texas+Department+of+Transportation&amp;jcid=e787d33bd6f3ef98&amp;sort=date" rel="nofollow" title="Texas Department of Transportation (33)">
                  Texas Department of Transportation
                 </a>
                 (33)
                </li>
               </ul>
               <div class="more_link">
                <span onclick="showAllRefinements('rb_Company'); return false;" onkeyup="if (event.keyCode == 13) showAllRefinements('rb_Company'); return false;" tabindex="0">
                 more »
                </span>
               </div>
              </div>
             </div>
             <div class="rbSection rbOpen" id="rb_Experience Level">
              <div class="rbHeader">
               <span aria-level="3" class="ws_bold" role="heading">
                Experience Level
               </span>
              </div>
              <div class="rbsrbo" id="EXP_LVL_rbo">
               <ul class="rbList">
                <li onmousedown="rbptk('rb', 'explvl', '1');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;explvl=mid_level&amp;sort=date" rel="nofollow" title="Mid Level (2147)">
                  Mid Level
                 </a>
                 (2147)
                </li>
                <li onmousedown="rbptk('rb', 'explvl', '2');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;explvl=entry_level&amp;sort=date" rel="nofollow" title="Entry Level (2145)">
                  Entry Level
                 </a>
                 (2145)
                </li>
                <li onmousedown="rbptk('rb', 'explvl', '3');">
                 <a href="/jobs?q=data&amp;l=Austin,+TX&amp;explvl=senior_level&amp;sort=date" rel="nofollow" title="Senior Level (441)">
                  Senior Level
                 </a>
                 (441)
                </li>
               </ul>
              </div>
             </div>
            </div>
           </td>
           <td id="resultsCol">
            <div class="messageContainer">
             <script type="text/javascript">
              function setRefineByCookie(refineByTypes) {
            var expires = new Date();
            expires.setTime(expires.getTime() + (10 * 1000));
            for (var i = 0; i < refineByTypes.length; i++) {
                setCookie(refineByTypes[i], "1", expires);
            }
        }
             </script>
            </div>
            <style type="text/css">
             #increased_radius_result {
            font-size: 16px;
            font-style: italic;
        }
        #original_radius_result{
            font-size: 13px;
            font-style: italic;
            color: #666666;
        }
            </style>
            <div class="resultsTop">
             <div id="searchCount">
              Page 1 of 5,474 jobs
             </div>
             <div data-tn-section="resumePromo" id="resumePromo">
              <a aria-hidden="true" href="/promo/resume" onclick="this.href = appendParamsOnce( this.href, '?from=serptop3&amp;subfrom=resprmrtop&amp;trk.origin=jobsearch&amp;trk.variant=resprmrtop&amp;trk.tk=1c6u39dhta24qbca')" tabindex="-1">
               <span aria-label="post resume icon" class="new-ico" role="img">
               </span>
              </a>
              <a class="resume-promo-link" href="/promo/resume" onclick="this.href = appendParamsOnce( this.href, '?from=serptop3&amp;subfrom=resprmrtop&amp;trk.origin=jobsearch&amp;trk.variant=resprmrtop&amp;trk.tk=1c6u39dhta24qbca')">
               <b>
                Upload your resume
               </b>
              </a>
              - Let employers find you
             </div>
            </div>
            <script type="text/javascript">
             window['sjl'] = "bzugp0PeDkk";
            </script>
            <style type="text/css">
             .bzugp0PeDkk { margin: 0 0 6px 0; padding: 0; _zoom:100%; border: 0; background-color: #fff; }
    		.bzugp0PeDkk .jobtitle { white-space: nowrap; float:left; _float: none; }
            .bzugp0PeDkk .sdn { color: #CD29C0; }
    		.f7q4pgaN .brdr { margin-top: 12px; }
    		.KQox8b8lBh .brdr { margin-bottom: 12px; }
    		@media only screen and (min-height:780px) {
                .f7q4pgaN { margin-bottom: 9px; }
    			.KQox8b8lBh .brdr,
                .PgFyEJcyC7,
    			.f7q4pgaN .brdr { margin-bottom: 9px; margin-top: 9px; }
    		}
            </style>
            <style type="text/css">
             .result-tab:empty {margin-top: 0;}
                .f7q4pgaN {
                    margin-bottom: 0;
                }
                @media only screen and (min-height:780px) {
                    .f7q4pgaN {
                        margin-bottom: 0;
                    }
                }
            </style>
            <div>
            </div>
            <a id="jobPostingsAnchor" tabindex="-1">
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
             <span class="company">
              <a href="/cmp/Absolute-Software" onmousedown="this.href = appendParamsOnce(this.href, 'from=SERP&amp;campaignid=serp-linkcompanyname&amp;fromjk=0829198f649e9c08&amp;jcid=7c30762e902763ee')" rel="noopener" target="_blank">
               Absolute Software
              </a>
             </span>
             -
             <a class="ratingsLabel" data-tn-element="reviewStars" data-tn-variant="cmplinktst2" href="/cmp/Absolute-Software/reviews" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=cmplinktst2&amp;from=SERP&amp;jt=Data+Entry+Associate&amp;fromjk=0829198f649e9c08&amp;jcid=7c30762e902763ee');" rel="noopener" target="_blank" title="Absolute Software reviews">
              <span class="ratings">
               <span class="rating" style="width:42.0px">
                <!-- -->
               </span>
              </span>
              <span class="slNoUnderline">
               13 reviews
              </span>
             </a>
             -
             <span class="location">
              Austin, TX 78758
              <span style="font-size: smaller">
               (North Austin area)
              </span>
             </span>
             <table border="0" cellpadding="0" cellspacing="0">
              <tr>
               <td class="snip">
                <div class="">
                 <span class="summary">
                  Ultimately, a successful
                  <b>
                   Data
                  </b>
                  Entry Associate will be responsible for maintaining accurate, up-to-date and usable
                  <b>
                   data
                  </b>
                  in Salesforce....
                 </span>
                </div>
                <div class="result-link-bar-container">
                 <div class="result-link-bar">
                  <span class="date">
                   Just posted
                  </span>
                  <span class="tt_set" id="tt_set_0">
                   -
                   <a class="sl resultLink save-job-link " href="#" id="sj_0829198f649e9c08" onclick="changeJobState('0829198f649e9c08', 'save', 'linkbar', false, ''); return false;" title="Save this job to my.indeed">
                    save job
                   </a>
                   -
                   <a class="sl resultLink more-link " href="#" id="tog_0" onclick="toggleMoreLinks('0829198f649e9c08'); return false;">
                    more...
                   </a>
                  </span>
                  <div class="edit_note_content" id="editsaved2_0829198f649e9c08" style="display:none;">
                  </div>
                  <script>
                   if (!window['result_0829198f649e9c08']) {window['result_0829198f649e9c08'] = {};}window['result_0829198f649e9c08']['showSource'] = false; window['result_0829198f649e9c08']['source'] = "Absolute Software"; window['result_0829198f649e9c08']['loggedIn'] = false; window['result_0829198f649e9c08']['showMyJobsLinks'] = false;window['result_0829198f649e9c08']['undoAction'] = "unsave";window['result_0829198f649e9c08']['relativeJobAge'] = "Just posted";window['result_0829198f649e9c08']['jobKey'] = "0829198f649e9c08"; window['result_0829198f649e9c08']['myIndeedAvailable'] = true; window['result_0829198f649e9c08']['showMoreActionsLink'] = window['result_0829198f649e9c08']['showMoreActionsLink'] || true; window['result_0829198f649e9c08']['resultNumber'] = 0; window['result_0829198f649e9c08']['jobStateChangedToSaved'] = false; window['result_0829198f649e9c08']['searchState'] = "q=data&amp;l=Austin%2C+TX&amp;sort=date"; window['result_0829198f649e9c08']['basicPermaLink'] = "https://www.indeed.com"; window['result_0829198f649e9c08']['saveJobFailed'] = false; window['result_0829198f649e9c08']['removeJobFailed'] = false; window['result_0829198f649e9c08']['requestPending'] = false; window['result_0829198f649e9c08']['notesEnabled'] = true; window['result_0829198f649e9c08']['currentPage'] = "serp"; window['result_0829198f649e9c08']['sponsored'] = false;window['result_0829198f649e9c08']['reportJobButtonEnabled'] = false; window['result_0829198f649e9c08']['showMyJobsHired'] = false; window['result_0829198f649e9c08']['showSaveForSponsored'] = false; window['result_0829198f649e9c08']['showJobAge'] = true;
                  </script>
                 </div>
                </div>
                <div class="tab-container">
                 <div class="more-links-container result-tab" id="tt_display_0" style="display:none;">
                  <a class="close-link closeLink" href="#" onclick="toggleMoreLinks('0829198f649e9c08'); return false;" title="Close">
                  </a>
                  <div class="more_actions" id="more_0">
                   <ul>
                    <li>
                     <span class="mat">
                      View all
                      <a href="/q-Absolute-Software-l-Austin,-TX-jobs.html" rel="nofollow">
                       Absolute Software jobs in Austin, TX
                      </a>
                      -
                      <a href="/l-Austin,-TX-jobs.html">
                       Austin jobs
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Salary Search:
                      <a href="/salaries/Data-Entry-Clerk-Salaries,-Austin-TX" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=serp-more&amp;fromjk=0829198f649e9c08&amp;from=serp-more');">
                       Data Entry Clerk salaries in Austin, TX
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Learn more about working at
                      <a href="/cmp/Absolute-Software" onmousedown="this.href = appendParamsOnce(this.href, '?fromjk=0829198f649e9c08&amp;from=serp-more&amp;campaignid=serp-more&amp;jcid=7c30762e902763ee');">
                       Absolute Software
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      <a href="/cmp/Absolute-Software/faq" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=0829198f649e9c08&amp;jcid=7c30762e902763ee');">
                       Absolute Software questions about work, benefits, interviews and hiring process:
                      </a>
                      <ul>
                       <li>
                        <a href="/cmp/Absolute-Software/faq/what-tips-or-advice-would-you-give-to-someone-interviewing-at-absolute-software?quid=1bku8b35taki09bv" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=0829198f649e9c08&amp;jcid=7c30762e902763ee');">
                         What tips or advice would you give to someone interviewing at Absolute S...
                        </a>
                       </li>
                       <li>
                        <a href="/cmp/Absolute-Software/faq/what-is-the-vacation-policy-like-how-many-vacation-days-do-you-get-per-year?quid=1bku8b390ak64atu" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=0829198f649e9c08&amp;jcid=7c30762e902763ee');">
                         What is the vacation policy like? How many vacation days do you get per ...
                        </a>
                       </li>
                      </ul>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Related forums:
                      <a href="/forum/cmp/Absolute-Software.html">
                       Absolute Software
                      </a>
                      -
                      <a href="/forum/loc/Austin-Texas.html">
                       Austin, Texas
                      </a>
                     </span>
                    </li>
                   </ul>
                  </div>
                 </div>
                 <div class="dya-container result-tab">
                 </div>
                 <div class="tellafriend-container result-tab email_job_content">
                 </div>
                 <div class="sign-in-container result-tab">
                 </div>
                 <div class="notes-container result-tab">
                 </div>
                </div>
               </td>
              </tr>
             </table>
            </div>
            <div class=" row result" data-jk="ed6c0f013d00569e" data-tn-component="organicJob" data-tu="" id="p_ed6c0f013d00569e">
             <h2 class="jobtitle" id="jl_ed6c0f013d00569e">
              <a class="turnstileLink" data-tn-element="jobTitle" href="/rc/clk?jk=ed6c0f013d00569e&amp;fccid=ff5d5c17ad67de08&amp;vjs=3" onclick="setRefineByCookie([]); return rclk(this,jobmap[1],true,0);" onmousedown="return rclk(this,jobmap[1],0);" rel="noopener nofollow" target="_blank" title="Business Intelligence Developer/ Analyst">
               Business Intelligence Developer/ Analyst
              </a>
             </h2>
             <span class="company">
              <a href="/cmp/Keller-Williams-Realty" onmousedown="this.href = appendParamsOnce(this.href, 'from=SERP&amp;campaignid=serp-linkcompanyname&amp;fromjk=ed6c0f013d00569e&amp;jcid=920fd18472f1d3fe')" rel="noopener" target="_blank">
               Keller Williams
              </a>
             </span>
             -
             <a class="ratingsLabel" data-tn-element="reviewStars" data-tn-variant="cmplinktst2" href="/cmp/Keller-Williams-Realty/reviews" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=cmplinktst2&amp;from=SERP&amp;jt=Business+Intelligence+Developer%5C%2F+Analyst&amp;fromjk=ed6c0f013d00569e&amp;jcid=920fd18472f1d3fe');" rel="noopener" target="_blank" title="Keller Williams reviews">
              <span class="ratings">
               <span class="rating" style="width:53.4px">
                <!-- -->
               </span>
              </span>
              <span class="slNoUnderline">
               2,200 reviews
              </span>
             </a>
             -
             <span class="location">
              Austin, TX 78746
             </span>
             <table border="0" cellpadding="0" cellspacing="0">
              <tr>
               <td class="snip">
                <div class="">
                 <span class="summary">
                  Develop an advanced understanding of our
                  <b>
                   data
                  </b>
                  . The reliance on
                  <b>
                   data
                  </b>
                  and business analytics is increasing exponentially as business owners realize the power of...
                 </span>
                </div>
                <div class="iaP">
                 <span class="iaLabel">
                  Easily apply
                 </span>
                </div>
                <div class="result-link-bar-container">
                 <div class="result-link-bar">
                  <span class="date">
                   Just posted
                  </span>
                  <span class="tt_set" id="tt_set_1">
                   -
                   <a class="sl resultLink save-job-link " href="#" id="sj_ed6c0f013d00569e" onclick="changeJobState('ed6c0f013d00569e', 'save', 'linkbar', false, ''); return false;" title="Save this job to my.indeed">
                    save job
                   </a>
                   -
                   <a class="sl resultLink more-link " href="#" id="tog_1" onclick="toggleMoreLinks('ed6c0f013d00569e'); return false;">
                    more...
                   </a>
                  </span>
                  <div class="edit_note_content" id="editsaved2_ed6c0f013d00569e" style="display:none;">
                  </div>
                  <script>
                   if (!window['result_ed6c0f013d00569e']) {window['result_ed6c0f013d00569e'] = {};}window['result_ed6c0f013d00569e']['showSource'] = false; window['result_ed6c0f013d00569e']['source'] = "Keller Williams"; window['result_ed6c0f013d00569e']['loggedIn'] = false; window['result_ed6c0f013d00569e']['showMyJobsLinks'] = false;window['result_ed6c0f013d00569e']['undoAction'] = "unsave";window['result_ed6c0f013d00569e']['relativeJobAge'] = "Just posted";window['result_ed6c0f013d00569e']['jobKey'] = "ed6c0f013d00569e"; window['result_ed6c0f013d00569e']['myIndeedAvailable'] = true; window['result_ed6c0f013d00569e']['showMoreActionsLink'] = window['result_ed6c0f013d00569e']['showMoreActionsLink'] || true; window['result_ed6c0f013d00569e']['resultNumber'] = 1; window['result_ed6c0f013d00569e']['jobStateChangedToSaved'] = false; window['result_ed6c0f013d00569e']['searchState'] = "q=data&amp;l=Austin%2C+TX&amp;sort=date"; window['result_ed6c0f013d00569e']['basicPermaLink'] = "https://www.indeed.com"; window['result_ed6c0f013d00569e']['saveJobFailed'] = false; window['result_ed6c0f013d00569e']['removeJobFailed'] = false; window['result_ed6c0f013d00569e']['requestPending'] = false; window['result_ed6c0f013d00569e']['notesEnabled'] = true; window['result_ed6c0f013d00569e']['currentPage'] = "serp"; window['result_ed6c0f013d00569e']['sponsored'] = false;window['result_ed6c0f013d00569e']['reportJobButtonEnabled'] = false; window['result_ed6c0f013d00569e']['showMyJobsHired'] = false; window['result_ed6c0f013d00569e']['showSaveForSponsored'] = false; window['result_ed6c0f013d00569e']['showJobAge'] = true;
                  </script>
                 </div>
                </div>
                <div class="tab-container">
                 <div class="more-links-container result-tab" id="tt_display_1" style="display:none;">
                  <a class="close-link closeLink" href="#" onclick="toggleMoreLinks('ed6c0f013d00569e'); return false;" title="Close">
                  </a>
                  <div class="more_actions" id="more_1">
                   <ul>
                    <li>
                     <span class="mat">
                      View all
                      <a href="/q-Keller-Williams-l-Austin,-TX-jobs.html" rel="nofollow">
                       Keller Williams jobs in Austin, TX
                      </a>
                      -
                      <a href="/l-Austin,-TX-jobs.html">
                       Austin jobs
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Salary Search:
                      <a href="/salaries/Business-Intelligence-Developer-Salaries,-Austin-TX" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=serp-more&amp;fromjk=ed6c0f013d00569e&amp;from=serp-more');">
                       Business Intelligence Developer salaries in Austin, TX
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Learn more about working at
                      <a href="/cmp/Keller-Williams-Realty" onmousedown="this.href = appendParamsOnce(this.href, '?fromjk=ed6c0f013d00569e&amp;from=serp-more&amp;campaignid=serp-more&amp;jcid=920fd18472f1d3fe');">
                       Keller Williams
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      <a href="/cmp/Keller-Williams-Realty/faq" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=ed6c0f013d00569e&amp;jcid=920fd18472f1d3fe');">
                       Keller Williams questions about work, benefits, interviews and hiring process:
                      </a>
                      <ul>
                       <li>
                        <a href="/cmp/Keller-Williams-Realty/faq/what-is-the-interview-process-like?quid=1an0pouki5ncm9v2" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=ed6c0f013d00569e&amp;jcid=920fd18472f1d3fe');">
                         What is the interview process like?
                        </a>
                       </li>
                       <li>
                        <a href="/cmp/Keller-Williams-Realty/faq/does-keller-williams-provide-support-e-g-mentoring-training-courses-lead-generation-assistance-etc-to-part-time-real-estate-agents-or-are-you?quid=1auvpsm0i5naibch" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=ed6c0f013d00569e&amp;jcid=920fd18472f1d3fe');">
                         Does Keller Williams provide support (e.g., mentoring, training courses,...
                        </a>
                       </li>
                      </ul>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Related forums:
                      <a href="/forum/cmp/Keller-Williams-Realty.html">
                       Keller Williams Realty
                      </a>
                      -
                      <a href="/forum/loc/Austin-Texas.html">
                       Austin, Texas
                      </a>
                     </span>
                    </li>
                   </ul>
                  </div>
                 </div>
                 <div class="dya-container result-tab">
                 </div>
                 <div class="tellafriend-container result-tab email_job_content">
                 </div>
                 <div class="sign-in-container result-tab">
                 </div>
                 <div class="notes-container result-tab">
                 </div>
                </div>
               </td>
              </tr>
             </table>
            </div>
            <div class=" row result" data-jk="5c07dc6626afe971" data-tn-component="organicJob" data-tu="" id="p_5c07dc6626afe971">
             <h2 class="jobtitle" id="jl_5c07dc6626afe971">
              <a class="turnstileLink" data-tn-element="jobTitle" href="/rc/clk?jk=5c07dc6626afe971&amp;fccid=a3f737e511d9fc8c&amp;vjs=3" onclick="setRefineByCookie([]); return rclk(this,jobmap[2],true,0);" onmousedown="return rclk(this,jobmap[2],0);" rel="noopener nofollow" target="_blank" title="HR Reporting &amp; Analytics Manager">
               HR Reporting &amp; Analytics Manager
              </a>
             </h2>
             <span class="company">
              <a href="/cmp/Visa" onmousedown="this.href = appendParamsOnce(this.href, 'from=SERP&amp;campaignid=serp-linkcompanyname&amp;fromjk=5c07dc6626afe971&amp;jcid=a3f737e511d9fc8c')" rel="noopener" target="_blank">
               Visa
              </a>
             </span>
             -
             <a class="ratingsLabel" data-tn-element="reviewStars" data-tn-variant="cmplinktst2" href="/cmp/Visa/reviews" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=cmplinktst2&amp;from=SERP&amp;jt=HR+Reporting+%26+Analytics+Manager&amp;fromjk=5c07dc6626afe971&amp;jcid=a3f737e511d9fc8c');" rel="noopener" target="_blank" title="Visa reviews">
              <span class="ratings">
               <span class="rating" style="width:51.0px">
                <!-- -->
               </span>
              </span>
              <span class="slNoUnderline">
               525 reviews
              </span>
             </a>
             -
             <span class="location">
              Austin, TX
             </span>
             <table border="0" cellpadding="0" cellspacing="0">
              <tr>
               <td class="snip">
                <div class="">
                 <span class="summary">
                  Certification or education in
                  <b>
                   data
                  </b>
                  privacy/protection preferred. Reporting &amp; Analytics team members are required to understand
                  <b>
                   data
                  </b>
                  from multiple transaction...
                 </span>
                </div>
                <div class="result-link-bar-container">
                 <div class="result-link-bar">
                  <span class="date">
                   Just posted
                  </span>
                  <span class="tt_set" id="tt_set_2">
                   -
                   <a class="sl resultLink save-job-link " href="#" id="sj_5c07dc6626afe971" onclick="changeJobState('5c07dc6626afe971', 'save', 'linkbar', false, ''); return false;" title="Save this job to my.indeed">
                    save job
                   </a>
                   -
                   <a class="sl resultLink more-link " href="#" id="tog_2" onclick="toggleMoreLinks('5c07dc6626afe971'); return false;">
                    more...
                   </a>
                  </span>
                  <div class="edit_note_content" id="editsaved2_5c07dc6626afe971" style="display:none;">
                  </div>
                  <script>
                   if (!window['result_5c07dc6626afe971']) {window['result_5c07dc6626afe971'] = {};}window['result_5c07dc6626afe971']['showSource'] = false; window['result_5c07dc6626afe971']['source'] = "Visa"; window['result_5c07dc6626afe971']['loggedIn'] = false; window['result_5c07dc6626afe971']['showMyJobsLinks'] = false;window['result_5c07dc6626afe971']['undoAction'] = "unsave";window['result_5c07dc6626afe971']['relativeJobAge'] = "Just posted";window['result_5c07dc6626afe971']['jobKey'] = "5c07dc6626afe971"; window['result_5c07dc6626afe971']['myIndeedAvailable'] = true; window['result_5c07dc6626afe971']['showMoreActionsLink'] = window['result_5c07dc6626afe971']['showMoreActionsLink'] || true; window['result_5c07dc6626afe971']['resultNumber'] = 2; window['result_5c07dc6626afe971']['jobStateChangedToSaved'] = false; window['result_5c07dc6626afe971']['searchState'] = "q=data&amp;l=Austin%2C+TX&amp;sort=date"; window['result_5c07dc6626afe971']['basicPermaLink'] = "https://www.indeed.com"; window['result_5c07dc6626afe971']['saveJobFailed'] = false; window['result_5c07dc6626afe971']['removeJobFailed'] = false; window['result_5c07dc6626afe971']['requestPending'] = false; window['result_5c07dc6626afe971']['notesEnabled'] = true; window['result_5c07dc6626afe971']['currentPage'] = "serp"; window['result_5c07dc6626afe971']['sponsored'] = false;window['result_5c07dc6626afe971']['reportJobButtonEnabled'] = false; window['result_5c07dc6626afe971']['showMyJobsHired'] = false; window['result_5c07dc6626afe971']['showSaveForSponsored'] = false; window['result_5c07dc6626afe971']['showJobAge'] = true;
                  </script>
                 </div>
                </div>
                <div class="tab-container">
                 <div class="more-links-container result-tab" id="tt_display_2" style="display:none;">
                  <a class="close-link closeLink" href="#" onclick="toggleMoreLinks('5c07dc6626afe971'); return false;" title="Close">
                  </a>
                  <div class="more_actions" id="more_2">
                   <ul>
                    <li>
                     <span class="mat">
                      View all
                      <a href="/q-Visa-l-Austin,-TX-jobs.html" rel="nofollow">
                       Visa jobs in Austin, TX
                      </a>
                      -
                      <a href="/l-Austin,-TX-jobs.html">
                       Austin jobs
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Salary Search:
                      <a href="/salaries/Human-Resources-Manager-Salaries,-Austin-TX" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=serp-more&amp;fromjk=5c07dc6626afe971&amp;from=serp-more');">
                       Human Resources Manager salaries in Austin, TX
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Learn more about working at
                      <a href="/cmp/Visa" onmousedown="this.href = appendParamsOnce(this.href, '?fromjk=5c07dc6626afe971&amp;from=serp-more&amp;campaignid=serp-more&amp;jcid=a3f737e511d9fc8c');">
                       Visa
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      <a href="/cmp/Visa/faq" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=5c07dc6626afe971&amp;jcid=a3f737e511d9fc8c');">
                       Visa questions about work, benefits, interviews and hiring process:
                      </a>
                      <ul>
                       <li>
                        <a href="/cmp/Visa/faq/what-is-the-work-environment-and-culture-like-at-visa?quid=1aqlgei7gas3pfut" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=5c07dc6626afe971&amp;jcid=a3f737e511d9fc8c');">
                         What is the work environment and culture like at Visa?
                        </a>
                       </li>
                       <li>
                        <a href="/cmp/Visa/faq/what-is-the-dress-code-i-know-many-it-companies-are-doing-away-with-business-casual?quid=1b6qbkjr6aqh6921" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=5c07dc6626afe971&amp;jcid=a3f737e511d9fc8c');">
                         What is the dress code?  I know many IT companies are doing away with bu...
                        </a>
                       </li>
                      </ul>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Related forums:
                      <a href="/forum/loc/Austin-Texas.html">
                       Austin, Texas
                      </a>
                      -
                      <a href="/forum/cmp/Visa.html">
                       Visa
                      </a>
                     </span>
                    </li>
                   </ul>
                  </div>
                 </div>
                 <div class="dya-container result-tab">
                 </div>
                 <div class="tellafriend-container result-tab email_job_content">
                 </div>
                 <div class="sign-in-container result-tab">
                 </div>
                 <div class="notes-container result-tab">
                 </div>
                </div>
               </td>
              </tr>
             </table>
            </div>
            <div class=" row result" data-jk="d42da7a52cf52852" data-tn-component="organicJob" data-tu="" id="p_d42da7a52cf52852">
             <h2 class="jobtitle" id="jl_d42da7a52cf52852">
              <a class="turnstileLink" data-tn-element="jobTitle" href="/rc/clk?jk=d42da7a52cf52852&amp;fccid=e62aa8d5bb020acb&amp;vjs=3" onclick="setRefineByCookie([]); return rclk(this,jobmap[3],true,0);" onmousedown="return rclk(this,jobmap[3],0);" rel="noopener nofollow" target="_blank" title="Marketing Data Analyst">
               Marketing
               <b>
                Data
               </b>
               Analyst
              </a>
             </h2>
             <span class="company">
              <a href="/cmp/Rategenius,-Inc." onmousedown="this.href = appendParamsOnce(this.href, 'from=SERP&amp;campaignid=serp-linkcompanyname&amp;fromjk=d42da7a52cf52852&amp;jcid=e62aa8d5bb020acb')" rel="noopener" target="_blank">
               RateGenius, Inc.
              </a>
             </span>
             -
             <a class="ratingsLabel" data-tn-element="reviewStars" data-tn-variant="cmplinktst2" href="/cmp/Rategenius,-Inc./reviews" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=cmplinktst2&amp;from=SERP&amp;jt=Marketing+Data+Analyst&amp;fromjk=d42da7a52cf52852&amp;jcid=e62aa8d5bb020acb');" rel="noopener" target="_blank" title="Rategenius reviews">
              <span class="ratings">
               <span class="rating" style="width:43.8px">
                <!-- -->
               </span>
              </span>
              <span class="slNoUnderline">
               4 reviews
              </span>
             </a>
             -
             <span class="location">
              Austin, TX 78758
              <span style="font-size: smaller">
               (North Austin area)
              </span>
             </span>
             <table border="0" cellpadding="0" cellspacing="0">
              <tr>
               <td class="snip">
                <div class="">
                 <span class="summary">
                  Acquire
                  <b>
                   data
                  </b>
                  from primary or secondary
                  <b>
                   data
                  </b>
                  sources and maintain databases/
                  <b>
                   data
                  </b>
                  systems. Proven working experience as a
                  <b>
                   data
                  </b>
                  analyst or business
                  <b>
                   data
                  </b>
                  analyst....
                 </span>
                </div>
                <div class="iaP">
                 <span class="iaLabel">
                  Easily apply
                 </span>
                </div>
                <div class="result-link-bar-container">
                 <div class="result-link-bar">
                  <span class="date">
                   Just posted
                  </span>
                  <span class="tt_set" id="tt_set_3">
                   -
                   <a class="sl resultLink save-job-link " href="#" id="sj_d42da7a52cf52852" onclick="changeJobState('d42da7a52cf52852', 'save', 'linkbar', false, ''); return false;" title="Save this job to my.indeed">
                    save job
                   </a>
                   -
                   <a class="sl resultLink more-link " href="#" id="tog_3" onclick="toggleMoreLinks('d42da7a52cf52852'); return false;">
                    more...
                   </a>
                  </span>
                  <div class="edit_note_content" id="editsaved2_d42da7a52cf52852" style="display:none;">
                  </div>
                  <script>
                   if (!window['result_d42da7a52cf52852']) {window['result_d42da7a52cf52852'] = {};}window['result_d42da7a52cf52852']['showSource'] = false; window['result_d42da7a52cf52852']['source'] = "RateGenius, Inc."; window['result_d42da7a52cf52852']['loggedIn'] = false; window['result_d42da7a52cf52852']['showMyJobsLinks'] = false;window['result_d42da7a52cf52852']['undoAction'] = "unsave";window['result_d42da7a52cf52852']['relativeJobAge'] = "Just posted";window['result_d42da7a52cf52852']['jobKey'] = "d42da7a52cf52852"; window['result_d42da7a52cf52852']['myIndeedAvailable'] = true; window['result_d42da7a52cf52852']['showMoreActionsLink'] = window['result_d42da7a52cf52852']['showMoreActionsLink'] || true; window['result_d42da7a52cf52852']['resultNumber'] = 3; window['result_d42da7a52cf52852']['jobStateChangedToSaved'] = false; window['result_d42da7a52cf52852']['searchState'] = "q=data&amp;l=Austin%2C+TX&amp;sort=date"; window['result_d42da7a52cf52852']['basicPermaLink'] = "https://www.indeed.com"; window['result_d42da7a52cf52852']['saveJobFailed'] = false; window['result_d42da7a52cf52852']['removeJobFailed'] = false; window['result_d42da7a52cf52852']['requestPending'] = false; window['result_d42da7a52cf52852']['notesEnabled'] = true; window['result_d42da7a52cf52852']['currentPage'] = "serp"; window['result_d42da7a52cf52852']['sponsored'] = false;window['result_d42da7a52cf52852']['reportJobButtonEnabled'] = false; window['result_d42da7a52cf52852']['showMyJobsHired'] = false; window['result_d42da7a52cf52852']['showSaveForSponsored'] = false; window['result_d42da7a52cf52852']['showJobAge'] = true;
                  </script>
                 </div>
                </div>
                <div class="tab-container">
                 <div class="more-links-container result-tab" id="tt_display_3" style="display:none;">
                  <a class="close-link closeLink" href="#" onclick="toggleMoreLinks('d42da7a52cf52852'); return false;" title="Close">
                  </a>
                  <div class="more_actions" id="more_3">
                   <ul>
                    <li>
                     <span class="mat">
                      View all
                      <a href="/jobs?q=Rategenius,+Inc&amp;l=Austin,+TX&amp;nc=jasx" rel="nofollow">
                       RateGenius, Inc. jobs in Austin, TX
                      </a>
                      -
                      <a href="/l-Austin,-TX-jobs.html">
                       Austin jobs
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Salary Search:
                      <a href="/salaries/Data-Analyst-Salaries,-Austin-TX" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=serp-more&amp;fromjk=d42da7a52cf52852&amp;from=serp-more');">
                       Data Analyst salaries in Austin, TX
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Learn more about working at
                      <a href="/cmp/Rategenius,-Inc." onmousedown="this.href = appendParamsOnce(this.href, '?fromjk=d42da7a52cf52852&amp;from=serp-more&amp;campaignid=serp-more&amp;jcid=e62aa8d5bb020acb');">
                       Rategenius, Inc.
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Related forums:
                      <a href="/forum/loc/Austin-Texas.html">
                       Austin, Texas
                      </a>
                     </span>
                    </li>
                   </ul>
                  </div>
                 </div>
                 <div class="dya-container result-tab">
                 </div>
                 <div class="tellafriend-container result-tab email_job_content">
                 </div>
                 <div class="sign-in-container result-tab">
                 </div>
                 <div class="notes-container result-tab">
                 </div>
                </div>
               </td>
              </tr>
             </table>
            </div>
            <div class=" row result" data-jk="0cd229ee6b711c05" data-tn-component="organicJob" data-tu="" id="p_0cd229ee6b711c05">
             <h2 class="jobtitle" id="jl_0cd229ee6b711c05">
              <a class="turnstileLink" data-tn-element="jobTitle" href="/rc/clk?jk=0cd229ee6b711c05&amp;fccid=8060659b1cece07d&amp;vjs=3" onclick="setRefineByCookie([]); return rclk(this,jobmap[4],true,1);" onmousedown="return rclk(this,jobmap[4],1);" rel="noopener nofollow" target="_blank" title="Management Analyst III">
               Management Analyst III
              </a>
             </h2>
             <span class="company">
              <a href="/cmp/Department-of-State-Health-Services" onmousedown="this.href = appendParamsOnce(this.href, 'from=SERP&amp;campaignid=serp-linkcompanyname&amp;fromjk=0cd229ee6b711c05&amp;jcid=0ab4fcb6780213bd')" rel="noopener" target="_blank">
               Dept of State Health Services
              </a>
             </span>
             -
             <a class="ratingsLabel" data-tn-element="reviewStars" data-tn-variant="cmplinktst2" href="/cmp/Department-of-State-Health-Services/reviews" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=cmplinktst2&amp;from=SERP&amp;jt=Management+Analyst+III&amp;fromjk=0cd229ee6b711c05&amp;jcid=0ab4fcb6780213bd');" rel="noopener" target="_blank">
              <span class="ratings">
               <span class="rating" style="width:43.2px">
                <!-- -->
               </span>
              </span>
              <span class="slNoUnderline">
               55 reviews
              </span>
             </a>
             -
             <span class="location">
              Austin, TX
             </span>
             <table border="0" cellpadding="0" cellspacing="0">
              <tr>
               <td class="snip">
                <div>
                 <span class="no-wrap">
                  $4,301 - $4,687 a month
                 </span>
                </div>
                <div class="">
                 <span class="summary">
                  Maintaining project management data; Responsibilities include developing project plans, conducting workgroup meetings, maintaining project management
                  <b>
                   data
                  </b>
                  , and...
                 </span>
                </div>
                <div class="result-link-bar-container">
                 <div class="result-link-bar">
                  <span class="result-link-source">
                   Texas Health and Human Services Commission
                  </span>
                  -
                  <span class="date">
                   Just posted
                  </span>
                  <span class="tt_set" id="tt_set_4">
                   -
                   <a class="sl resultLink save-job-link " href="#" id="sj_0cd229ee6b711c05" onclick="changeJobState('0cd229ee6b711c05', 'save', 'linkbar', false, ''); return false;" title="Save this job to my.indeed">
                    save job
                   </a>
                   -
                   <a class="sl resultLink more-link " href="#" id="tog_4" onclick="toggleMoreLinks('0cd229ee6b711c05'); return false;">
                    more...
                   </a>
                  </span>
                  <div class="edit_note_content" id="editsaved2_0cd229ee6b711c05" style="display:none;">
                  </div>
                  <script>
                   if (!window['result_0cd229ee6b711c05']) {window['result_0cd229ee6b711c05'] = {};}window['result_0cd229ee6b711c05']['showSource'] = true; window['result_0cd229ee6b711c05']['source'] = "Texas Health and Human Services Commission"; window['result_0cd229ee6b711c05']['loggedIn'] = false; window['result_0cd229ee6b711c05']['showMyJobsLinks'] = false;window['result_0cd229ee6b711c05']['undoAction'] = "unsave";window['result_0cd229ee6b711c05']['relativeJobAge'] = "Just posted";window['result_0cd229ee6b711c05']['jobKey'] = "0cd229ee6b711c05"; window['result_0cd229ee6b711c05']['myIndeedAvailable'] = true; window['result_0cd229ee6b711c05']['showMoreActionsLink'] = window['result_0cd229ee6b711c05']['showMoreActionsLink'] || true; window['result_0cd229ee6b711c05']['resultNumber'] = 4; window['result_0cd229ee6b711c05']['jobStateChangedToSaved'] = false; window['result_0cd229ee6b711c05']['searchState'] = "q=data&amp;l=Austin%2C+TX&amp;sort=date"; window['result_0cd229ee6b711c05']['basicPermaLink'] = "https://www.indeed.com"; window['result_0cd229ee6b711c05']['saveJobFailed'] = false; window['result_0cd229ee6b711c05']['removeJobFailed'] = false; window['result_0cd229ee6b711c05']['requestPending'] = false; window['result_0cd229ee6b711c05']['notesEnabled'] = true; window['result_0cd229ee6b711c05']['currentPage'] = "serp"; window['result_0cd229ee6b711c05']['sponsored'] = false;window['result_0cd229ee6b711c05']['reportJobButtonEnabled'] = false; window['result_0cd229ee6b711c05']['showMyJobsHired'] = false; window['result_0cd229ee6b711c05']['showSaveForSponsored'] = false; window['result_0cd229ee6b711c05']['showJobAge'] = true;
                  </script>
                 </div>
                </div>
                <div class="tab-container">
                 <div class="more-links-container result-tab" id="tt_display_4" style="display:none;">
                  <a class="close-link closeLink" href="#" onclick="toggleMoreLinks('0cd229ee6b711c05'); return false;" title="Close">
                  </a>
                  <div class="more_actions" id="more_4">
                   <ul>
                    <li>
                     <span class="mat">
                      View all
                      <a href="/q-Dept-of-State-Health-Services-l-Austin,-TX-jobs.html" rel="nofollow">
                       Dept of State Health Services jobs in Austin, TX
                      </a>
                      -
                      <a href="/l-Austin,-TX-jobs.html">
                       Austin jobs
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Salary Search:
                      <a href="/salaries/Management-Analyst-Salaries,-Austin-TX" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=serp-more&amp;fromjk=0cd229ee6b711c05&amp;from=serp-more');">
                       Management Analyst salaries in Austin, TX
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Learn more about working at
                      <a href="/cmp/Department-of-State-Health-Services" onmousedown="this.href = appendParamsOnce(this.href, '?fromjk=0cd229ee6b711c05&amp;from=serp-more&amp;campaignid=serp-more&amp;jcid=0ab4fcb6780213bd');">
                       Dept of State Health Services
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      <a href="/cmp/Department-of-State-Health-Services/faq" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=0cd229ee6b711c05&amp;jcid=0ab4fcb6780213bd');">
                       Dept of State Health Services questions about work, benefits, interviews and hiring process:
                      </a>
                      <ul>
                       <li>
                        <a href="/cmp/Department-of-State-Health-Services/faq/how-long-does-it-take-to-get-hired-from-start-to-finish-what-are-the-steps-along-the-way?quid=1anhme8oaas25a84" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=0cd229ee6b711c05&amp;jcid=0ab4fcb6780213bd');">
                         How long does it take to get hired from start to finish? What are the st...
                        </a>
                       </li>
                       <li>
                        <a href="/cmp/Department-of-State-Health-Services/faq/do-they-pay-once-a-month-or-twice?quid=1b5nrojmqaqgmdqj" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=0cd229ee6b711c05&amp;jcid=0ab4fcb6780213bd');">
                         Do they pay once a month or twice?
                        </a>
                       </li>
                      </ul>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Related forums:
                      <a href="/forum/job/Management-Analyst.html">
                       Management Analyst
                      </a>
                      -
                      <a href="/forum/loc/Austin-Texas.html">
                       Austin, Texas
                      </a>
                      -
                      <a href="/forum/cmp/Department-of-State-Health-Services.html">
                       Department of State Health Services
                      </a>
                     </span>
                    </li>
                   </ul>
                  </div>
                 </div>
                 <div class="dya-container result-tab">
                 </div>
                 <div class="tellafriend-container result-tab email_job_content">
                 </div>
                 <div class="sign-in-container result-tab">
                 </div>
                 <div class="notes-container result-tab">
                 </div>
                </div>
               </td>
              </tr>
             </table>
            </div>
            <div class=" row result" data-jk="bbb3d7f08e6073ee" data-tn-component="organicJob" data-tu="" id="p_bbb3d7f08e6073ee">
             <h2 class="jobtitle" id="jl_bbb3d7f08e6073ee">
              <a class="turnstileLink" data-tn-element="jobTitle" href="/rc/clk?jk=bbb3d7f08e6073ee&amp;fccid=12042485d1cec3f9&amp;vjs=3" onclick="setRefineByCookie([]); return rclk(this,jobmap[5],true,0);" onmousedown="return rclk(this,jobmap[5],0);" rel="noopener nofollow" target="_blank" title="Market Research Analyst I">
               Market Research Analyst I
              </a>
             </h2>
             <span class="company">
              <a href="/cmp/Igt" onmousedown="this.href = appendParamsOnce(this.href, 'from=SERP&amp;campaignid=serp-linkcompanyname&amp;fromjk=bbb3d7f08e6073ee&amp;jcid=12042485d1cec3f9')" rel="noopener" target="_blank">
               IGT
              </a>
             </span>
             -
             <a class="ratingsLabel" data-tn-element="reviewStars" data-tn-variant="cmplinktst2" href="/cmp/Igt/reviews" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=cmplinktst2&amp;from=SERP&amp;jt=Market+Research+Analyst+I&amp;fromjk=bbb3d7f08e6073ee&amp;jcid=12042485d1cec3f9');" rel="noopener" target="_blank" title="Igt reviews">
              <span class="ratings">
               <span class="rating" style="width:43.2px">
                <!-- -->
               </span>
              </span>
              <span class="slNoUnderline">
               321 reviews
              </span>
             </a>
             -
             <span class="location">
              Austin, TX 78727
             </span>
             <table border="0" cellpadding="0" cellspacing="0">
              <tr>
               <td class="snip">
                <div class="">
                 <span class="summary">
                  Work with large volumes of disparate
                  <b>
                   data
                  </b>
                  and monitor
                  <b>
                   data
                  </b>
                  for integrity and accuracy. Strong aptitude for
                  <b>
                   data
                  </b>
                  collection and mining....
                 </span>
                </div>
                <div class="result-link-bar-container">
                 <div class="result-link-bar">
                  <span class="date">
                   Just posted
                  </span>
                  <span class="tt_set" id="tt_set_5">
                   -
                   <a class="sl resultLink save-job-link " href="#" id="sj_bbb3d7f08e6073ee" onclick="changeJobState('bbb3d7f08e6073ee', 'save', 'linkbar', false, ''); return false;" title="Save this job to my.indeed">
                    save job
                   </a>
                   -
                   <a class="sl resultLink more-link " href="#" id="tog_5" onclick="toggleMoreLinks('bbb3d7f08e6073ee'); return false;">
                    more...
                   </a>
                  </span>
                  <div class="edit_note_content" id="editsaved2_bbb3d7f08e6073ee" style="display:none;">
                  </div>
                  <script>
                   if (!window['result_bbb3d7f08e6073ee']) {window['result_bbb3d7f08e6073ee'] = {};}window['result_bbb3d7f08e6073ee']['showSource'] = false; window['result_bbb3d7f08e6073ee']['source'] = "IGT"; window['result_bbb3d7f08e6073ee']['loggedIn'] = false; window['result_bbb3d7f08e6073ee']['showMyJobsLinks'] = false;window['result_bbb3d7f08e6073ee']['undoAction'] = "unsave";window['result_bbb3d7f08e6073ee']['relativeJobAge'] = "Just posted";window['result_bbb3d7f08e6073ee']['jobKey'] = "bbb3d7f08e6073ee"; window['result_bbb3d7f08e6073ee']['myIndeedAvailable'] = true; window['result_bbb3d7f08e6073ee']['showMoreActionsLink'] = window['result_bbb3d7f08e6073ee']['showMoreActionsLink'] || true; window['result_bbb3d7f08e6073ee']['resultNumber'] = 5; window['result_bbb3d7f08e6073ee']['jobStateChangedToSaved'] = false; window['result_bbb3d7f08e6073ee']['searchState'] = "q=data&amp;l=Austin%2C+TX&amp;sort=date"; window['result_bbb3d7f08e6073ee']['basicPermaLink'] = "https://www.indeed.com"; window['result_bbb3d7f08e6073ee']['saveJobFailed'] = false; window['result_bbb3d7f08e6073ee']['removeJobFailed'] = false; window['result_bbb3d7f08e6073ee']['requestPending'] = false; window['result_bbb3d7f08e6073ee']['notesEnabled'] = true; window['result_bbb3d7f08e6073ee']['currentPage'] = "serp"; window['result_bbb3d7f08e6073ee']['sponsored'] = false;window['result_bbb3d7f08e6073ee']['reportJobButtonEnabled'] = false; window['result_bbb3d7f08e6073ee']['showMyJobsHired'] = false; window['result_bbb3d7f08e6073ee']['showSaveForSponsored'] = false; window['result_bbb3d7f08e6073ee']['showJobAge'] = true;
                  </script>
                 </div>
                </div>
                <div class="tab-container">
                 <div class="more-links-container result-tab" id="tt_display_5" style="display:none;">
                  <a class="close-link closeLink" href="#" onclick="toggleMoreLinks('bbb3d7f08e6073ee'); return false;" title="Close">
                  </a>
                  <div class="more_actions" id="more_5">
                   <ul>
                    <li>
                     <span class="mat">
                      View all
                      <a href="/q-Igt-l-Austin,-TX-jobs.html" rel="nofollow">
                       IGT jobs in Austin, TX
                      </a>
                      -
                      <a href="/l-Austin,-TX-jobs.html">
                       Austin jobs
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Salary Search:
                      <a href="/salaries/Market-Researcher-Salaries,-Austin-TX" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=serp-more&amp;fromjk=bbb3d7f08e6073ee&amp;from=serp-more');">
                       Market Researcher salaries in Austin, TX
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Learn more about working at
                      <a href="/cmp/Igt" onmousedown="this.href = appendParamsOnce(this.href, '?fromjk=bbb3d7f08e6073ee&amp;from=serp-more&amp;campaignid=serp-more&amp;jcid=12042485d1cec3f9');">
                       Igt
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      <a href="/cmp/Igt/faq" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=bbb3d7f08e6073ee&amp;jcid=12042485d1cec3f9');">
                       Igt questions about work, benefits, interviews and hiring process:
                      </a>
                      <ul>
                       <li>
                        <a href="/cmp/Igt/faq/what-would-you-suggest-igt-management-do-to-prevent-others-from-leaving-for-this-reason?quid=1bc354ss1brdbcn0" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=bbb3d7f08e6073ee&amp;jcid=12042485d1cec3f9');">
                         What would you suggest IGT management do to prevent others from leaving ...
                        </a>
                       </li>
                       <li>
                        <a href="/cmp/Igt/faq/how-are-the-working-hours?quid=1an9jcrio1ah37cj" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=bbb3d7f08e6073ee&amp;jcid=12042485d1cec3f9');">
                         How are the working hours?
                        </a>
                       </li>
                      </ul>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Related forums:
                      <a href="/forum/loc/Austin-Texas.html">
                       Austin, Texas
                      </a>
                      -
                      <a href="/forum/cmp/Igt.html">
                       IGT
                      </a>
                      -
                      <a href="/forum/job/Market-Research-Analyst.html">
                       Market Research Analyst
                      </a>
                     </span>
                    </li>
                   </ul>
                  </div>
                 </div>
                 <div class="dya-container result-tab">
                 </div>
                 <div class="tellafriend-container result-tab email_job_content">
                 </div>
                 <div class="sign-in-container result-tab">
                 </div>
                 <div class="notes-container result-tab">
                 </div>
                </div>
               </td>
              </tr>
             </table>
            </div>
            <div class=" row result" data-jk="5923aa20581d8796" data-tn-component="organicJob" data-tu="" id="p_5923aa20581d8796">
             <h2 class="jobtitle" id="jl_5923aa20581d8796">
              <a class="turnstileLink" data-tn-element="jobTitle" href="/rc/clk?jk=5923aa20581d8796&amp;fccid=a1b3a06b75f24ce1&amp;vjs=3" onclick="setRefineByCookie([]); return rclk(this,jobmap[6],true,0);" onmousedown="return rclk(this,jobmap[6],0);" rel="noopener nofollow" target="_blank" title="Systems Analyst VI">
               Systems Analyst VI
              </a>
             </h2>
             <span class="company">
              DEPARTMENT OF INFORMATION RESOURCES
             </span>
             -
             <span class="location">
              Austin, TX
             </span>
             <table border="0" cellpadding="0" cellspacing="0">
              <tr>
               <td class="snip">
                <div class="">
                 <span class="summary">
                  Uses automated tool to analyze
                  <b>
                   data
                  </b>
                  exports and spreadsheet
                  <b>
                   data
                  </b>
                  addressing usage of capacity, disk storage, tape, etc....
                 </span>
                </div>
                <div class="result-link-bar-container">
                 <div class="result-link-bar">
                  <span class="result-link-source">
                   Texas Department of Public Safety
                  </span>
                  -
                  <span class="date">
                   Just posted
                  </span>
                  <span class="tt_set" id="tt_set_6">
                   -
                   <a class="sl resultLink save-job-link " href="#" id="sj_5923aa20581d8796" onclick="changeJobState('5923aa20581d8796', 'save', 'linkbar', false, ''); return false;" title="Save this job to my.indeed">
                    save job
                   </a>
                   -
                   <a class="sl resultLink more-link " href="#" id="tog_6" onclick="toggleMoreLinks('5923aa20581d8796'); return false;">
                    more...
                   </a>
                  </span>
                  <div class="edit_note_content" id="editsaved2_5923aa20581d8796" style="display:none;">
                  </div>
                  <script>
                   if (!window['result_5923aa20581d8796']) {window['result_5923aa20581d8796'] = {};}window['result_5923aa20581d8796']['showSource'] = true; window['result_5923aa20581d8796']['source'] = "Texas Department of Public Safety"; window['result_5923aa20581d8796']['loggedIn'] = false; window['result_5923aa20581d8796']['showMyJobsLinks'] = false;window['result_5923aa20581d8796']['undoAction'] = "unsave";window['result_5923aa20581d8796']['relativeJobAge'] = "Just posted";window['result_5923aa20581d8796']['jobKey'] = "5923aa20581d8796"; window['result_5923aa20581d8796']['myIndeedAvailable'] = true; window['result_5923aa20581d8796']['showMoreActionsLink'] = window['result_5923aa20581d8796']['showMoreActionsLink'] || true; window['result_5923aa20581d8796']['resultNumber'] = 6; window['result_5923aa20581d8796']['jobStateChangedToSaved'] = false; window['result_5923aa20581d8796']['searchState'] = "q=data&amp;l=Austin%2C+TX&amp;sort=date"; window['result_5923aa20581d8796']['basicPermaLink'] = "https://www.indeed.com"; window['result_5923aa20581d8796']['saveJobFailed'] = false; window['result_5923aa20581d8796']['removeJobFailed'] = false; window['result_5923aa20581d8796']['requestPending'] = false; window['result_5923aa20581d8796']['notesEnabled'] = true; window['result_5923aa20581d8796']['currentPage'] = "serp"; window['result_5923aa20581d8796']['sponsored'] = false;window['result_5923aa20581d8796']['reportJobButtonEnabled'] = false; window['result_5923aa20581d8796']['showMyJobsHired'] = false; window['result_5923aa20581d8796']['showSaveForSponsored'] = false; window['result_5923aa20581d8796']['showJobAge'] = true;
                  </script>
                 </div>
                </div>
                <div class="tab-container">
                 <div class="more-links-container result-tab" id="tt_display_6" style="display:none;">
                  <a class="close-link closeLink" href="#" onclick="toggleMoreLinks('5923aa20581d8796'); return false;" title="Close">
                  </a>
                  <div class="more_actions" id="more_6">
                   <ul>
                    <li>
                     <span class="mat">
                      View all
                      <a href="/q-Department-of-Information-Resources-l-Austin,-TX-jobs.html" rel="nofollow">
                       DEPARTMENT OF INFORMATION RESOURCES jobs in Austin, TX
                      </a>
                      -
                      <a href="/l-Austin,-TX-jobs.html">
                       Austin jobs
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Salary Search:
                      <a href="/salaries/Systems-Analyst-Salaries,-Austin-TX" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=serp-more&amp;fromjk=5923aa20581d8796&amp;from=serp-more');">
                       Systems Analyst salaries in Austin, TX
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Learn more about working at
                      <a href="/cmp/Department-of-Information-Resources" onmousedown="this.href = appendParamsOnce(this.href, '?fromjk=5923aa20581d8796&amp;from=serp-more&amp;campaignid=serp-more&amp;jcid=075b49fe6eeb6013');">
                       Department of Information Resources
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Related forums:
                      <a href="/forum/loc/Austin-Texas.html">
                       Austin, Texas
                      </a>
                      -
                      <a href="/forum/cmp/Department-of-Information-Resources.html">
                       Department of Information Resources
                      </a>
                      -
                      <a href="/forum/job/Systems-Analyst.html">
                       Systems Analyst
                      </a>
                     </span>
                    </li>
                   </ul>
                  </div>
                 </div>
                 <div class="dya-container result-tab">
                 </div>
                 <div class="tellafriend-container result-tab email_job_content">
                 </div>
                 <div class="sign-in-container result-tab">
                 </div>
                 <div class="notes-container result-tab">
                 </div>
                </div>
               </td>
              </tr>
             </table>
            </div>
            <div class=" row result" data-jk="7cc26370ceeb26a7" data-tn-component="organicJob" data-tu="" id="p_7cc26370ceeb26a7">
             <h2 class="jobtitle" id="jl_7cc26370ceeb26a7">
              <a class="turnstileLink" data-tn-element="jobTitle" href="/rc/clk?jk=7cc26370ceeb26a7&amp;fccid=113517153f849886&amp;vjs=3" onclick="setRefineByCookie([]); return rclk(this,jobmap[7],true,1);" onmousedown="return rclk(this,jobmap[7],1);" rel="noopener nofollow" target="_blank" title="Systems Analyst IV">
               Systems Analyst IV
              </a>
             </h2>
             <span class="company">
              <a href="/cmp/Texas-Health-and-Human-Services-Commission" onmousedown="this.href = appendParamsOnce(this.href, 'from=SERP&amp;campaignid=serp-linkcompanyname&amp;fromjk=7cc26370ceeb26a7&amp;jcid=ec038939be4318c8')" rel="noopener" target="_blank">
               Health &amp; Human Services Comm
              </a>
             </span>
             -
             <a class="ratingsLabel" data-tn-element="reviewStars" data-tn-variant="cmplinktst2" href="/cmp/Texas-Health-and-Human-Services-Commission/reviews" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=cmplinktst2&amp;from=SERP&amp;jt=Systems+Analyst+IV&amp;fromjk=7cc26370ceeb26a7&amp;jcid=ec038939be4318c8');" rel="noopener" target="_blank">
              <span class="ratings">
               <span class="rating" style="width:42.6px">
                <!-- -->
               </span>
              </span>
              <span class="slNoUnderline">
               616 reviews
              </span>
             </a>
             -
             <span class="location">
              Austin, TX
             </span>
             <table border="0" cellpadding="0" cellspacing="0">
              <tr>
               <td class="snip">
                <div>
                 <span class="no-wrap">
                  $4,301 - $7,040 a month
                 </span>
                </div>
                <div class="">
                 <span class="summary">
                  20% • Creates or modifies test
                  <b>
                   data
                  </b>
                  , test plans and/or test case work. 10% • Serves as a liaison with vendors, state
                  <b>
                   data
                  </b>
                  center operations, and state technical...
                 </span>
                </div>
                <div class="result-link-bar-container">
                 <div class="result-link-bar">
                  <span class="result-link-source">
                   Texas Health and Human Services Commission
                  </span>
                  -
                  <span class="date">
                   Just posted
                  </span>
                  <span class="tt_set" id="tt_set_7">
                   -
                   <a class="sl resultLink save-job-link " href="#" id="sj_7cc26370ceeb26a7" onclick="changeJobState('7cc26370ceeb26a7', 'save', 'linkbar', false, ''); return false;" title="Save this job to my.indeed">
                    save job
                   </a>
                   -
                   <a class="sl resultLink more-link " href="#" id="tog_7" onclick="toggleMoreLinks('7cc26370ceeb26a7'); return false;">
                    more...
                   </a>
                  </span>
                  <div class="edit_note_content" id="editsaved2_7cc26370ceeb26a7" style="display:none;">
                  </div>
                  <script>
                   if (!window['result_7cc26370ceeb26a7']) {window['result_7cc26370ceeb26a7'] = {};}window['result_7cc26370ceeb26a7']['showSource'] = true; window['result_7cc26370ceeb26a7']['source'] = "Texas Health and Human Services Commission"; window['result_7cc26370ceeb26a7']['loggedIn'] = false; window['result_7cc26370ceeb26a7']['showMyJobsLinks'] = false;window['result_7cc26370ceeb26a7']['undoAction'] = "unsave";window['result_7cc26370ceeb26a7']['relativeJobAge'] = "Just posted";window['result_7cc26370ceeb26a7']['jobKey'] = "7cc26370ceeb26a7"; window['result_7cc26370ceeb26a7']['myIndeedAvailable'] = true; window['result_7cc26370ceeb26a7']['showMoreActionsLink'] = window['result_7cc26370ceeb26a7']['showMoreActionsLink'] || true; window['result_7cc26370ceeb26a7']['resultNumber'] = 7; window['result_7cc26370ceeb26a7']['jobStateChangedToSaved'] = false; window['result_7cc26370ceeb26a7']['searchState'] = "q=data&amp;l=Austin%2C+TX&amp;sort=date"; window['result_7cc26370ceeb26a7']['basicPermaLink'] = "https://www.indeed.com"; window['result_7cc26370ceeb26a7']['saveJobFailed'] = false; window['result_7cc26370ceeb26a7']['removeJobFailed'] = false; window['result_7cc26370ceeb26a7']['requestPending'] = false; window['result_7cc26370ceeb26a7']['notesEnabled'] = true; window['result_7cc26370ceeb26a7']['currentPage'] = "serp"; window['result_7cc26370ceeb26a7']['sponsored'] = false;window['result_7cc26370ceeb26a7']['reportJobButtonEnabled'] = false; window['result_7cc26370ceeb26a7']['showMyJobsHired'] = false; window['result_7cc26370ceeb26a7']['showSaveForSponsored'] = false; window['result_7cc26370ceeb26a7']['showJobAge'] = true;
                  </script>
                 </div>
                </div>
                <div class="tab-container">
                 <div class="more-links-container result-tab" id="tt_display_7" style="display:none;">
                  <a class="close-link closeLink" href="#" onclick="toggleMoreLinks('7cc26370ceeb26a7'); return false;" title="Close">
                  </a>
                  <div class="more_actions" id="more_7">
                   <ul>
                    <li>
                     <span class="mat">
                      View all
                      <a href="/q-Health-&amp;-Human-Services-Comm-l-Austin,-TX-jobs.html" rel="nofollow">
                       Health &amp; Human Services Comm jobs in Austin, TX
                      </a>
                      -
                      <a href="/l-Austin,-TX-jobs.html">
                       Austin jobs
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Salary Search:
                      <a href="/salaries/Senior-Systems-Analyst-Salaries,-Austin-TX" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=serp-more&amp;fromjk=7cc26370ceeb26a7&amp;from=serp-more');">
                       Senior Systems Analyst salaries in Austin, TX
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Learn more about working at
                      <a href="/cmp/Texas-Health-and-Human-Services-Commission" onmousedown="this.href = appendParamsOnce(this.href, '?fromjk=7cc26370ceeb26a7&amp;from=serp-more&amp;campaignid=serp-more&amp;jcid=ec038939be4318c8');">
                       Health &amp; Human Services Comm
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      <a href="/cmp/Texas-Health-and-Human-Services-Commission/faq" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=7cc26370ceeb26a7&amp;jcid=ec038939be4318c8');">
                       Health &amp; Human Services Comm questions about work, benefits, interviews and hiring process:
                      </a>
                      <ul>
                       <li>
                        <a href="/cmp/Texas-Health-and-Human-Services-Commission/faq/what-is-the-interview-process-like?quid=1amsi90roak4ha2d" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=7cc26370ceeb26a7&amp;jcid=ec038939be4318c8');">
                         What is the interview process like?
                        </a>
                       </li>
                       <li>
                        <a href="/cmp/Texas-Health-and-Human-Services-Commission/faq/what-is-the-most-stressful-part-about-working-at-health-and-human-services-commission?quid=1b4c96ko6akavc8u" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=7cc26370ceeb26a7&amp;jcid=ec038939be4318c8');">
                         What is the most stressful part about working at Health and Human Servic...
                        </a>
                       </li>
                      </ul>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Related forums:
                      <a href="/forum/loc/Austin-Texas.html">
                       Austin, Texas
                      </a>
                      -
                      <a href="/forum/cmp/Texas-Health-and-Human-Services-Commission.html">
                       Texas Health and Human Services Commission
                      </a>
                      -
                      <a href="/forum/job/Systems-Analyst.html">
                       Systems Analyst
                      </a>
                     </span>
                    </li>
                   </ul>
                  </div>
                 </div>
                 <div class="dya-container result-tab">
                 </div>
                 <div class="tellafriend-container result-tab email_job_content">
                 </div>
                 <div class="sign-in-container result-tab">
                 </div>
                 <div class="notes-container result-tab">
                 </div>
                </div>
               </td>
              </tr>
             </table>
            </div>
            <div class=" row result" data-jk="0d9a0d1520d456d6" data-tn-component="organicJob" data-tu="" id="p_0d9a0d1520d456d6">
             <h2 class="jobtitle" id="jl_0d9a0d1520d456d6">
              <a class="turnstileLink" data-tn-element="jobTitle" href="/rc/clk?jk=0d9a0d1520d456d6&amp;fccid=32e2902cbfbc8198&amp;vjs=3" onclick="setRefineByCookie([]); return rclk(this,jobmap[8],true,0);" onmousedown="return rclk(this,jobmap[8],0);" rel="noopener nofollow" target="_blank" title="Level 4 Server / Data Center Support Windows / Linux">
               Level 4 Server /
               <b>
                Data
               </b>
               Center Support Windows / Linux
              </a>
             </h2>
             <span class="company">
              eXcell
             </span>
             -
             <span class="location">
              Austin, TX 78746
             </span>
             <table border="0" cellpadding="0" cellspacing="0">
              <tr>
               <td class="snip">
                <div class="">
                 <span class="summary">
                  <b>
                   Data
                  </b>
                  Center Support:. Training and backfilling for partner IT groups, including
                  <b>
                   Data
                  </b>
                  Center Housing, IT Service Centers and Audio Visual departments....
                 </span>
                </div>
                <div class="iaP">
                 <span class="iaLabel">
                  Easily apply
                 </span>
                </div>
                <div class="result-link-bar-container">
                 <div class="result-link-bar">
                  <span class="date">
                   Just posted
                  </span>
                  <span class="tt_set" id="tt_set_8">
                   -
                   <a class="sl resultLink save-job-link " href="#" id="sj_0d9a0d1520d456d6" onclick="changeJobState('0d9a0d1520d456d6', 'save', 'linkbar', false, ''); return false;" title="Save this job to my.indeed">
                    save job
                   </a>
                   -
                   <a class="sl resultLink more-link " href="#" id="tog_8" onclick="toggleMoreLinks('0d9a0d1520d456d6'); return false;">
                    more...
                   </a>
                  </span>
                  <div class="edit_note_content" id="editsaved2_0d9a0d1520d456d6" style="display:none;">
                  </div>
                  <script>
                   if (!window['result_0d9a0d1520d456d6']) {window['result_0d9a0d1520d456d6'] = {};}window['result_0d9a0d1520d456d6']['showSource'] = false; window['result_0d9a0d1520d456d6']['source'] = "Excell"; window['result_0d9a0d1520d456d6']['loggedIn'] = false; window['result_0d9a0d1520d456d6']['showMyJobsLinks'] = false;window['result_0d9a0d1520d456d6']['undoAction'] = "unsave";window['result_0d9a0d1520d456d6']['relativeJobAge'] = "Just posted";window['result_0d9a0d1520d456d6']['jobKey'] = "0d9a0d1520d456d6"; window['result_0d9a0d1520d456d6']['myIndeedAvailable'] = true; window['result_0d9a0d1520d456d6']['showMoreActionsLink'] = window['result_0d9a0d1520d456d6']['showMoreActionsLink'] || true; window['result_0d9a0d1520d456d6']['resultNumber'] = 8; window['result_0d9a0d1520d456d6']['jobStateChangedToSaved'] = false; window['result_0d9a0d1520d456d6']['searchState'] = "q=data&amp;l=Austin%2C+TX&amp;sort=date"; window['result_0d9a0d1520d456d6']['basicPermaLink'] = "https://www.indeed.com"; window['result_0d9a0d1520d456d6']['saveJobFailed'] = false; window['result_0d9a0d1520d456d6']['removeJobFailed'] = false; window['result_0d9a0d1520d456d6']['requestPending'] = false; window['result_0d9a0d1520d456d6']['notesEnabled'] = true; window['result_0d9a0d1520d456d6']['currentPage'] = "serp"; window['result_0d9a0d1520d456d6']['sponsored'] = false;window['result_0d9a0d1520d456d6']['reportJobButtonEnabled'] = false; window['result_0d9a0d1520d456d6']['showMyJobsHired'] = false; window['result_0d9a0d1520d456d6']['showSaveForSponsored'] = false; window['result_0d9a0d1520d456d6']['showJobAge'] = true;
                  </script>
                 </div>
                </div>
                <div class="tab-container">
                 <div class="more-links-container result-tab" id="tt_display_8" style="display:none;">
                  <a class="close-link closeLink" href="#" onclick="toggleMoreLinks('0d9a0d1520d456d6'); return false;" title="Close">
                  </a>
                  <div class="more_actions" id="more_8">
                   <ul>
                    <li>
                     <span class="mat">
                      View all
                      <a href="/q-Excell-l-Austin,-TX-jobs.html" rel="nofollow">
                       eXcell jobs in Austin, TX
                      </a>
                      -
                      <a href="/l-Austin,-TX-jobs.html">
                       Austin jobs
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Salary Search:
                      <a href="/salaries/Data-Center-Technician-Salaries,-Austin-TX" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=serp-more&amp;fromjk=0d9a0d1520d456d6&amp;from=serp-more');">
                       Data Center Technician salaries in Austin, TX
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Learn more about working at
                      <a href="/cmp/Excell-2" onmousedown="this.href = appendParamsOnce(this.href, '?fromjk=0d9a0d1520d456d6&amp;from=serp-more&amp;campaignid=serp-more&amp;jcid=00347ecfd5bc04ed');">
                       Excell
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      <a href="/cmp/Excell-2/faq" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=0d9a0d1520d456d6&amp;jcid=00347ecfd5bc04ed');">
                       Excell questions about work, benefits, interviews and hiring process:
                      </a>
                      <ul>
                       <li>
                        <a href="/cmp/Excell-2/faq/on-average-how-many-hours-do-you-work-a-day?quid=1c4kuvu4952tocv4" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=0d9a0d1520d456d6&amp;jcid=00347ecfd5bc04ed');">
                         On average, how many hours do you work a day?
                        </a>
                       </li>
                       <li>
                        <a href="/cmp/Excell-2/faq/how-did-you-feel-about-telling-people-you-worked-at-excell?quid=1c4kuvu485ndkbtk" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=0d9a0d1520d456d6&amp;jcid=00347ecfd5bc04ed');">
                         How did you feel about telling people you worked at eXcell?
                        </a>
                       </li>
                      </ul>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Related forums:
                      <a href="/forum/loc/Austin-Texas.html">
                       Austin, Texas
                      </a>
                      -
                      <a href="/forum/cmp/Excell-2.html">
                       eXcell
                      </a>
                     </span>
                    </li>
                   </ul>
                  </div>
                 </div>
                 <div class="dya-container result-tab">
                 </div>
                 <div class="tellafriend-container result-tab email_job_content">
                 </div>
                 <div class="sign-in-container result-tab">
                 </div>
                 <div class="notes-container result-tab">
                 </div>
                </div>
               </td>
              </tr>
             </table>
            </div>
            <div class="lastRow row result" data-jk="10b5a33e6be90586" data-tn-component="organicJob" data-tu="" id="p_10b5a33e6be90586">
             <h2 class="jobtitle" id="jl_10b5a33e6be90586">
              <a class="turnstileLink" data-tn-element="jobTitle" href="/rc/clk?jk=10b5a33e6be90586&amp;fccid=651f65cb69970e95&amp;vjs=3" onclick="setRefineByCookie([]); return rclk(this,jobmap[9],true,1);" onmousedown="return rclk(this,jobmap[9],1);" rel="noopener nofollow" target="_blank" title="Court Clerk Assistant">
               Court Clerk Assistant
              </a>
             </h2>
             <span class="company">
              <a href="/cmp/City-of-Austin" onmousedown="this.href = appendParamsOnce(this.href, 'from=SERP&amp;campaignid=serp-linkcompanyname&amp;fromjk=10b5a33e6be90586&amp;jcid=651f65cb69970e95')" rel="noopener" target="_blank">
               City of Austin
              </a>
             </span>
             -
             <a class="ratingsLabel" data-tn-element="reviewStars" data-tn-variant="cmplinktst2" href="/cmp/City-of-Austin/reviews" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=cmplinktst2&amp;from=SERP&amp;jt=Court+Clerk+Assistant&amp;fromjk=10b5a33e6be90586&amp;jcid=651f65cb69970e95');" rel="noopener" target="_blank" title="City of Austin reviews">
              <span class="ratings">
               <span class="rating" style="width:52.2px">
                <!-- -->
               </span>
              </span>
              <span class="slNoUnderline">
               227 reviews
              </span>
             </a>
             -
             <span class="location">
              Austin, TX 78702
              <span style="font-size: smaller">
               (Rosewood area)
              </span>
             </span>
             <table border="0" cellpadding="0" cellspacing="0">
              <tr>
               <td class="snip">
                <div>
                 <span class="no-wrap">
                  $16.32 - $20.31 an hour
                 </span>
                </div>
                <div class="">
                 <span class="summary">
                  Skill in
                  <b>
                   data
                  </b>
                  analysis and problem solving. Graduation from an accredited high school or equivalent plus four (4) years of related experience....
                 </span>
                </div>
                <div class="result-link-bar-container">
                 <div class="result-link-bar">
                  <span class="date">
                   Just posted
                  </span>
                  <span class="tt_set" id="tt_set_9">
                   -
                   <a class="sl resultLink save-job-link " href="#" id="sj_10b5a33e6be90586" onclick="changeJobState('10b5a33e6be90586', 'save', 'linkbar', false, ''); return false;" title="Save this job to my.indeed">
                    save job
                   </a>
                   -
                   <a class="sl resultLink more-link " href="#" id="tog_9" onclick="toggleMoreLinks('10b5a33e6be90586'); return false;">
                    more...
                   </a>
                  </span>
                  <div class="edit_note_content" id="editsaved2_10b5a33e6be90586" style="display:none;">
                  </div>
                  <script>
                   if (!window['result_10b5a33e6be90586']) {window['result_10b5a33e6be90586'] = {};}window['result_10b5a33e6be90586']['showSource'] = false; window['result_10b5a33e6be90586']['source'] = "City of Austin"; window['result_10b5a33e6be90586']['loggedIn'] = false; window['result_10b5a33e6be90586']['showMyJobsLinks'] = false;window['result_10b5a33e6be90586']['undoAction'] = "unsave";window['result_10b5a33e6be90586']['relativeJobAge'] = "Just posted";window['result_10b5a33e6be90586']['jobKey'] = "10b5a33e6be90586"; window['result_10b5a33e6be90586']['myIndeedAvailable'] = true; window['result_10b5a33e6be90586']['showMoreActionsLink'] = window['result_10b5a33e6be90586']['showMoreActionsLink'] || true; window['result_10b5a33e6be90586']['resultNumber'] = 9; window['result_10b5a33e6be90586']['jobStateChangedToSaved'] = false; window['result_10b5a33e6be90586']['searchState'] = "q=data&amp;l=Austin%2C+TX&amp;sort=date"; window['result_10b5a33e6be90586']['basicPermaLink'] = "https://www.indeed.com"; window['result_10b5a33e6be90586']['saveJobFailed'] = false; window['result_10b5a33e6be90586']['removeJobFailed'] = false; window['result_10b5a33e6be90586']['requestPending'] = false; window['result_10b5a33e6be90586']['notesEnabled'] = true; window['result_10b5a33e6be90586']['currentPage'] = "serp"; window['result_10b5a33e6be90586']['sponsored'] = false;window['result_10b5a33e6be90586']['reportJobButtonEnabled'] = false; window['result_10b5a33e6be90586']['showMyJobsHired'] = false; window['result_10b5a33e6be90586']['showSaveForSponsored'] = false; window['result_10b5a33e6be90586']['showJobAge'] = true;
                  </script>
                 </div>
                </div>
                <div class="tab-container">
                 <div class="more-links-container result-tab" id="tt_display_9" style="display:none;">
                  <a class="close-link closeLink" href="#" onclick="toggleMoreLinks('10b5a33e6be90586'); return false;" title="Close">
                  </a>
                  <div class="more_actions" id="more_9">
                   <ul>
                    <li>
                     <span class="mat">
                      View all
                      <a href="/q-City-of-Austin-l-Austin,-TX-jobs.html" rel="nofollow">
                       City of Austin jobs in Austin, TX
                      </a>
                      -
                      <a href="/l-Austin,-TX-jobs.html">
                       Austin jobs
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Salary Search:
                      <a href="/salaries/Court-Clerk-Salaries,-Austin-TX" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=serp-more&amp;fromjk=10b5a33e6be90586&amp;from=serp-more');">
                       Court Clerk salaries in Austin, TX
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Learn more about working at
                      <a href="/cmp/City-of-Austin" onmousedown="this.href = appendParamsOnce(this.href, '?fromjk=10b5a33e6be90586&amp;from=serp-more&amp;campaignid=serp-more&amp;jcid=651f65cb69970e95');">
                       City of Austin
                      </a>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      <a href="/cmp/City-of-Austin/faq" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=10b5a33e6be90586&amp;jcid=651f65cb69970e95');">
                       City of Austin questions about work, benefits, interviews and hiring process:
                      </a>
                      <ul>
                       <li>
                        <a href="/cmp/City-of-Austin/faq/how-did-you-feel-about-telling-people-you-worked-at-city-of-austin?quid=1b5qvb3dnak8pasi" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=10b5a33e6be90586&amp;jcid=651f65cb69970e95');">
                         How did you feel about telling people you worked at City of Austin?
                        </a>
                       </li>
                       <li>
                        <a href="/cmp/City-of-Austin/faq/how-long-could-be-the-process-to-get-hired?quid=1b61vhi38ak8pa6d" onmousedown="this.href = appendParamsOnce(this.href, '?from=serp-more&amp;campaignid=serp-more&amp;fromjk=10b5a33e6be90586&amp;jcid=651f65cb69970e95');">
                         How long could be the process to get hired?
                        </a>
                       </li>
                      </ul>
                     </span>
                    </li>
                    <li>
                     <span class="mat">
                      Related forums:
                      <a href="/forum/cmp/City-of-Austin.html">
                       City of Austin
                      </a>
                      -
                      <a href="/forum/loc/Austin-Texas.html">
                       Austin, Texas
                      </a>
                     </span>
                    </li>
                   </ul>
                  </div>
                 </div>
                 <div class="dya-container result-tab">
                 </div>
                 <div class="tellafriend-container result-tab email_job_content">
                 </div>
                 <div class="sign-in-container result-tab">
                 </div>
                 <div class="notes-container result-tab">
                 </div>
                </div>
               </td>
              </tr>
             </table>
            </div>
            <div>
            </div>
            <div class="bzugp0PeDkk KQox8b8lBh">
             <div class="row result" data-advn="6111343572945211" data-jk="d04f631427a29323" data-tu="https://jsv3.recruitics.com/partner/a51b8de1-f7bf-11e7-9edd-d951492604d9.gif?client=876&amp;job=US576JAC_24231890&amp;rx_a=25&amp;rx_c=kcn&amp;rx_campaign=indeed25&amp;rx_group=101400&amp;rx_source=Indeed&amp;indeed=sp" id="pj_d04f631427a29323">
              <a class="jobtitle turnstileLink" data-tn-element="jobTitle" href="/pagead/clk?mo=r&amp;ad=-6NYlbfkN0D6qFSVCaa8tXn-rJ3OcXif2lPyFmwsE2iZBGE4YLg1g5TVF4TIVaZeunR7yCIbgMgQlWUA7XB0tO6fpDEzIy5kQFFDNq4i30w9QgCb6B9JvFCuij39MCPd2__vN-LBwQpGFNDKJjcL399DQDL-_lBbWKZMdI8g87diMg4Zi5ljK4FvFBZJGSEU-0_M0ESzbNJ8JtpJVSO5vN2cs8FHDmfeCMnVzZRl4FWwbXPqDrYK8Gz8Z5Y26wXC1vHy_aze9aqsrriNv3gcyYfdoOlwOD6BBx-YrhM8-3JDktfsJyGorNmUwHdCP5y-aXkSzWJ_sFUWTgco1JOAcAMZfF8JJmv7Xgu3V73i-Y-sP4jgDlmKlvNeVwiyAiqgP1IFwJHu00cNH-9i6mpnUcld1c6zGBoD0nAF5OHI_bhIJjwxjNfr3ffQ0voSeRUbH1cKA5L3nLnxpE8pvthg6H3opfSb9k_OoLflcsCK1M9_O0dMNOlMzP5EOYBaidE4XU5aAHQEBCE1CW3gsQDoeDAWNQYOHXJwjnzWSuahokub9a7TpcgvO3E_8RdHSSYkyYQYtI1bYUrkOqomZvLfx6F_ZJoF0JxO-BXPuG72WHcpkIH9RL9Tk2KysD7nyLJMAH7U2c-XLBcMIIESsuehr8jwIc17FJDkBSBinDLd6woNSksj4uycMezu94Bf0dRtddNSBGP_WQwB6ej92U8CtnxaJ_5UxmlOL1yKo8I5GYLibE9erzB8y7n1xcfrr4m7UvLMoJEI9bI=&amp;vjs=3&amp;p=1&amp;sk=&amp;fvj=0" id="sja1" onclick="setRefineByCookie([]); sjoc('sja1',0); convCtr('SJ')" onmousedown="sjomd('sja1'); clk('sja1');" rel="noopener nofollow" target="_blank" title="Data Entry $15.00 hourly">
               <b>
                Data
               </b>
               Entry $15.00 hourly
              </a>
              <br/>
              <div class="sjcl">
               <span class="company">
                <a class="turnstileLink" data-tn-element="companyName" href="/cmp/Kelly-Services" onmousedown="this.href = appendParamsOnce(this.href, 'from=SERP&amp;campaignid=serp-linkcompanyname&amp;fromjk=d04f631427a29323&amp;jcid=26a0583287ba1940')" rel="noopener" target="_blank">
                 Kelly Services
                </a>
               </span>
               -
               <a class="ratingsLabel" data-tn-element="reviewStars" data-tn-variant="cmplinktst2" href="/cmp/Kelly-Services/reviews" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=cmplinktst2&amp;from=SERP&amp;jt=Data+Entry+%2415.00+hourly&amp;fromjk=d04f631427a29323&amp;jcid=26a0583287ba1940');" rel="noopener" target="_blank" title="Kelly Services reviews">
                <span class="ratings">
                 <span class="rating" style="width:44.4px">
                  <!-- -->
                 </span>
                </span>
                <span class="slNoUnderline">
                 10,404 reviews
                </span>
               </a>
               -
               <span class="location">
                Austin, TX 78758
               </span>
              </div>
              <div class="">
               <table border="0" cellpadding="0" cellspacing="0">
                <tr>
                 <td class="snip">
                  <span class="summary">
                   <b>
                    Data
                   </b>
                   Entry 1-3 years experience. Entering all invoices into computer and attaching to account....
                  </span>
                 </td>
                </tr>
               </table>
              </div>
              <div class="sjCapt">
               <div class="result-link-bar-container">
                <div class="result-link-bar">
                 <span class=" sponsoredGray ">
                  Sponsored
                 </span>
                 -
                 <span class="tt_set" id="tt_set_10">
                  <a class="sl resultLink save-job-link " href="#" id="sj_d04f631427a29323" onclick="changeJobState('d04f631427a29323', 'save', 'linkbar', true, ''); return false;" title="Save this job to my.indeed">
                   save job
                  </a>
                 </span>
                 <div class="edit_note_content" id="editsaved2_d04f631427a29323" style="display:none;">
                 </div>
                 <script>
                  if (!window['sj_result_d04f631427a29323']) {window['sj_result_d04f631427a29323'] = {};}window['sj_result_d04f631427a29323']['showSource'] = false; window['sj_result_d04f631427a29323']['source'] = "Kelly Services"; window['sj_result_d04f631427a29323']['loggedIn'] = false; window['sj_result_d04f631427a29323']['showMyJobsLinks'] = false;window['sj_result_d04f631427a29323']['undoAction'] = "unsave";window['sj_result_d04f631427a29323']['jobKey'] = "d04f631427a29323"; window['sj_result_d04f631427a29323']['myIndeedAvailable'] = true; window['sj_result_d04f631427a29323']['showMoreActionsLink'] = window['sj_result_d04f631427a29323']['showMoreActionsLink'] || false; window['sj_result_d04f631427a29323']['resultNumber'] = 10; window['sj_result_d04f631427a29323']['jobStateChangedToSaved'] = false; window['sj_result_d04f631427a29323']['searchState'] = "q=data&amp;l=Austin%2C+TX&amp;sort=date"; window['sj_result_d04f631427a29323']['basicPermaLink'] = "https://www.indeed.com"; window['sj_result_d04f631427a29323']['saveJobFailed'] = false; window['sj_result_d04f631427a29323']['removeJobFailed'] = false; window['sj_result_d04f631427a29323']['requestPending'] = false; window['sj_result_d04f631427a29323']['notesEnabled'] = false; window['sj_result_d04f631427a29323']['currentPage'] = "serp"; window['sj_result_d04f631427a29323']['sponsored'] = true;window['sj_result_d04f631427a29323']['showSponsor'] = true;window['sj_result_d04f631427a29323']['reportJobButtonEnabled'] = false; window['sj_result_d04f631427a29323']['showMyJobsHired'] = false; window['sj_result_d04f631427a29323']['showSaveForSponsored'] = true; window['sj_result_d04f631427a29323']['showJobAge'] = true;
                 </script>
                </div>
               </div>
               <div class="tab-container">
                <div class="sign-in-container result-tab">
                </div>
                <div class="tellafriend-container result-tab email_job_content">
                </div>
               </div>
              </div>
             </div>
             <div class="row result" data-advn="2155191336115633" data-jk="d3651049ccc706d2" data-tu="" id="pj_d3651049ccc706d2">
              <a class="jobtitle turnstileLink" data-tn-element="jobTitle" href="/pagead/clk?mo=r&amp;ad=-6NYlbfkN0DKYsWfXGebLeZezv9lqeOuR0I5YQGd9F2mP5RS9tO9dJW6BXeqEszZA-3ZKcEtunatCFM3-fMSQJtELRArrI4DXe_uRW0RBrDPxQ4xxoJcCos31_SULOXAo5aUAwwrzAR29fUssIg2LUAPt3aWjF-czxiRcdpTxRCgvLgquFvqz-LW8rb7rHOQn4VuC8DMFTYoG7FMIjNLhd1ThRBUp-Uy3HL9CVyPLvMSlBZZMW7dHUi6rceLPhsTp-1mS3_MYhFTpVn3FDfZgdyfXvPd2cGIUXDe01Epz0h2Vaj1uTzDOip3AeEeUrnLe1JjBfeJtgVsgdq2sk2Gw6sXYo6gwgKK3spUoakMfWKYlZLDweAqpGn7xY0JUVYoxNC_bUWIXfmHt-A5HGGcsbY9xYbMPBz32dnrAnICK-6yqI-1Vbb_1YuLpAO9dya6&amp;vjs=3&amp;p=2&amp;sk=&amp;fvj=1" id="sja2" onclick="setRefineByCookie([]); sjoc('sja2',0); convCtr('SJ')" onmousedown="sjomd('sja2'); clk('sja2');" rel="noopener nofollow" target="_blank" title="System Analyst">
               System Analyst
              </a>
              <br/>
              <div class="sjcl">
               <span class="company">
                <a class="turnstileLink" data-tn-element="companyName" href="/cmp/Trusource-Labs" onmousedown="this.href = appendParamsOnce(this.href, 'from=SERP&amp;campaignid=serp-linkcompanyname&amp;fromjk=d3651049ccc706d2&amp;jcid=528730143cd3265d')" rel="noopener" target="_blank">
                 Trusource Labs
                </a>
               </span>
               -
               <a class="ratingsLabel" data-tn-element="reviewStars" data-tn-variant="cmplinktst2" href="/cmp/Trusource-Labs/reviews" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=cmplinktst2&amp;from=SERP&amp;jt=System+Analyst&amp;fromjk=d3651049ccc706d2&amp;jcid=528730143cd3265d');" rel="noopener" target="_blank" title="Trusource Labs reviews">
                <span class="ratings">
                 <span class="rating" style="width:43.8px">
                  <!-- -->
                 </span>
                </span>
                <span class="slNoUnderline">
                 36 reviews
                </span>
               </a>
               -
               <span class="location">
                Austin, TX 78744
               </span>
              </div>
              <div class="">
               <table border="0" cellpadding="0" cellspacing="0">
                <tr>
                 <td class="snip">
                  <span class="summary">
                   Create and maintain system and
                   <b>
                    data
                   </b>
                   reporting related to the systems and servers infrastructure. Responsible for setup, maintaining and monitoring internal...
                  </span>
                 </td>
                </tr>
               </table>
              </div>
              <div class="sjCapt">
               <div class="iaP">
                <span class="iaLabel">
                 Easily apply
                </span>
               </div>
               <div class="result-link-bar-container">
                <div class="result-link-bar">
                 <span class=" sponsoredGray ">
                  Sponsored
                 </span>
                 -
                 <span class="tt_set" id="tt_set_11">
                  <a class="sl resultLink save-job-link " href="#" id="sj_d3651049ccc706d2" onclick="changeJobState('d3651049ccc706d2', 'save', 'linkbar', true, ''); return false;" title="Save this job to my.indeed">
                   save job
                  </a>
                 </span>
                 <div class="edit_note_content" id="editsaved2_d3651049ccc706d2" style="display:none;">
                 </div>
                 <script>
                  if (!window['sj_result_d3651049ccc706d2']) {window['sj_result_d3651049ccc706d2'] = {};}window['sj_result_d3651049ccc706d2']['showSource'] = false; window['sj_result_d3651049ccc706d2']['source'] = "Indeed"; window['sj_result_d3651049ccc706d2']['loggedIn'] = false; window['sj_result_d3651049ccc706d2']['showMyJobsLinks'] = false;window['sj_result_d3651049ccc706d2']['undoAction'] = "unsave";window['sj_result_d3651049ccc706d2']['jobKey'] = "d3651049ccc706d2"; window['sj_result_d3651049ccc706d2']['myIndeedAvailable'] = true; window['sj_result_d3651049ccc706d2']['showMoreActionsLink'] = window['sj_result_d3651049ccc706d2']['showMoreActionsLink'] || false; window['sj_result_d3651049ccc706d2']['resultNumber'] = 11; window['sj_result_d3651049ccc706d2']['jobStateChangedToSaved'] = false; window['sj_result_d3651049ccc706d2']['searchState'] = "q=data&amp;l=Austin%2C+TX&amp;sort=date"; window['sj_result_d3651049ccc706d2']['basicPermaLink'] = "https://www.indeed.com"; window['sj_result_d3651049ccc706d2']['saveJobFailed'] = false; window['sj_result_d3651049ccc706d2']['removeJobFailed'] = false; window['sj_result_d3651049ccc706d2']['requestPending'] = false; window['sj_result_d3651049ccc706d2']['notesEnabled'] = false; window['sj_result_d3651049ccc706d2']['currentPage'] = "serp"; window['sj_result_d3651049ccc706d2']['sponsored'] = true;window['sj_result_d3651049ccc706d2']['showSponsor'] = true;window['sj_result_d3651049ccc706d2']['reportJobButtonEnabled'] = false; window['sj_result_d3651049ccc706d2']['showMyJobsHired'] = false; window['sj_result_d3651049ccc706d2']['showSaveForSponsored'] = true; window['sj_result_d3651049ccc706d2']['showJobAge'] = true;
                 </script>
                </div>
               </div>
               <div class="tab-container">
                <div class="sign-in-container result-tab">
                </div>
                <div class="tellafriend-container result-tab email_job_content">
                </div>
               </div>
              </div>
             </div>
             <div class="row sjlast result" data-advn="8777669303266684" data-jk="4ae29996c0cf2693" data-tu="" id="pj_4ae29996c0cf2693">
              <a class="jobtitle turnstileLink" data-tn-element="jobTitle" href="/pagead/clk?mo=r&amp;ad=-6NYlbfkN0BqJjBsvJkVIRVupdyx-l7jJlkPL5nU6SVET5Mq4mDejWWyVHqkHhIOF8Jj9vV_OYB4wAjRJXJfg9nJDHrEeIILCpaW5XVyWUkQOHeYAKzntWrfATy9cBTm2f7IWXcB0pQcq62bIpkJc5JVs2RlWCe8cKNsdHFxLEXPJj3r0Y06I3Jumvfsy-ohWf35CcYEA40VQTeo2EYGRSM4u6R0Zxp_OQ0aoMC-hBjFppvahZBaOEsDMCJHAsFqP2sCzfadSPk5oUhvoGrBJL79jE0tVWSYyIC0mklpshvSvm3dgoHCItEwwdK4AccfCNBvV_41gtGb0MczYd4qpuzpLthOZcovMVXWROy6xbLgPeNOQuLoEpsVhca5Gr5HoG-Wn_hek8AxU7re5wZteQ7PSfeq0o-9m7VeCcqikYr7snRGPPA4w8Wmej3hUDlvwIpqpzWrJmviS0WYka5D3enHbAeITDOILorcoBNqDtt291RHJt7kmXZXN0jaho2ZAqeBEV3oCYwNWSaLoskuhhl_sfpD3CPiJIJP7AJLBYsyGqC9fZTUae4KJOlA4R48jmlzH-0IwKe6GUtf3hfiVw==&amp;vjs=3&amp;p=3&amp;sk=&amp;fvj=0" id="sja3" onclick="setRefineByCookie([]); sjoc('sja3',0); convCtr('SJ')" onmousedown="sjomd('sja3'); clk('sja3');" rel="noopener nofollow" target="_blank" title="Sr. Analyst, Data Analytics">
               Sr. Analyst,
               <b>
                Data
               </b>
               Analytics
              </a>
              <br/>
              <div class="sjcl">
               <span class="company">
                <a class="turnstileLink" data-tn-element="companyName" href="/cmp/General-Motors" onmousedown="this.href = appendParamsOnce(this.href, 'from=SERP&amp;campaignid=serp-linkcompanyname&amp;fromjk=4ae29996c0cf2693&amp;jcid=116680a29a847a70')" rel="noopener" target="_blank">
                 General Motors
                </a>
               </span>
               -
               <a class="ratingsLabel" data-tn-element="reviewStars" data-tn-variant="cmplinktst2" href="/cmp/General-Motors/reviews" onmousedown="this.href = appendParamsOnce(this.href, '?campaignid=cmplinktst2&amp;from=SERP&amp;jt=Sr.+Analyst%2C+Data+Analytics&amp;fromjk=4ae29996c0cf2693&amp;jcid=116680a29a847a70');" rel="noopener" target="_blank" title="General Motors reviews">
                <span class="ratings">
                 <span class="rating" style="width:52.2px">
                  <!-- -->
                 </span>
                </span>
                <span class="slNoUnderline">
                 4,945 reviews
                </span>
               </a>
               -
               <span class="location">
                United States
               </span>
              </div>
              <div class="">
               <table border="0" cellpadding="0" cellspacing="0">
                <tr>
                 <td class="snip">
                  <span class="summary">
                   Experienced with
                   <b>
                    data
                   </b>
                   acquisition and manipulation, modeling, and analyzing
                   <b>
                    data
                   </b>
                   from core financial platforms (SAP, Hyperion, Enterprise
                   <b>
                    Data
                   </b>
                   Warehouse) and...
                  </span>
                 </td>
                </tr>
               </table>
              </div>
              <div class="sjCapt">
               <div class="result-link-bar-container">
                <div class="result-link-bar">
                 <span class=" sponsoredGray ">
                  Sponsored
                 </span>
                 -
                 <span class="tt_set" id="tt_set_12">
                  <a class="sl resultLink save-job-link " href="#" id="sj_4ae29996c0cf2693" onclick="changeJobState('4ae29996c0cf2693', 'save', 'linkbar', true, ''); return false;" title="Save this job to my.indeed">
                   save job
                  </a>
                 </span>
                 <div class="edit_note_content" id="editsaved2_4ae29996c0cf2693" style="display:none;">
                 </div>
                 <script>
                  if (!window['sj_result_4ae29996c0cf2693']) {window['sj_result_4ae29996c0cf2693'] = {};}window['sj_result_4ae29996c0cf2693']['showSource'] = false; window['sj_result_4ae29996c0cf2693']['source'] = "General Motors"; window['sj_result_4ae29996c0cf2693']['loggedIn'] = false; window['sj_result_4ae29996c0cf2693']['showMyJobsLinks'] = false;window['sj_result_4ae29996c0cf2693']['undoAction'] = "unsave";window['sj_result_4ae29996c0cf2693']['jobKey'] = "4ae29996c0cf2693"; window['sj_result_4ae29996c0cf2693']['myIndeedAvailable'] = true; window['sj_result_4ae29996c0cf2693']['showMoreActionsLink'] = window['sj_result_4ae29996c0cf2693']['showMoreActionsLink'] || false; window['sj_result_4ae29996c0cf2693']['resultNumber'] = 12; window['sj_result_4ae29996c0cf2693']['jobStateChangedToSaved'] = false; window['sj_result_4ae29996c0cf2693']['searchState'] = "q=data&amp;l=Austin%2C+TX&amp;sort=date"; window['sj_result_4ae29996c0cf2693']['basicPermaLink'] = "https://www.indeed.com"; window['sj_result_4ae29996c0cf2693']['saveJobFailed'] = false; window['sj_result_4ae29996c0cf2693']['removeJobFailed'] = false; window['sj_result_4ae29996c0cf2693']['requestPending'] = false; window['sj_result_4ae29996c0cf2693']['notesEnabled'] = false; window['sj_result_4ae29996c0cf2693']['currentPage'] = "serp"; window['sj_result_4ae29996c0cf2693']['sponsored'] = true;window['sj_result_4ae29996c0cf2693']['showSponsor'] = true;window['sj_result_4ae29996c0cf2693']['reportJobButtonEnabled'] = false; window['sj_result_4ae29996c0cf2693']['showMyJobsHired'] = false; window['sj_result_4ae29996c0cf2693']['showSaveForSponsored'] = true; window['sj_result_4ae29996c0cf2693']['showJobAge'] = true;
                 </script>
                </div>
               </div>
               <div class="tab-container">
                <div class="sign-in-container result-tab">
                </div>
                <div class="tellafriend-container result-tab email_job_content">
                </div>
               </div>
              </div>
             </div>
            </div>
            <script type="text/javascript">
             function ptk(st,p) {
      document.cookie = 'PTK="tk=&type=jobsearch&subtype=' + st + (p ? '&' + p : '')
         + (st == 'pagination' ? '&fp=1' : '')
        +'"; path=/';
    }
            </script>
            <script type="text/javascript">
             function pclk(event) {
      var evt = event || window.event;
      var target = evt.target || evt.srcElement;
      var el = target.nodeType == 1 ? target : target.parentNode;
      var tag = el.tagName.toLowerCase();
      if (tag == 'span' || tag == 'a') {
        ptk('pagination');
      }
      return true;
    }
    function addPPUrlParam(obj) {
      var pp = obj.getAttribute('data-pp');
      var href = obj.getAttribute('href');
      if (pp && href) {
        obj.setAttribute('href', href + '&pp=' + pp);
      }
    }
            </script>
            <div class="pagination" onmousedown="pclk(event);">
             Results Page:
             <b>
              1
             </b>
             <a data-pp="AAoAAAAAAAAAAAAAAAEwDLSsAQAcV1jqsjqOKQdGumgBJAT-Yo-HWrbDkX2CwyG86Apc" href="/jobs?q=data&amp;l=Austin%2C+TX&amp;sort=date&amp;start=10" onmousedown="addPPUrlParam &amp;&amp; addPPUrlParam(this);">
              <span class="pn">
               2
              </span>
             </a>
             <a data-pp="ABQAAAAAAAAAAAAAAAEwDLSsAQEBCE51hAlCfeGZ8O_UsuOyN6zj5IGDm6sgvSUP5DDbEs3njnjE-i9NZOChjUXIvxCoEDY" href="/jobs?q=data&amp;l=Austin%2C+TX&amp;sort=date&amp;start=20" onmousedown="addPPUrlParam &amp;&amp; addPPUrlParam(this);">
              <span class="pn">
               3
              </span>
             </a>
             <a data-pp="AB4AAAAAAAAAAAAAAAEwDLSsAQEBCFEOhNUnfKndZCcNYdK0j6XUd3-wYiKCQGZ1XS2LL27RLbA5CyqGTGi3oBK8jHuiplyovqNF6vWf6RoAauMG4aKpHGu-aA" href="/jobs?q=data&amp;l=Austin%2C+TX&amp;sort=date&amp;start=30" onmousedown="addPPUrlParam &amp;&amp; addPPUrlParam(this);">
              <span class="pn">
               4
              </span>
             </a>
             <a data-pp="ACgAAAAAAAAAAAAAAAEwDLSsAQIBCBIGAqp4XIqR5phQC-dgSrFB7ga7ZrhPYjrbuF1wJCNJTVtKCHoWGXA0roELZvhu6OrNAa3pNDA8tq8HA4pghteQOIgTG7XrggXgtDA0nS3pym7QGonW" href="/jobs?q=data&amp;l=Austin%2C+TX&amp;sort=date&amp;start=40" onmousedown="addPPUrlParam &amp;&amp; addPPUrlParam(this);">
              <span class="pn">
               5
              </span>
             </a>
             <a data-pp="AAoAAAAAAAAAAAAAAAEwDLSsAQAcV1jqsjqOKQdGumgBJAT-Yo-HWrbDkX2CwyG86Apc" href="/jobs?q=data&amp;l=Austin%2C+TX&amp;sort=date&amp;start=10" onmousedown="addPPUrlParam &amp;&amp; addPPUrlParam(this);">
              <span class="pn">
               <span class="np">
                Next »
               </span>
              </span>
             </a>
            </div>
            <div class="related_searches">
             <div class="related_searches_list">
              <b>
               People also searched:
              </b>
              <ul class="relatedQueries-listView-pageFirst">
               <li class="relatedQueries-listItem-pageFirst rightBorder">
                <a href="/q-Data-Analyst-l-Austin,-TX-jobs.html" onmousedown="this.href = appendParamsOnce(this.href, '?from=relatedQueries&amp;saIdx=1&amp;rqf=1&amp;parentQnorm=data');">
                 data analyst
                </a>
               </li>
               <li class="relatedQueries-listItem-pageFirst rightBorder">
                <a href="/q-Data-Entry-l-Austin,-TX-jobs.html" onmousedown="this.href = appendParamsOnce(this.href, '?from=relatedQueries&amp;saIdx=2&amp;rqf=1&amp;parentQnorm=data');">
                 data entry
                </a>
               </li>
               <li class="relatedQueries-listItem-pageFirst rightBorder">
                <a href="/q-Analyst-l-Austin,-TX-jobs.html" onmousedown="this.href = appendParamsOnce(this.href, '?from=relatedQueries&amp;saIdx=3&amp;rqf=1&amp;parentQnorm=data');">
                 analyst
                </a>
               </li>
               <li class="relatedQueries-listItem-pageFirst rightBorder">
                <a href="/q-Data-Scientist-l-Austin,-TX-jobs.html" onmousedown="this.href = appendParamsOnce(this.href, '?from=relatedQueries&amp;saIdx=4&amp;rqf=1&amp;parentQnorm=data');">
                 data scientist
                </a>
               </li>
               <li class="relatedQueries-listItem-pageFirst rightBorder">
                <a href="/q-Entry-Level-l-Austin,-TX-jobs.html" onmousedown="this.href = appendParamsOnce(this.href, '?from=relatedQueries&amp;saIdx=5&amp;rqf=1&amp;parentQnorm=data');">
                 entry level
                </a>
               </li>
               <li class="relatedQueries-listItem-pageFirst rightBorder">
                <a href="/q-SQL-l-Austin,-TX-jobs.html" onmousedown="this.href = appendParamsOnce(this.href, '?from=relatedQueries&amp;saIdx=6&amp;rqf=1&amp;parentQnorm=data');">
                 sql
                </a>
               </li>
               <li class="relatedQueries-listItem-pageFirst rightBorder">
                <a href="/q-Part-Time-l-Austin,-TX-jobs.html" onmousedown="this.href = appendParamsOnce(this.href, '?from=relatedQueries&amp;saIdx=7&amp;rqf=1&amp;parentQnorm=data');">
                 part time
                </a>
               </li>
               <li class="relatedQueries-listItem-pageFirst rightBorder">
                <a href="/q-Research-l-Austin,-TX-jobs.html" onmousedown="this.href = appendParamsOnce(this.href, '?from=relatedQueries&amp;saIdx=8&amp;rqf=1&amp;parentQnorm=data');">
                 research
                </a>
               </li>
               <li class="relatedQueries-listItem-pageFirst rightBorder">
                <a href="/q-Business-Analyst-l-Austin,-TX-jobs.html" onmousedown="this.href = appendParamsOnce(this.href, '?from=relatedQueries&amp;saIdx=9&amp;rqf=1&amp;parentQnorm=data');">
                 business analyst
                </a>
               </li>
               <li class="relatedQueries-listItem-pageFirst">
                <a href="/q-Database-l-Austin,-TX-jobs.html" onmousedown="this.href = appendParamsOnce(this.href, '?from=relatedQueries&amp;saIdx=10&amp;rqf=1&amp;parentQnorm=data');">
                 database
                </a>
               </li>
              </ul>
             </div>
             <style type="text/css">
              .relatedQueries-listView-pageFirst {
                        list-style-type: none;
                        margin: 0;
                        padding: 0;
                    }
                    .relatedQueries-listItem-pageFirst {
                        display: inline-block;
                    }
                    .rightBorder {
                        border-right: 1px solid #77c;
                        margin-right: 5px;
                        padding-right: 5px;
                    }
                    .related_searches_list > * {
                        display: inline;
                    }
             </style>
             <div class="related_searches_list">
              <b class="related_searches_title">
               Related Forums:
              </b>
              <a href="/forum/job/Management-Analyst.html" onclick="logRSC('rfsclk', 'Management Analyst', '1c6u39dhta24qbca')">
               Management Analyst
              </a>
              -
              <a href="/forum/job/Market-Research-Analyst.html" onclick="logRSC('rfsclk', 'Market Research Analyst', '1c6u39dhta24qbca')">
               Market Research Analyst
              </a>
              -
              <a href="/forum/job/Systems-Analyst.html" onclick="logRSC('rfsclk', 'Systems Analyst', '1c6u39dhta24qbca')">
               Systems Analyst
              </a>
              -
              <a href="/forum/loc/Austin-Texas.html" onclick="logRSC('rfsclk', 'Austin, Texas', '1c6u39dhta24qbca')">
               Austin, Texas
              </a>
             </div>
             <div class="related_searches_list">
              <b class="related_searches_title">
               <a href="/salaries/Salaries,-Austin-TX?from=serp" onclick="logRSC('rssclk', 'Austin, TX', '1c6u39dhta24qbca')">
                Salaries in Austin, TX:
               </a>
              </b>
              <a href="/salaries/Management-Analyst-Salaries,-Austin,%20TX?from=serp" onclick="logRSC('rssclk', 'Management Analyst', '1c6u39dhta24qbca')">
               Management Analyst salary
              </a>
              -
              <a href="/salaries/Market-Researcher-Salaries,-Austin,%20TX?from=serp" onclick="logRSC('rssclk', 'Market Research Analyst', '1c6u39dhta24qbca')">
               Market Research Analyst salary
              </a>
              -
              <a href="/salaries/Systems-Analyst-Salaries,-Austin,%20TX?from=serp" onclick="logRSC('rssclk', 'Systems Analyst', '1c6u39dhta24qbca')">
               Systems Analyst salary
              </a>
             </div>
            </div>
           </td>
           <td id="auxCol" role="complementary">
            <div id="jobalertswrapper">
             <div class="open jaui " id="jobalerts">
              <div class="jobalertlabel">
               <span class="jobalerts_title" id="jobalertlabel">
                <span aria-label="alert icon" class="ico" role="img">
                </span>
                Be the first to see new
                <b>
                 data jobs in Austin, TX
                </b>
               </span>
              </div>
              <div class="jaform" id="jobalertform">
               <span class="ja_checkmark_ui" id="jobalerttext">
               </span>
               <span id="jobalertsending">
               </span>
               <div id="jobalertmessage">
                <form action="/alert" method="POST" onsubmit="return addAlertFormSubmit()">
                 <input name="a" type="hidden" value="add"/>
                 <input name="q" type="hidden" value="data"/>
                 <input name="l" type="hidden" value="Austin, TX"/>
                 <input name="radius" type="hidden" value="25"/>
                 <input name="noscript" type="hidden" value="1"/>
                 <input name="tk" type="hidden" value="1c6u39dhta24qbca"/>
                 <input id="alertverified" name="verified" type="hidden" value="0"/>
                 <input name="alertparams" type="hidden" value=""/>
                 <label for="alertemail">
                  My email:
                 </label>
                 <input id="alertemail" maxlength="100" name="email" size="25" type="text" value=""/>
                 <label for="recjobalert" id="recjobalertlabel">
                  <input checked="" id="recjobalert" name="recjobalert" type="checkbox"/>
                  <span>
                   Also get an email with jobs recommended just for me
                  </span>
                 </label>
                 <span class="indeed-serp-button">
                  <span class="indeed-serp-button-inner">
                   <input class="indeed-serp-button-label" id="alertsubmit" type="submit" value="Activate"/>
                  </span>
                 </span>
                 <style type="text/css">
                  .indeed-serp-button { cursor : pointer !important; display : inline-block !important; padding : 1px !important; height : 31px !important; -moz-border-radius : 7px !important; border-radius : 7px !important; position : relative !important; text-decoration : none !important;background-color:#79788B; filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#BCBBCD', endColorstr='#79788B', GradientType=0);background-image: -webkit-gradient(linear, center top, center bottom, from(#BCBBCD), to(#79788B)) !important;background-image: -webkit-linear-gradient(top, #BCBBCD, #79788B) !important;background-image: -moz-linear-gradient(top, #BCBBCD, #79788B) !important;background-image: -o-linear-gradient(top, #BCBBCD, #79788B) !important;background-image: -ms-linear-gradient(top, #BCBBCD, #79788B) !important;background-image: linear-gradient(top, #BCBBCD, #79788B) !important;-webkit-box-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;-moz-box-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;box-shadow: 0 1px 2px rgba(0,0,0,0.2) !important; } #indeed-ia-1329175190441-0:link, #indeed-ia-1329175190441-0:visited, #indeed-ia-1329175190441-0:hover, #indeed-ia-1329175190441-0:active { border : 0 !important; text-decoration : none !important; }
    
        .indeed-serp-button:hover { filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#6D99F6', endColorstr='#1B45A3', GradientType=0);background-image: -webkit-gradient(linear, center top, center bottom, from(#6D99F6), to(#1B45A3)) !important;background-image: -webkit-linear-gradient(top, #6D99F6, #1B45A3) !important;background-image: -moz-linear-gradient(top, #6D99F6, #1B45A3) !important;background-image: -o-linear-gradient(top, #6D99F6, #1B45A3) !important;background-image: -ms-linear-gradient(top, #6D99F6, #1B45A3) !important;background-image: linear-gradient(top, #6D99F6, #1B45A3) !important; }
    
        .indeed-apply-state-clicked .indeed-serp-button,
        .indeed-serp-button:active { filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#B3BACA', endColorstr='#7C8493', GradientType=0);background-image: -webkit-gradient(linear, center top, center bottom, from(#B3BACA), to(#7C8493)) !important;background-image: -webkit-linear-gradient(top, #B3BACA, #7C8493) !important;background-image: -moz-linear-gradient(top, #B3BACA, #7C8493) !important;background-image: -o-linear-gradient(top, #B3BACA, #7C8493) !important;background-image: -ms-linear-gradient(top, #B3BACA, #7C8493) !important;background-image: linear-gradient(top, #B3BACA, #7C8493) !important;-webkit-box-shadow: none !important;-moz-box-shadow: none !important;box-shadow: none !important; }
    
        .indeed-serp-button-inner { display : inline-block !important; height : 31px !important; -moz-border-radius : 6px !important; border-radius : 6px !important; font : 18px 'Helvetica Neue','Helvetica',Arial !important; font-weight : 200 !important; text-decoration : none !important; text-shadow : 0px 1px #F1F1F4 !important;background-color:#D9D9E2;  color: #FF6703;filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#FAFAFB', endColorstr='#D9D9E2', GradientType=0);background-image: -webkit-gradient(linear, center top, center bottom, from(#FAFAFB), to(#D9D9E2)) !important;background-image: -webkit-linear-gradient(top, #FAFAFB, #D9D9E2) !important;background-image: -moz-linear-gradient(top, #FAFAFB, #D9D9E2) !important;background-image: -o-linear-gradient(top, #FAFAFB, #D9D9E2) !important;background-image: -ms-linear-gradient(top, #FAFAFB, #D9D9E2) !important;background-image: linear-gradient(top, #FAFAFB, #D9D9E2) !important; }
    
        .indeed-serp-button:active .indeed-serp-button-inner { filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#E8E8E9', endColorstr='#CBCBD3', GradientType=0);background-image: -webkit-gradient(linear, center top, center bottom, from(#E8E8E9), to(#CBCBD3)) !important;background-image: -webkit-linear-gradient(top, #E8E8E9, #CBCBD3) !important;background-image: -moz-linear-gradient(top, #E8E8E9, #CBCBD3) !important;background-image: -o-linear-gradient(top, #E8E8E9, #CBCBD3) !important;background-image: -ms-linear-gradient(top, #E8E8E9, #CBCBD3) !important;background-image: linear-gradient(top, #E8E8E9, #CBCBD3) !important; }
    
        .indeed-serp-button-label {cursor: pointer; text-align : center !important; border:0; background: transparent;font-size: 12px; font-family: Arial, sans-serif; padding:3px 14px 2px 12px; margin:0; line-height: 26px; }
    
        .indeed-serp-button:active .indeed-serp-button-label,
        .indeed-apply-state-clicked .indeed-serp-button-label { -ms-filter: "progid:DXImageTransform.Microsoft.Alpha(Opacity=0.75)" !important;filter: alpha(opacity=75) !important;-moz-opacity: 0.75 !important;-khtml-opacity: 0.75 !important;opacity: 0.75 !important; }
    
        #alertemail {  height: 27px; line-height: 24px; padding-left: 6px; padding-right: 6px; font-size: 14px; font-family: Arial, sans-serif; }
    
        .jobalertform-terms-outer-wrapper label {  position: fixed;  font-size: 1px;  transform: scale(0.3);  }
    
        .jobalertform-terms-inner-wrapper {  position: relative;  z-index: 20;  width: 50%;  background: #ebebeb;  height: 12px;  }
                 </style>
                 <div class="g-recaptcha" id="invisible-recaptcha-div">
                 </div>
                </form>
               </div>
              </div>
             </div>
            </div>
            <script type="text/javascript">
             var addAlertFormSubmit = function() {
    		var email = document.getElementById('alertemail').value;
    		var verified = document.getElementById('alertverified').value;
    		var recJACheckbox = document.getElementById('recjobalert');
    		var tacCheckbox = document.getElementById('termsandconditionscheckbox');
    		var recjobalertchecked = recJACheckbox ? recJACheckbox.checked : false;
    		var termsandconditionschecked = tacCheckbox ? tacCheckbox.checked : false;
    
    		return addalertdelegate(
    			'data',
    			'Austin%2C+TX',
    			'',
    			email,
    			'1c6u39dhta24qbca',
    			verified,
    			true,
    			'661982',
    			'US',
    			'fe8488f22fdc0b8f4a571f572ebdc656',
    			recjobalertchecked,
    			false,
    			termsandconditionschecked,
    			true
    		);
    	}
            </script>
            <div id="femp_list">
             <div class="femp_header">
              Company with data jobs
             </div>
             <div class="femp_item">
              <div class="femp_logo">
               <a href="/cmp/Kelly-Services" onmousedown="this.href = appendParamsOnce(this.href, 'tk=1c6u39dhta24qbca&amp;campaignid=femp&amp;from=femp');" rel="noopener" target="_blank">
                <img alt="Kelly Services" src="https://d2q79iu7y748jz.cloudfront.net/s/_logo/239d6d947474c2573dea9c4ca5c7ad1b.png" width="120"/>
               </a>
              </div>
              <div class="femp_cmp">
               <a class="femp_cmp_link" href="/cmp/Kelly-Services" onmousedown="this.href = appendParamsOnce(this.href, 'tk=1c6u39dhta24qbca&amp;campaignid=femp&amp;from=femp');" rel="noopener" target="_blank">
                Kelly Services
               </a>
              </div>
              <div class="femp_desc">
               We are proud to employ nearly 500,000 people around the world and connect thousands more with work through our network of talent partners.
              </div>
              <div class="featemp_sjl" id="featemp_sj" style="display: none">
               <div id="indJobContent">
               </div>
              </div>
              <script type="text/javascript">
               var ind_nr = true;
            var ind_pub = '8772657697788355';
            var ind_el = 'indJobContent';
            var ind_pf = '';
            var ind_q = '';
            var ind_fcckey = 'fa59859d11ecf8a4';
            var ind_l = 'Austin, TX';
            var ind_chnl = 'Kelly Services';
            var ind_n = 3;
            var ind_d = '';
            var ind_t = 60;
            var ind_c = 30;
            var ind_rq = 'data';
              </script>
              <script async="" src="/jobroll-widget-v3-fc2.js" type="text/javascript">
              </script>
              <div class="femp_links">
               <div class="jobs">
                <a href="/jobs?q=company%3A%22Kelly+Services%22&amp;l=Austin%2C+TX" onmousedown="this.href = appendParamsOnce(this.href, '&amp;from=femp');">
                 Jobs
                </a>
               </div>
               <div class="reviews">
                <a href="/cmp/Kelly-Services/reviews" onmousedown="this.href = appendParamsOnce(this.href, '&amp;campaignid=femp&amp;from=femp');">
                 Reviews (10,404)
                </a>
                <span class="ratings">
                 <span class="rating" style="width:44.4px">
                  <!-- -->
                 </span>
                </span>
               </div>
               <div class="photos">
                <a href="/cmp/Kelly-Services/photos" onmousedown="this.href = appendParamsOnce(this.href, '&amp;campaignid=femp&amp;from=femp');">
                 Photos (32)
                </a>
               </div>
               <div class="salaries">
                <a href="/cmp/Kelly-Services/salaries" onmousedown="this.href = appendParamsOnce(this.href, '&amp;campaignid=femp&amp;from=femp');">
                 Salaries (10,457)
                </a>
               </div>
              </div>
             </div>
            </div>
            <div id="univsrch-salary-v3">
             <div id="univsrch-salary-info">
              <div id="univsrch-salary-title">
               Data Entry Clerk salaries in Austin, TX
              </div>
              <div id="univsrch-salary-eval">
               <div class="v3" id="univsrch-salary-rates">
                <p class="v3" id="univsrch-salary-currentsalary">
                 <b>
                  $12.70
                 </b>
                 per hour
                </p>
               </div>
               <div class="v3" id="univsrch-salary-stats">
                <p class="v3" id="univsrch-salary-stats-para">
                 Based on 1,907 salaries
                </p>
               </div>
               <div id="univsrch-sal-distribution">
                <ul>
                 <li style="height:65.54529703330537%; width:9.2%; left:0.0%;">
                 </li>
                 <li style="height:81.42344622661558%; width:9.2%; left:10.088999999999999%;">
                 </li>
                 <li style="height:89.67257488941567%; width:9.2%; left:20.177999999999997%;">
                 </li>
                 <li class="univsrch-sal-highlight" style="height:90.0%; width:9.2%; left:30.266999999999996%;">
                 </li>
                 <li style="height:84.0011323593686%; width:9.2%; left:40.355999999999995%;">
                 </li>
                 <li style="height:74.02355396886965%; width:9.2%; left:50.44499999999999%;">
                 </li>
                 <li style="height:62.30388491467639%; width:9.2%; left:60.53399999999999%;">
                 </li>
                 <li style="height:50.536533422526794%; width:9.2%; left:70.62299999999999%;">
                 </li>
                 <li style="height:39.78218768403822%; width:9.2%; left:80.71199999999999%;">
                 </li>
                 <li style="height:30.56217335542356%; width:9.2%; left:90.80099999999999%;">
                 </li>
                </ul>
                <div class="univsrch-sal-min univsrch-sal-caption float-left">
                 <div class="float-left" style="width: 13.07px;">
                 </div>
                 <span>
                  Min
                  <br>
                   $7.25
                  </br>
                 </span>
                </div>
                <div class="univsrch-sal-max univsrch-sal-caption float-right">
                 <div class="float-right" style="width: 13.07px;">
                 </div>
                 <span>
                  Max
                  <br>
                   $24.85
                  </br>
                 </span>
                </div>
               </div>
               <div id="univsrch-salary-link">
                <a href="/salaries/Data-Entry-Clerk-Salaries,-Austin-TX" onmousedown="this.href = appendParamsOnce(this.href, 'from=serpsalaryblock');" title="Data Entry Clerk salaries by company in Austin, TX">
                 Data Entry Clerk salaries by company in Austin, TX
                </a>
               </div>
              </div>
             </div>
            </div>
            <script type="text/javascript">
             usBindSalaryWidgetLoggingNew();
            </script>
            <script id="jaFloatScript" type="text/javascript">
             floatJobAlert();
            </script>
           </td>
          </tr>
         </table>
         <script>
          var focusHandlers = [];
    var linkHighlighter = new LinkHighlighter();
    focusHandlers.push(googBind(linkHighlighter.fadeToOriginalColor, linkHighlighter));
    var lostFocusHandlers = [];
    lostFocusHandlers.push(googBind(linkHighlighter.clickedAway, linkHighlighter, "#551a8b"));
    
    if (!showVjOnSerp) {
        var didYouApplyPrompt = new DidYouApplyPrompt('1c6u39dhta24qbca', 60, 'serp',
            false);
        focusHandlers.push(googBind(didYouApplyPrompt.returnedToPage, didYouApplyPrompt));
        lostFocusHandlers.push(googBind(didYouApplyPrompt.leftPage, didYouApplyPrompt));
        didYouApplyPrompt.dyaChangeFromCookie();
    }
    var clickTime = new ClickTime(window.tk, 'serp', 'jobtitle', focusHandlers, lostFocusHandlers);
         </script>
         <script type="text/javascript">
          if (showVjOnSerp) {
        viewJobOnSerp.bindJobKeys(jobKeysWithInfo);
        if (vjUrl) {
          var vjFrom = viewJobOnSerp.getFrom();
          var vjTk = viewJobOnSerp.getTk();
          viewJobOnSerp.renderOnJobKey(vjk, vjFrom, vjTk, undefined, function() {
            window.location.replace(vjUrl);
          });
        }
    }
         </script>
         <script>
          usBindRightRailLogging();
         </script>
         <style type="text/css">
          .row.result {
            margin-bottom: 4px;
        }
    
        .clickcard:hover {
            border: 1px solid #d8d8d8 !important;
            cursor: pointer !important;
        }
    
        #jobalertswrapper,
        #femp_list,
        #picard-profile-completeness-widget,
        #univsrch-salary-v3,
        .rightRailAd {
            margin-left: 12px;
        }
    
        .ltr #serpRecommendations .row, .ltr .row.result {
            margin-right: -24px;
        }
    
        .ltr #serpRecommendations .row, .ltr .row.result.vjs-highlight {
            margin-right: -29px;
        }
         </style>
         <script>
          window['recaptchaSitekeyV2'] = "6Lc1uUEUAAAAAPHQRK9uCuJsLbJbUpwmnx5SARHu";
            window['recaptchaSitekeyInvisible'] = "6Lc5uUEUAAAAAHBFgzyc9no20EC0e7A-_R0QFgww";
         </script>
         <style type="text/css">
          #secondary_nav a,
    #secondary_nav a:link,
    #secondary_nav a:visited { color: #77c; text-decoration: none; }
    #secondary_nav a:hover { text-decoration: underline;  }
         </style>
         <!-- jobs -->
         <div id="footerWrapper" role="contentinfo" style="text-align:center;">
          <div id="footer" style="text-align:left;">
           <div class="separator_bottom">
           </div>
           <div id="secondary_nav">
            <div style="margin: 1em;">
             <span class="gaj_heading">
              Indeed helps people get jobs:
             </span>
             <a class="sl" href="/promo/gotajob" onmousedown="null">
              Over 10 million stories shared
             </a>
            </div>
            <a href="/" id="jobs_product_link" title="Jobs">
             Jobs
            </a>
            -
            <a href="/jobtrends/category-trends">
             Job Category Trends
            </a>
            -
            <a href="/career-advice?isid=jasx_us-en&amp;ikw=jsfooter" id="careeradvice_product_link" title="Career Advice">
             Career Advice
            </a>
            -
            <script type="text/javascript">
             var jobsProductLink = document.getElementById('jobs_product_link');
            </script>
            <a href="http://www.hiringlab.org">
             Hiring Lab
            </a>
            -
            <a href="/find-jobs.jsp">
             Browse Jobs
            </a>
            -
            <a href="/tools/jobseeker/">
             Tools
            </a>
            -
            <a href="http://www.indeed.jobs">
             Work at Indeed
            </a>
            -
            <a href="/publisher">
             API
            </a>
            -
            <a href="/intl/en/about.html">
             <span style="white-space: nowrap;">
              About
             </span>
            </a>
            -
            <a href="https://indeed.zendesk.com/hc/en-us">
             Help Center
            </a>
            <style type="text/css">
             #footer-legal {
                        margin-top: 10px;
                        font-size: 9pt;
                    }
            </style>
            <div id="footer-legal">
             <div class="legal-footer">
              ©2018 Indeed -
              <a href="/legal">
               Cookies, Privacy and Terms
              </a>
             </div>
            </div>
           </div>
          </div>
          <style type="text/css">
           /*Need to update this file when there is a update on frontend-icl library, details see ITAD-358*/
    .icl-Button{box-sizing:border-box;display:inline-block;vertical-align:middle;font-family:Helvetica Neue,Helvetica,Arial,Roboto,Noto,sans-serif;font-size:1.125rem;line-height:1.8rem;font-weight:700;text-decoration:none;text-overflow:ellipsis;white-space:nowrap;filter:progid:DXImageTransform.Microsoft.dropshadow(OffX=0,OffY=-1,Color=#e80f2299,Positive=true);-webkit-appearance:none;-moz-appearance:none;appearance:none;overflow:hidden;-webkit-user-select:none;-moz-user-select:none;-ms-user-select:none;user-select:none;-webkit-touch-callout:none;-webkit-highlight:none;-webkit-tap-highlight-color:transparent}[dir] .icl-Button{padding:.5rem 1.125rem;text-align:center;background-repeat:repeat-x;border:1px solid;border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,.2);cursor:pointer}[dir=ltr] .icl-Button{margin:12px 12px 12px 0}[dir=rtl] .icl-Button{margin:12px 0 12px 12px}[dir] .icl-Button::-moz-focus-inner{border:0}[dir] .icl-Button:hover{background-image:none;box-shadow:0 1px 5px rgba(0,0,0,.4)}.icl-Button:active{outline:none}[dir] .icl-Button:active{background-image:none;box-shadow:inset 0 2px 4px rgba(0,0,0,.15),0 1px 2px rgba(0,0,0,.05)}.icl-Button:disabled{opacity:.65}[dir] .icl-Button:disabled{background-image:none;box-shadow:none;cursor:default}[dir] .icl-Button:disabled:hover{box-shadow:none}.icl-Button--primary{color:#f8f8f9;filter:progid:DXImageTransform.Microsoft.gradient(startColorstr="#ff6598fe",endColorstr="#ff3c69e0",GradientType=0)}[dir] .icl-Button--primary{text-shadow:0 -1px #0f2299;background-color:#5585f2;background-image:-webkit-linear-gradient(top,#6598ff,#2e5ad7);background-image:linear-gradient(180deg,#6598ff,#2e5ad7);border-color:#1642bb;border-bottom-color:#1642bb}.icl-Button--primary:hover{text-decoration:none}[dir] .icl-Button--primary:active,[dir] .icl-Button--primary:disabled,[dir] .icl-Button--primary:hover{background-color:#2e5ad7}.icl-Button--secondary{color:#333;filter:progid:DXImageTransform.Microsoft.gradient(startColorstr="#fff8f8f9",endColorstr="#ffe6e6e6",GradientType=0)}[dir] .icl-Button--secondary{text-shadow:0 1px #fff;background-color:#d9d9e2;background-image:-webkit-linear-gradient(top,#f8f8f9,#d9d9e2);background-image:linear-gradient(180deg,#f8f8f9,#d9d9e2);border-color:#9a99ac;border-bottom-color:#a2a2a2}.icl-Button--secondary:hover{text-decoration:none}[dir] .icl-Button--secondary:active,[dir] .icl-Button--secondary:disabled,[dir] .icl-Button--secondary:hover{background-color:#f8f8f9}.icl-Button--special{color:#f8f8f9;filter:progid:DXImageTransform.Microsoft.gradient(startColorstr="#ff6598fe",endColorstr="#ff3c69e0",GradientType=0)}[dir] .icl-Button--special{text-shadow:0 -1px #000;background-color:#f14200;background-image:-webkit-linear-gradient(top,#f60,#f14200);background-image:linear-gradient(180deg,#f60,#f14200);border-color:#ba3200;border-bottom-color:#ba3200}.icl-Button--special:hover{text-decoration:none}[dir] .icl-Button--special:active,[dir] .icl-Button--special:disabled,[dir] .icl-Button--special:hover{background-color:#f14200}.icl-Button--danger{color:#f8f8f9;filter:progid:DXImageTransform.Microsoft.gradient(startColorstr="#ff6598fe",endColorstr="#ff3c69e0",GradientType=0)}[dir] .icl-Button--danger{text-shadow:0 -1px #000;background-color:#b01825;background-image:-webkit-linear-gradient(top,#d1787f,#b01825);background-image:linear-gradient(180deg,#d1787f,#b01825);border-color:#83121b;border-bottom-color:#83121b}.icl-Button--danger:hover{text-decoration:none}[dir] .icl-Button--danger:active,[dir] .icl-Button--danger:disabled,[dir] .icl-Button--danger:hover{background-color:#b01825}[dir] .icl-Button--working{border-color:#9a99ac;border-bottom-color:#a2a2a2}.icl-Button--working:hover{text-decoration:none}[dir] .icl-Button--working:disabled{background-color:#f8f8f9}.icl-Button--transparent{color:#00c}[dir] .icl-Button--transparent{text-shadow:none;background:transparent none;border:none;box-shadow:none}.icl-Button--transparent:hover{color:#00c;text-decoration:underline}[dir] .icl-Button--transparent:active,[dir] .icl-Button--transparent:hover{box-shadow:0 0 0 transparent}.icl-Button--transparent:disabled:hover{color:#00c}.icl-Button--block{display:block;width:100%;max-width:351px}[dir] .icl-Button--block{margin:12px auto}.icl-Button--md{font-size:.9375rem;font-weight:600}[dir] .icl-Button--md{padding:.5rem 1rem;border-radius:5px}.icl-Button--sm{font-size:.8125rem;font-weight:500}[dir] .icl-Button--sm{padding:.325rem .8125rem;border-radius:4px}[dir] .icl-Button--group{border-radius:0;box-shadow:0 0 0 transparent}[dir=ltr] .icl-Button--group{float:left;margin:0 0 0 -1px}[dir=rtl] .icl-Button--group{float:right;margin:0 -1px 0 0}[dir] .icl-Button--group:hover{box-shadow:0 0 0 transparent}[dir=ltr] .icl-Button--group:first-child{margin-left:0;border-bottom-left-radius:6px;border-top-left-radius:6px}[dir=rtl] .icl-Button--group:first-child{margin-right:0}[dir=ltr] .icl-Button--group:last-child,[dir=rtl] .icl-Button--group:first-child{border-bottom-right-radius:6px;border-top-right-radius:6px}[dir=rtl] .icl-Button--group:last-child{border-bottom-left-radius:6px;border-top-left-radius:6px}[dir=ltr] .icl-Button--icon,[dir=rtl] .icl-Button--icon{padding-left:10px;padding-right:10px}
    /*# sourceMappingURL=Button.css.map*/
          </style>
          <div id="resumeCtaFooter" style="height:64px;">
           <div style="position:absolute;width:100%;">
            <style type="text/css">
             .footerCta {
                            text-align:center;
                            margin:0px;
                            font-size:15px;
                            width:100%;
                        }
    
                        .footerCta.blueBar {
                            background-color:#2164f3;
                            color:#ffffff;
                        }
    
                        .footerCta.greyBar {
                            background-color:#ebebeb;
                            color:#000000;
                        }
    
                        div.content >table {
                            margin-bottom: 4em;
                        }
            </style>
            <script type="text/javascript">
             if(null !== call_when_jsall_loaded) {call_when_jsall_loaded(function() {if(!!window.logPromoImpression) {window.logPromoImpression('trk.origin=jobsearch&trk.variant=FooterGrayBelow&trk.pos=below&trk.tk=1c6u39dhta24qbca', 'resume');}})}
            </script>
            <div class="footerCta greyBar">
             Let Employers Find You
             <style type="text/css">
              .cta_button {
                text-decoration:none !important;
                margin: 12px !important;
            }
    
            .cta_button.blue  {
                color: #f8f8f9 !important;
            }
    
            .cta_button.grey {
                color: #000000 !important;
            }
             </style>
             <span dir="ltr">
              <a class="icl-Button icl-Button--primary icl-Button--sm cta_button blue" href="/promo/resume?from=bottomResumeCTAjobsearch&amp;trk.origin=jobsearch" onclick="if(!!window.logPromoClick) {window.logPromoClick('trk.origin=jobsearch&amp;trk.variant=FooterGrayBelow&amp;trk.pos=below&amp;trk.tk=1c6u39dhta24qbca', 'resume','/promo/resume?from=bottomResumeCTAjobsearch&amp;trk.origin=jobsearch');}">
               Upload Your Resume
              </a>
             </span>
            </div>
           </div>
          </div>
         </div>
        </td>
       </tr>
      </table>
      <script type="text/javascript">
       function sm_cv_tag(activityId) {
                var ebRand = Math.random()+'';
                ebRand = ebRand * 1000000;
    
                var tagContainer = document.body.appendChild(document.createElement("div"));
                tagContainer.style.position="absolute";
                tagContainer.style.top="0";
                tagContainer.style.left="0";
                tagContainer.style.width="1px";
                tagContainer.style.height="1px";
                tagContainer.style.display="none";
    
                var jsTag = document.createElement('script');
                jsTag.src = '//bs.serving-sys.com/Serving/ActivityServer.bs?cn=as&ActivityID=' + activityId + '&rnd=' + ebRand;
                jsTag.setAttribute('crossorigin', 'anonymous');
                jsTag.style.width="1px";
                jsTag.style.height="1px";
                jsTag.style.border="0";
                jsTag.async = 1;
    
                var noScript = document.createElement("noscript");
                var noScriptText = '<img width="1" height="1" style="border:0" src="//bs.serving-sys.com/Serving/ActivityServer.bs?cn=as&ActivityID=' + activityId + '&ns=1"/>';
    
                // IE less than 9 and RCs do not support innerHTML on some DOM elements, but supports .text for it
                if (((typeof noScript.canHaveHTML) === "boolean") && (noScript.canHaveHTML === false)) { // canHaveHTML only exists for IE
                    noScript.text = noScriptText;
                } else {
                    noScript.innerHTML = noScriptText;
                }
    
                tagContainer.appendChild(jsTag);
                tagContainer.appendChild(noScript);
            }
      </script>
      <script type="text/javascript">
       <!--
    (function ( tk ) { if ( tk && document.images ) { var s="/", q="?", a="&", e="="; rpc(s+"rpc"+s+"log"+q+"a"+e+"jsv"+a+"tk"+e+tk); } })('1c6u39dhta24qbca');
    function jsall_loaded() {
        
    
        initProcessLeftoverDwellEntries();
        
        detectBrowserState('jobsearch', '1c6u39dhta24qbca');
    
        attachSjBlock('');
        attachJaBlock('');
    }
    if (window['closureReady'] === true) {
        jsall_loaded();
    }
    //-->
      </script>
      <script type="text/javascript">
       PENDING_ANALYTICS_VARS = window.PENDING_ANALYTICS_VARS || [];
    PENDING_ANALYTICS_VARS[PENDING_ANALYTICS_VARS.length] = ['_setCustomVar', 5, 'loggedIn', 'false', 3];
      </script>
      <script type="text/javascript">
       var ga_domains = [];
            ga_domains.push('indeed.co.in');ga_domains.push('indeed.lu');ga_domains.push('indeed.fr');ga_domains.push('indeed.de');ga_domains.push('indeed.com.br');ga_domains.push('indeed.co.uk');ga_domains.push('indeed.hk');ga_domains.push('indeed.fi');ga_domains.push('indeed.pt');ga_domains.push('indeed.jp');ga_domains.push('indeed.com');ga_domains.push('indeed.com.sg');ga_domains.push('indeed.nl');ga_domains.push('indeed.com.pk');ga_domains.push('indeed.cl');ga_domains.push('indeed.es');ga_domains.push('indeed.co.ve');ga_domains.push('indeed.ae');ga_domains.push('indeed.com.mx');ga_domains.push('indeed.com.my');ga_domains.push('indeed.ch');ga_domains.push('indeed.com.co');ga_domains.push('indeed.com.ph');ga_domains.push('indeed.co.za');ga_domains.push('indeed.ie');ga_domains.push('indeed.com.au');ga_domains.push('indeed.ca');ga_domains.push('indeed.com.pe');
    
            (function (i, s, o, g, r, a, m) {
                i['GoogleAnalyticsObject'] = r;
                i[r] = i[r] || function () {
                    (i[r].q = i[r].q || []).push(arguments)
                }, i[r].l = 1 * new Date();
                a = s.createElement(o),
                        m = s.getElementsByTagName(o)[0];
                a.async = 1;
                a.src = g;
                m.parentNode.insertBefore(a, m)
            })(window, document, 'script', '//www.google-analytics.com/analytics.js', 'ga');
    
            var ga = ga || [];
            ga('create', 'UA-90780-1', 'auto', {
                'allowLinker': true
            });
            ga('require', 'linkid');
            ga('require', 'linker');
            ga('linker:autoLink', ga_domains, false, true);
            ga('require', 'displayfeatures');
            ga('send', 'pageview');
    
            
            (function () {
                if (window.PENDING_ANALYTICS_VARS && window.PENDING_ANALYTICS_VARS.length > 0) {
                    for (var i in PENDING_ANALYTICS_VARS) {
                        ga('set', PENDING_ANALYTICS_VARS[i][2], PENDING_ANALYTICS_VARS[i][3]);
                    }
                }
            })();
      </script>
      <script>
       !function(f,b,e,v,n,t,s){if(f.fbq)return;n=f.fbq=function(){n.callMethod?
                n.callMethod.apply(n,arguments):n.queue.push(arguments)};if(!f._fbq)f._fbq=n;
            n.push=n;n.loaded=!0;n.version='2.0';n.queue=[];t=b.createElement(e);t.async=!0;
            t.src=v;s=b.getElementsByTagName(e)[0];s.parentNode.insertBefore(t,s)}(window,
                document,'script','https://connect.facebook.net/en_US/fbevents.js');
    
        fbq('init', '579216298929618');
        fbq('track', "PageView");
      </script>
      <noscript>
       <img height="1" src="https://www.facebook.com/tr?id=579216298929618&amp;ev=PageView&amp;noscript=1" style="display:none" width="1"/>
      </noscript>
      <script>
       var _comscore = _comscore || [];
    _comscore.push({ c1: "2", c2: "6486505", c4:"www.indeed.com/jobs", c15:"1c6u39dg0a24qd40"});
    (function() { var s = document.createElement("script"), el = document.getElementsByTagName("script")[0]; s.async = true; s.src = (document.location.protocol == "https:" ? "https://sb" : "http://b") + ".scorecardresearch.com/beacon.js"; el.parentNode.insertBefore(s, el); })();
      </script>
      <noscript>
       <img alt="" height="0" src="http://b.scorecardresearch.com/p?c1=2&amp;c2=6486505&amp;c4=www.indeed.com%2Fjobs&amp;c15=1c6u39dg0a24qd40&amp;cv=2.0&amp;cj=1" style="display:none" width="0"/>
      </noscript>
     </body>
    </html>
    
    

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
def emailme(from_addr = 'malhotrajat@gmail.com', to_addr = 'rajat.malhotra@utexas.edu', subject = 'Daily Data Science Jobs Update Scraped from Indeed', text = None):
    
    message = 'Subject: {0}\n\nJobs: {1}'.format(subject, text)

    # login information
    username = 'malhotrajat@gmail.com'
    password = 'Funnybones7!'
    
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
    
    
    'Glocomms' , 'Data Scientist' , 'Just posted'
    R count: 0 , Python count: 1 , SQL count: 1 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Wunderman' , 'Financial Analyst - Global Finance' , 'Just posted'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'NYU School of Medicine Pediatrics (S840)' , 'Research Data Associate' , 'Just posted'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'NYU School of Medicine Pediatrics-Bellevue (S623)' , 'Research Data Associate *Must Speak Spanish' , 'Just posted'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Oliver James Associates' , 'Data Scientist - Life Insurance - NYC' , 'Just posted'
    R count: 1 , Python count: 1 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Brookhaven National Laboratory' , 'Beamline Scientist - - Tender-Energy X-ray Absorption Spectroscopy (TES)' , 'Just posted'
    R count: 0 , Python count: 2 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Clin Labs-Outpatient Lab (H188) NYU Langone Hospit...' , 'Clerk' , 'Just posted'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Neurology-Neurogenetic Div (S604) NYU School of Me...' , 'Research Coordinator' , 'Just posted'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Clin Labs-Central Lab Service (H185) NYU Langone H...' , 'Lab Info Systems Coordinator' , 'Just posted'
    R count: 1 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    URL: http://www.indeed.com/jobs?q=data+scientist&l=New+York&sort=date&start=10 
    
    'FIN-SPLY CHN-Central Distribut (H599) NYU Langone...' , 'Distribution Attendant' , 'Just posted'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'NSG-OR TH 6 (H377) NYU Langone Hospitals' , 'Secretary II' , 'Just posted'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Albert Einstein College of Medicine' , 'Research Technician C' , 'Just posted'
    R count: 2 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Bristol-Myers Squibb' , 'Supply Chain Technical Specialist' , 'Just posted'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Open Systems Technologies, Inc.' , 'Senior Java Machine Learning Developer' , 'Just posted'
    R count: 0 , Python count: 1 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Ceros' , 'Data Scientist' , 'Today'
    R count: 0 , Python count: 0 , SQL count: 1 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Techwave Consulting Inc.' , 'Data Scientist - Jr. / Mid Level' , 'Today'
    R count: 0 , Python count: 1 , SQL count: 1 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Bank of America' , 'Investment Manager Research Analyst' , 'Today'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'No Water No Life' , 'Research Assistant Internship' , 'Today'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Rise corp' , 'Analytics Consultant' , 'Today'
    R count: 2 , Python count: 1 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    URL: http://www.indeed.com/jobs?q=data+scientist&l=Chicago&sort=date&start=0 
    
    'CapTech Consulting' , 'Data Engineer' , 'Just posted'
    R count: 0 , Python count: 2 , SQL count: 5 , Hadoop count: 0 , Tableau count: 1 ,
    
    
    'Kraft Heinz Company' , 'Scientist, R&D - Specification' , 'Just posted'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Kraft Heinz Company' , 'Associate Scientist, R&D- Packaging Development' , 'Just posted'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Conversant Media' , 'Data Scientist' , 'Today'
    R count: 1 , Python count: 3 , SQL count: 2 , Hadoop count: 1 , Tableau count: 0 ,
    
    
    'AllianceData' , 'Data Scientist' , 'Today'
    R count: 1 , Python count: 3 , SQL count: 2 , Hadoop count: 1 , Tableau count: 0 ,
    
    
    'Epsilon' , 'Data Scientist' , 'Today'
    R count: 1 , Python count: 3 , SQL count: 2 , Hadoop count: 1 , Tableau count: 0 ,
    
    
    'Groupon' , 'Data Scientist - Optimize' , 'Today'
    R count: 2 , Python count: 2 , SQL count: 2 , Hadoop count: 1 , Tableau count: 0 ,
    
    
    'Workbridge Associates' , 'Data Scientist (Python, R, SQL)' , 'Today'
    R count: 3 , Python count: 4 , SQL count: 4 , Hadoop count: 0 , Tableau count: 1 ,
    
    
    'Nielsen' , 'Sr. Software Engineer (Full Stack)' , 'Today'
    R count: 0 , Python count: 1 , SQL count: 2 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Technology Partners' , 'Software QA/Test-Expert' , 'Today'
    R count: 0 , Python count: 2 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    URL: http://www.indeed.com/jobs?q=data+scientist&l=Chicago&sort=date&start=10 
    
    'Block Six Analytics' , 'Web Designer' , 'Today'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Chicago State University' , 'Student Research Assistant' , 'Today'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'TransTech LLC' , 'Sr. Data Scientist' , 'Today'
    R count: 1 , Python count: 2 , SQL count: 2 , Hadoop count: 1 , Tableau count: 1 ,
    
    
    'The Boston Consulting Group' , 'Senior Systems (DevOps) Engineer' , 'Today'
    R count: 0 , Python count: 1 , SQL count: 2 , Hadoop count: 1 , Tableau count: 0 ,
    
    
    'Blue Cross Blue Shield of IL, MT, NM, OK & TX' , 'Associate Data Scientist' , 'Today'
    R count: 1 , Python count: 1 , SQL count: 1 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Catalina Marketing' , 'Senior Director, Data Science' , 'Today'
    R count: 2 , Python count: 3 , SQL count: 2 , Hadoop count: 2 , Tableau count: 0 ,
    
    
    'The Marketing Store' , 'Lead Data Scientist' , 'Today'
    R count: 2 , Python count: 2 , SQL count: 2 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Bank of America' , 'Sr Quantitative Finance Analyst' , 'Today'
    R count: 0 , Python count: 1 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Honeywell' , 'Sr Research & Development Tech' , 'Today'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Dataiku' , 'Sales Engineer for Data Science- West Coast or Central Region' , '1 day ago'
    R count: 2 , Python count: 1 , SQL count: 0 , Hadoop count: 1 , Tableau count: 0 ,
    
    
    URL: http://www.indeed.com/jobs?q=data+scientist&l=Austin&sort=date&start=0 
    
    'IGT' , 'Market Research Analyst I' , 'Just posted'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 1 ,
    
    
    'Camillogic' , 'Data Scientist - Sr Level - 120k-140k' , 'Today'
    R count: 1 , Python count: 1 , SQL count: 1 , Hadoop count: 1 , Tableau count: 0 ,
    
    
    'COMPTROLLER OF PUBLIC ACCOUNTS' , 'CPA- Data Analyst (Data Analysis & Transparency)' , 'Today'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'National instrument' , 'Software Engineer - Documentation Technologies' , 'Today'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'RetailMeNot, Inc.' , 'Quality Engineer II' , 'Today'
    R count: 0 , Python count: 1 , SQL count: 2 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Msys Technologies' , 'Sr. Data Scientist ( more than 10 + exp)' , 'Today'
    R count: 1 , Python count: 1 , SQL count: 0 , Hadoop count: 1 , Tableau count: 0 ,
    
    
    'National Instruments' , 'Software Engineer - Documentation Technologies' , 'Today'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Department of the Interior' , 'Geographer Recent Graduate, GS-0150-07 (SH-RG)' , 'Today'
    R count: 1 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Walmart' , '2018 Full time: GBS - Sr. Statistical Analyst, DES - Austin, TX' , '1 day ago'
    R count: 0 , Python count: 0 , SQL count: 2 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'University of Texas at Austin' , 'Research Scientist - Sedimentary Petrographer' , '1 day ago'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    URL: http://www.indeed.com/jobs?q=data+scientist&l=Austin&sort=date&start=10 
    
    'KORE1 Technologies' , 'Senior Data Scientist' , '1 day ago'
    R count: 0 , Python count: 1 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Civitas Learning' , 'Product Marketing Manager' , '1 day ago'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'HomeAway' , 'Senior Data Scientist' , '2 days ago'
    R count: 1 , Python count: 1 , SQL count: 1 , Hadoop count: 1 , Tableau count: 0 ,
    
    
    'Senseye' , 'Sr. Research Scientist' , '2 days ago'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Far Harbor, LLC' , 'Public Health Research Statistician' , '2 days ago'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'Senseye' , 'Research Assistant' , '2 days ago'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'State Street' , 'Entry Level - Data Scientist and Cognitive Engineer - Austin' , '3 days ago'
    R count: 0 , Python count: 1 , SQL count: 3 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'State Street' , 'Entry Level - Cognitive Software Engineer - Austin' , '3 days ago'
    R count: 0 , Python count: 1 , SQL count: 2 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    'RetailMeNot, Inc.' , 'Lead Data Scientist' , '4 days ago'
    R count: 0 , Python count: 1 , SQL count: 1 , Hadoop count: 1 , Tableau count: 0 ,
    
    
    'Invenio Marketing Solutions' , 'Inside Sales Representative - Mediacom' , '4 days ago'
    R count: 0 , Python count: 0 , SQL count: 0 , Hadoop count: 0 , Tableau count: 0 ,
    
    
    Please check your mail
    

# **Ending Remarks**

This was a pretty interesting project to complete and a lot of fun too. I am sure there are many improvements that can be made and it can give more information too. More changes may be made in future.
