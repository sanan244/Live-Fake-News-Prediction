import httplib2
import urllib.request
from bs4 import BeautifulSoup, SoupStrainer
from bs4.element import Comment
from requests_html import HTMLSession

base_url = 'https://transcripts.cnn.com/'

def get_links(source):
    sss = HTMLSession()
    k = sss.get(source)
    links = k.html.absolute_links
    print(links)
    return links

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(string=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)

print("Getting links...")
#get links
links = get_links(base_url)

print("Scraping links...")
#traverse and scrape each link
for link in links:
    html = urllib.request.urlopen(link).read()
    print(text_from_html(html))

#parse text and create dataframe
print("Parsing text and creating dataframe...")