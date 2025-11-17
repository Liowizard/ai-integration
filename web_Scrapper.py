from bs4 import BeautifulSoup as soup
import requests
import pandas as pd

def get_chennai_news():
    # ---------------------------- THE HINDU ----------------------------
    r = requests.get("https://www.thehindu.com/news/cities/chennai/")
    b = soup(r.content, "html.parser")

    hindu_links = []
    for i in b.find_all('h3', {"class": "title"}):
        a_tag = i.find("a")
        if a_tag and a_tag.get("href"):
            hindu_links.append(a_tag["href"])

    TH = {"title": [], "news": []}

    for url in hindu_links:
        req = requests.get(url)
        blog = soup(req.content, "html.parser")

        h1 = blog.find("h1")
        TH["title"].append(h1.text.strip() if h1 else "No Title Found")

        paragraphs = blog.find_all("p")
        news_text = "\n".join([p.text.strip() for p in paragraphs if p.text.strip()])
        TH["news"].append(news_text)

    hindu_news = pd.DataFrame(TH)

    # ---------------------------- INDIA TODAY ----------------------------
    r = requests.get("https://www.indiatoday.in/cities/chennai-news")
    b = soup(r.content, "html.parser")

    india_today_links = []
    for i in b.find_all('div', {"class": "B1S3_content__wrap__9mSB6"}):
        a_tag = i.find("a")
        if a_tag and a_tag.get("href"):
            india_today_links.append("https://www.indiatoday.in" + a_tag["href"])

    IT = {"title": [], "news": []}

    for url in india_today_links:
        req = requests.get(url)
        blog = soup(req.content, "html.parser")

        h1 = blog.find("h1")
        IT["title"].append(h1.text.strip() if h1 else "No Title Found")

        paragraphs = blog.find_all("p")
        news_text = "\n".join([p.text.strip() for p in paragraphs if p.text.strip()])
        IT["news"].append(news_text)

    india_today_news = pd.DataFrame(IT)


    # ---------------------------- news18 (with pagination) ----------------------------
    news18_links = []

    BASE_URL = "https://www.news18.com/cities/chennai-news/"

    # Loop through page-1 to page-5
    for page in range(1, 6):
        if page == 1:
            url = BASE_URL
        else:
            url = f"{BASE_URL}page-{page}/"


        r = requests.get(url)
        b = soup(r.content, "html.parser")

        for i in b.find_all('li', {"class": "jsx-2985166500"}):
            a_tag = i.find("a")
            if a_tag and a_tag.get("href"):
                news18_links.append(a_tag["href"])


    # ---------------------------- Fetch full news ----------------------------
    n18 = {"title": [], "news": []}

    for url in news18_links:
        full_url = "https://www.news18.com" + url

        req = requests.get(full_url)
        blog = soup(req.content, "html.parser")

        # Title
        h1 = blog.find("h1")
        n18["title"].append(h1.text.strip() if h1 else "No Title Found")

        # Body text
        paragraphs = blog.find_all("p")
        news_text = "\n".join([p.text.strip() for p in paragraphs if p.text.strip()])
        n18["news"].append(news_text)

    news18_news = pd.DataFrame(n18)


    # ---------------------------- MERGE BOTH ----------------------------
    combined_news = pd.concat([hindu_news, india_today_news,news18_news], ignore_index=True)

    return combined_news
