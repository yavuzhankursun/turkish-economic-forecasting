import requests
from datetime import datetime
from config.config import NEWSAPI_CONFIG

class NewsAPIClient:
    def __init__(self):
        self.api_key = NEWSAPI_CONFIG["api_key"]
        self.base_url = NEWSAPI_CONFIG["base_url"]
        self.language = NEWSAPI_CONFIG["default_language"]
        self.page_size = NEWSAPI_CONFIG["page_size"]

    def fetch_news(self, query, from_date=None, to_date=None, max_articles=100):
        """
        NewsAPI üzerinden haberleri çeker.
        :param query: Aranacak anahtar kelime veya kelimeler
        :param from_date: Başlangıç tarihi (YYYY-MM-DD)
        :param to_date: Bitiş tarihi (YYYY-MM-DD)
        :param max_articles: Maksimum çekilecek haber sayısı
        :return: Haberlerin listesi (dict)
        """
        articles = []
        page = 1
        total_fetched = 0
        while total_fetched < max_articles:
            params = {
                "q": query,
                "language": self.language,
                "apiKey": self.api_key,
                "pageSize": min(self.page_size, max_articles - total_fetched),
                "page": page
            }
            if from_date:
                params["from"] = from_date
            if to_date:
                params["to"] = to_date
            response = requests.get(self.base_url + "everything", params=params)
            if response.status_code != 200:
                raise Exception(f"NewsAPI hatası: {response.status_code} - {response.text}")
            data = response.json()
            fetched = data.get("articles", [])
            if not fetched:
                break
            for article in fetched:
                articles.append({
                    "title": article.get("title"),
                    "content": article.get("content"),
                    "source": article.get("source", {}).get("name"),
                    "publishedAt": article.get("publishedAt"),
                    "url": article.get("url")
                })
            total_fetched += len(fetched)
            if len(fetched) < params["pageSize"]:
                break
            page += 1
        return articles 