import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from config import TIME_OUT


class Crawler:

    def crawl_website(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=TIME_OUT, headers={
                                    'User-Agent': 'rag-chatbot/5.0'})
            response.raise_for_status()
            return response.text
        except Exception:
            return ""

    def extract_links(self, base_url, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        links = set()
        for link in soup.find_all('a', href=True):
            absolute_link = urljoin(base_url, link['href'])
            parsed_link = urlparse(absolute_link)
            if parsed_link.scheme in ['http', 'https']:
                links.add(absolute_link)
        return links

    def cleaner(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")

        for tag in soup(["nav", "footer", "header", "script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        return " ".join(text.split())

    def crawl(self, url, max_pages=10):
        visited = set()
        to_visit = [url]
        all_content = []

        while to_visit and len(visited) < max_pages:
            current_url = to_visit.pop(0)
            if current_url in visited:
                continue

            print(f"Crawling page {len(visited)+1}: {current_url}")
            content = self.crawl_website(url=current_url)

            if content:

                visited.add(current_url)

                hyperlinks = self.extract_links(
                    base_url=current_url, html_content=content)
                content = self.cleaner(content)
                all_content.append(content)
                for link in hyperlinks:
                    if link not in visited:
                        to_visit.append(link)

        return all_content
