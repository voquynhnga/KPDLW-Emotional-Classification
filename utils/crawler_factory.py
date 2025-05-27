from crawler.DMXCrawler import DMXCrawler
from crawler.TIKICrawler import TIKICrawler

class CrawlerFactory:
    @staticmethod
    def get_crawler(url: str):
        if url.startswith('https://www.dienmayxanh.com/'):
            return DMXCrawler()
        elif url.startswith('https://tiki.vn/'):
            return TIKICrawler()
        else:
            raise ValueError("URL không được hỗ trợ")