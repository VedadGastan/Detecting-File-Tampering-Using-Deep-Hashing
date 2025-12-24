from scraper import ImageScraper
from tampering import ImageTamperer

def build():
    scraper = ImageScraper()
    scraper.download(total=3000)

    tamperer = ImageTamperer()
    tamperer.create(per_image=4)

if __name__ == "__main__":
    build()
