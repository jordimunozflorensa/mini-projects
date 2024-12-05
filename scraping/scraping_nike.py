import requests
from bs4 import BeautifulSoup

url = "https://www.nike.com/es/w/hombre-zapatillas-nik1zy7ok"

response = requests.get(url)
response.raise_for_status()

soup = BeautifulSoup(response.content, 'html.parser')
product_titles = soup.find_all('div', class_='product-card__title')

for title in product_titles:
    print(title.get_text())

print(len(product_titles))
