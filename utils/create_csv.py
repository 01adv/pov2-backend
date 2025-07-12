import json
import csv
from bs4 import BeautifulSoup

def create_csv_from_json(json_file_path, csv_file_path):
    """
    Parses a JSON file containing product data and creates a CSV file
    with selected fields.

    Args:
        json_file_path (str): The path to the input JSON file.
        csv_file_path (str): The path to the output CSV file.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        products = json.load(f)

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['title', 'handle', 'body_html', 'product_type', 'sizes', 'price', 'colors'])

        for product in products:
            title = product.get('title')
            handle = product.get('handle')
            body_html = product.get('body_html')
            if body_html:
                soup = BeautifulSoup(body_html, 'html.parser')
                body_html = soup.get_text().strip()

            product_type = product.get('product_type')

            sizes = []
            colors = []
            for option in product.get('options', []):
                if option.get('name') == 'Size':
                    sizes = option.get('values', [])
                if option.get('name') == 'Color':
                    colors = option.get('values', [])

            price = None
            if product.get('variants'):
                price = product['variants'][0].get('price')

            writer.writerow([
                title,
                handle,
                body_html,
                product_type,
                ', '.join(sizes),
                price,
                ', '.join(colors)
            ])

if __name__ == '__main__':
    create_csv_from_json('27jun_all_products.json', 'products.csv')
