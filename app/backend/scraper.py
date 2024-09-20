import requests
from bs4 import BeautifulSoup
import sqlite3

def scrape_isetcom_website():
    # URL of the ISETCOM website
    url = "http://www.isetcom.tn/"  # Replace with the actual URL
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # This is a simplified example. You'll need to adjust the selectors
    # based on the actual structure of the ISETCOM website.
    sections = soup.find_all('div', class_='content-section')
    
    data = []
    for section in sections:
        title = section.find('h2').text.strip()
        content = section.find('p').text.strip()
        data.append((title, content))
    
    return data

def save_to_database(data):
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    
    # Assuming you want to save this data in the 'faq' table
    cursor.executemany("INSERT INTO faq (question, answer) VALUES (?, ?)", data)
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    scraped_data = scrape_isetcom_website()
    save_to_database(scraped_data)
    print(f"Scraped and saved {len(scraped_data)} items.")