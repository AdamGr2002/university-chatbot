import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from bs4 import BeautifulSoup
import sqlite3
from io import StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import ssl
import certifi
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create a custom SSL context
context = ssl.create_default_context(cafile=certifi.where())

Base = declarative_base()

class FAQ(Base):
    __tablename__ = 'faq'

    id = Column(Integer, primary_key=True)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)

# Suppress only the single warning from urllib3 needed.
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def scrape_isetcom_website():
    # URL of the ISETCOM website
    url = "https://www.isetcom.tn/"  # Make sure to use https://
    
    try:
        # Use verify=False to bypass SSL verification
        response = requests.get(url, verify=False, timeout=10)
        print(f"Status code: {response.status_code}")
        print(f"Content: {response.text[:100]}...")  # Print first 100 characters
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # This is a simplified example. You'll need to adjust the selectors
    # based on the actual structure of the ISETCOM website.
    sections = soup.find_all('div', class_='content-section')
    
    data = []
    for section in sections:
        title = section.find('h2').text.strip() if section.find('h2') else ""
        content = section.find('p').text.strip() if section.find('p') else ""
        if title and content:
            data.append((title, content))
    
    return data

def save_to_database(data):
    engine = create_engine('sqlite:///chatbot.db')  # Changed from faq.db to chatbot.db
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Clear existing data
        session.query(FAQ).delete()
        
        for question, answer in data:
            faq_item = FAQ(question=question, answer=answer)
            session.add(faq_item)
        session.commit()
        print(f"Saved {len(data)} items to the database.")
    except Exception as e:
        print(f"An error occurred: {e}")
        session.rollback()
    finally:
        session.close()

def scrape_pdf(pdf_file_name):
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the full path to the PDF file
    pdf_file_path = os.path.join(current_dir, pdf_file_name)
    
    if not os.path.exists(pdf_file_path):
        print(f"File not found: {pdf_file_path}")
        return None

    with open(pdf_file_path, 'rb') as file:
        output_string = StringIO()
        extract_text_to_fp(file, output_string, laparams=LAParams())
        text = output_string.getvalue()
        
        data = []
        current_question = ""
        current_answer = ""
        for line in text.split('\n'):
            line = line.strip()
            if line.isupper() and len(line) > 10:  # Potential question
                if current_question and current_answer:
                    data.append((current_question, current_answer))
                current_question = line
                current_answer = ""
            elif current_question:
                current_answer += line + " "
        
        # Add the last Q&A pair if exists
        if current_question and current_answer:
            data.append((current_question, current_answer))

        # Filter out any Q&A pairs that are too short
        data = [(q, a) for q, a in data if len(q) > 10 and len(a) > 20]
    
    return data

def save_to_database(data):
    engine = create_engine('sqlite:///chatbot.db')
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Clear existing data
        session.query(FAQ).delete()
        
        for question, answer in data:
            faq_item = FAQ(question=question, answer=answer)
            session.add(faq_item)
        session.commit()
        print(f"Saved {len(data)} items to the database.")
    except Exception as e:
        print(f"An error occurred: {e}")
        session.rollback()
    finally:
        session.close()

def scrape_pdf_from_url(pdf_url):
    response = requests.get(pdf_url)
    pdf_file = StringIO(response.content.decode('latin-1'))
    
    output_string = StringIO()
    extract_text_to_fp(pdf_file, output_string, laparams=LAParams())
    text = output_string.getvalue()
    
    # Implement parsing logic here
    data = parse_pdf_text(text)
    
    return data

def parse_pdf_text(text):
    # Implement your parsing logic here
    # This is a placeholder implementation
    data = []
    paragraphs = text.split('\n\n')
    for i in range(0, len(paragraphs), 2):
        if i + 1 < len(paragraphs):
            title = paragraphs[i].strip()
            content = paragraphs[i+1].strip()
            data.append((title, content))
    return data

if __name__ == "__main__":
    scraped_data = scrape_isetcom_website()
    
    # Example usage for local PDF
    pdf_data = scrape_pdf('Livret 2024-2025.pdf')
    scraped_data.extend(pdf_data)
    

    # Example usage for PDF from URL
    # pdf_url = "http://example.com/sample.pdf"
    # pdf_data = scrape_pdf_from_url(pdf_url)
    # scraped_data.extend(pdf_data)
    
    save_to_database(scraped_data)
    print(f"Scraped and saved {len(scraped_data)} items.")