import re
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
from fuzzywuzzy import fuzz

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
    url = "https://www.isetcom.tn/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=30)
        response.raise_for_status()
        print(f"Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    data = []
    
    # Scrape main sections
    main_sections = soup.find_all('div', class_='elementor-widget-container')
    for section in main_sections:
        title = section.find('h2', class_='elementor-heading-title')
        content = section.find('div', class_='elementor-text-editor')
        if title and content:
            data.append((title.text.strip(), content.text.strip()))
    
    # Scrape news items
    news_items = soup.find_all('article', class_='elementor-post')
    for item in news_items:
        title = item.find('h3', class_='elementor-post__title')
        excerpt = item.find('div', class_='elementor-post__excerpt')
        if title and excerpt:
            data.append((title.text.strip(), excerpt.text.strip()))
    
    # Scrape contact information
    contact_info = soup.find('div', class_='elementor-widget-wrap', string=re.compile('Contact'))
    if contact_info:
        contact_details = contact_info.find_all('div', class_='elementor-text-editor')
        for detail in contact_details:
            data.append(("Contact Information", detail.text.strip()))
    
    # Scrape footer information
    footer = soup.find('footer', class_='site-footer')
    if footer:
        footer_sections = footer.find_all('div', class_='elementor-widget-wrap')
        for section in footer_sections:
            title = section.find('h2', class_='elementor-heading-title')
            content = section.find('div', class_='elementor-text-editor')
            if title and content:
                data.append((title.text.strip(), content.text.strip()))
    
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
            print(f"Saving to DB: {question[:30]}... - {answer[:30]}...")
        session.commit()
        print(f"Saved {len(data)} items to the database.")
    except Exception as e:
        print(f"An error occurred: {e}")
        session.rollback()
    finally:
        session.close()

def scrape_pdf(pdf_file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_file_path = os.path.join(current_dir, pdf_file_name)
    
    if not os.path.exists(pdf_file_path):
        print(f"File not found: {pdf_file_path}")
        return None

    with open(pdf_file_path, 'rb') as file:
        output_string = StringIO()
        extract_text_to_fp(file, output_string, laparams=LAParams())
        text = output_string.getvalue()
        
        data = []
        lines = text.split('\n')
        current_question = ""
        current_answer = ""
        
        for line in lines:
            line = line.strip()
            if line.isupper() and len(line) > 10:  # Potential question
                if current_question and current_answer:
                    data.append((current_question, current_answer.strip()))
                    print(f"Found Q&A: {current_question[:30]}... - {current_answer[:30]}...")
                current_question = line
                current_answer = ""
            elif current_question:
                if line:  # Only add non-empty lines to the answer
                    current_answer += line + " "
        
        # Add the last Q&A pair if exists
        if current_question and current_answer:
            data.append((current_question, current_answer.strip()))
            print(f"Found Q&A: {current_question[:30]}... - {current_answer[:30]}...")

        # Filter out any Q&A pairs that are too short or duplicates
        seen = set()
        filtered_data = []
        for q, a in data:
            if len(q) > 10 and len(a) > 20 and q not in seen:
                filtered_data.append((q, a))
                seen.add(q)
    
    print(f"Total Q&A pairs found: {len(filtered_data)}")
    return filtered_data

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

def get_best_match(user_question, session):
    all_faqs = session.query(FAQ).all()
    matches = []

    for faq in all_faqs:
        ratio = fuzz.token_set_ratio(user_question.lower(), faq.question.lower())
        if ratio > 60:  # Adjust this threshold as needed
            matches.append((faq, ratio))

    # Sort matches by ratio in descending order
    matches.sort(key=lambda x: x[1], reverse=True)

    # Return the top 3 matches
    return matches[:3]

def get_answer(user_question):
    engine = create_engine('sqlite:///chatbot.db')
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        best_matches = get_best_match(user_question, session)
        if best_matches:
            # Return the best match and its similarity score
            return best_matches[0][0].answer, best_matches[0][1]
        else:
            return "I'm sorry, I couldn't find a relevant answer to your question.", 0
    except Exception as e:
        print(f"An error occurred while retrieving the answer: {e}")
        return "I'm sorry, an error occurred while processing your question.", 0
    finally:
        session.close()

# Modify the main block at the end of the file
if __name__ == "__main__":
    scraped_data = scrape_isetcom_website()
    
    # Example usage for local PDF
    pdf_data = scrape_pdf('Livret 2024-2025.pdf')
    scraped_data.extend(pdf_data)
    
    save_to_database(scraped_data)
    print(f"Scraped and saved {len(scraped_data)} items.")

    # Test the fuzzy matching with scraped questions
    engine = create_engine('sqlite:///chatbot.db')
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Get all questions from the database
        all_questions = session.query(FAQ.question).all()
        
        # Select a subset of questions for testing (e.g., every 10th question)
        test_questions = [q[0] for q in all_questions[::10]]
        
        print("\nTesting fuzzy matching with scraped questions:")
        for question in test_questions:
            print(f"\nOriginal Question: {question}")
            # Simulate a user input by slightly modifying the question
            user_question = question.replace("?", "").lower()
            print(f"Simulated User Question: {user_question}")
            answer, similarity = get_answer(user_question)
            print(f"Answer (Similarity: {similarity}): {answer[:100]}...")  # Print first 100 characters of the answer

    except Exception as e:
        print(f"An error occurred during testing: {e}")
    finally:
        session.close()