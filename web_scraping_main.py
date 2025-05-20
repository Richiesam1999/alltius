#v2

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from collections import OrderedDict
import re

# Base URL and user agent to mimic a real browser
BASE_URL = "https://www.angelone.in/support/margin-pledging-and-margin-trading-facility/margin-pledge-unpledge"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# List of link texts to follow
LINK_TEXTS = [
    "Add and Withdraw Funds",
    "Angel One Recommendations",
    "Charges and Cashbacks",
    "Charts",
    "Compliance",
    "Fixed Deposits",
    "IPO & OFS",
    "Loans",
    "Margin Pledging and Margin Trading Facility",
    "Mutual Funds",
    "Portfolio and Corporate Actions",
    "Reports and Statements",
    "Your Account",
    "Your Orders"
]

def extract_unique_text(soup, stop_line):
    """Extract unique text content up to the stop line"""
    seen_text = OrderedDict()  # Preserves insertion order
    elements_to_ignore = ['script', 'style', 'noscript', 'meta', 'link', 'svg', 'img']
    
    # Find the main content element - different sites might need different selectors
    main_content = soup.find('main') or soup.find('div', class_='content') or soup
    
    # Pattern to identify stock price listings
    stock_pattern = re.compile(r'.*Share Price$')
    
    # Lists to track text for deduplication
    content_sections = []
    
    # Iterate through appropriate tags that typically contain content
    for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div']):
        # Skip unwanted elements
        if element.name in elements_to_ignore:
            continue
        
        # Check parent to avoid deeply nested content in navigation, etc.
        parent_classes = element.parent.get('class', [])
        if any(cls in ['navbar', 'nav', 'footer', 'sidebar', 'menu'] for cls in parent_classes):
            continue
            
        # Stop when we reach the termination line
        if stop_line in element.get_text():
            break
        
        # Get clean text
        text = element.get_text(' ', strip=True)
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Skip empty or very short text
        if not text or len(text) < 3:
            continue
        
        # Skip common repeating elements and stock prices
        if any(phrase in text for phrase in ["Please Wait", "Popular Stocks"]) or stock_pattern.match(text):
            continue
        
        # Add to seen texts if not already present
        if text not in seen_text:
            seen_text[text] = True
            content_sections.append(text)
    
    return '\n'.join(content_sections)

def get_page_content(url):
    """Fetch and parse a webpage"""
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract content up to the stop line
        stop_line = "Want to connect with us? Our experts will be happy to assist you"
        content = extract_unique_text(soup, stop_line)
        
        return content
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def find_links_by_text(soup, link_texts):
    """Find all links that match the given link texts"""
    links = []
    for a in soup.find_all('a', href=True):
        link_text = a.get_text(strip=True)
        if any(text.lower() in link_text.lower() for text in link_texts) and a['href'] not in links:
            links.append(a['href'])
    return links

def crawl_site():
    # Start with the base URL
    base_content = get_page_content(BASE_URL)
    if not base_content:
        print("Failed to fetch base URL")
        return
    
    # Save base content
    with open('angelone_base_clean.txt', 'w', encoding='utf-8') as f:
        f.write(f"=== BASE URL: {BASE_URL} ===\n\n")
        f.write(base_content)
    print(f"Successfully extracted base URL content")
    
    # Parse the base page to find all relevant links
    response = requests.get(BASE_URL, headers=HEADERS)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all links that match our target texts
    links = find_links_by_text(soup, LINK_TEXTS)
    
    # Make sure links are absolute and unique
    links = list(set(urljoin(BASE_URL, link) for link in links))
    
    print(f"Found {len(links)} relevant links to crawl")
    
    # Crawl each linked page
    for i, link in enumerate(links, 1):
        print(f"Crawling link {i}/{len(links)}: {link}")
        content = get_page_content(link)
        if content:
            # Save content to file
            filename = f'angelone_link_{i}_clean.txt'
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"=== LINK: {link} ===\n\n")
                f.write(content)
            print(f"  Saved clean content to {filename}")
        
        # Be polite with a delay between requests
        time.sleep(2)

if __name__ == "__main__":
    crawl_site()
    print("Crawling completed!")