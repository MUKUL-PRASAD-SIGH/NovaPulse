"""Web Scraper Tool - Extract full article content from URLs."""
import asyncio
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup
from app.memory.store import log_tool_execution


class WebScraper:
    """Advanced web scraping with content extraction and cleaning."""
    
    def __init__(self):
        self.timeout = 10
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    async def scrape_article(self, url: str) -> Dict:
        """
        Scrape article content from URL.
        
        Args:
            url: Article URL to scrape
            
        Returns:
            Dict with title, content, author, publish_date, images
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers, verify=False, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract metadata
                title = self._extract_title(soup)
                content = self._extract_content(soup)
                author = self._extract_author(soup)
                publish_date = self._extract_date(soup)
                images = self._extract_images(soup, url)
                
                # Clean content
                content = self._clean_text(content)
                
                return {
                    "success": True,
                    "url": url,
                    "title": title,
                    "content": content,
                    "author": author,
                    "publish_date": publish_date,
                    "images": images,
                    "word_count": len(content.split()),
                    "reading_time_minutes": len(content.split()) // 200
                }
                
        except Exception as e:
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title."""
        # Try multiple selectors
        selectors = [
            ('meta', {'property': 'og:title'}),
            ('meta', {'name': 'twitter:title'}),
            ('h1', {}),
            ('title', {})
        ]
        
        for tag, attrs in selectors:
            element = soup.find(tag, attrs)
            if element:
                return element.get('content', element.get_text()).strip()
        
        return "Unknown Title"
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main article content."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
            element.decompose()
        
        # Try article-specific selectors
        content_selectors = [
            'article',
            '[class*="article-content"]',
            '[class*="post-content"]',
            '[class*="entry-content"]',
            'main'
        ]
        
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                # Extract paragraphs
                paragraphs = content.find_all('p')
                if paragraphs:
                    return '\n\n'.join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
        
        # Fallback: get all paragraphs
        paragraphs = soup.find_all('p')
        return '\n\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50)
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article author."""
        selectors = [
            ('meta', {'name': 'author'}),
            ('meta', {'property': 'article:author'}),
            ('[class*="author"]', {}),
            ('[rel="author"]', {})
        ]
        
        for tag, attrs in selectors:
            element = soup.find(tag, attrs)
            if element:
                return element.get('content', element.get_text()).strip()
        
        return None
    
    def _extract_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract publish date."""
        selectors = [
            ('meta', {'property': 'article:published_time'}),
            ('meta', {'name': 'publish_date'}),
            ('time', {'datetime': True})
        ]
        
        for tag, attrs in selectors:
            element = soup.find(tag, attrs)
            if element:
                return element.get('content') or element.get('datetime') or element.get_text()
        
        return None
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract article images."""
        images = []
        
        # Try Open Graph image
        og_image = soup.find('meta', {'property': 'og:image'})
        if og_image and og_image.get('content'):
            images.append(og_image['content'])
        
        # Try article images
        article = soup.find('article') or soup.find('main')
        if article:
            for img in article.find_all('img', src=True)[:5]:  # Limit to 5 images
                src = img['src']
                if src.startswith('http'):
                    images.append(src)
                elif src.startswith('/'):
                    domain = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"
                    images.append(f"{domain}{src}")
        
        return images
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()


async def scrape_url(url: str) -> Dict:
    """
    Scrape content from a single URL.
    
    Args:
        url: URL to scrape
        
    Returns:
        Scraped article data
    """
    scraper = WebScraper()
    result = await scraper.scrape_article(url)
    
    await log_tool_execution(
        tool_name="web_scraper",
        params={"url": url},
        result={"success": result.get("success"), "word_count": result.get("word_count", 0)}
    )
    
    return result


async def scrape_multiple_urls(urls: List[str]) -> List[Dict]:
    """
    Scrape multiple URLs in parallel.
    
    Args:
        urls: List of URLs to scrape
        
    Returns:
        List of scraped article data
    """
    scraper = WebScraper()
    tasks = [scraper.scrape_article(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "success": False,
                "url": urls[i],
                "error": str(result)
            })
        else:
            processed_results.append(result)
    
    await log_tool_execution(
        tool_name="web_scraper",
        params={"urls": urls, "count": len(urls)},
        result={"total": len(processed_results), "successful": sum(1 for r in processed_results if r.get("success"))}
    )
    
    return processed_results
