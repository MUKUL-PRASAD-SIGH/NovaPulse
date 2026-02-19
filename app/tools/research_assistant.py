"""Research Assistant Tool - Search academic papers, patents, and technical docs."""
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from urllib.parse import quote_plus
import httpx
from app.memory.store import log_tool_execution


class ResearchAssistant:
    """Search and analyze academic and technical content."""
    
    def __init__(self):
        self.timeout = 15
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    async def search_papers(self, query: str, limit: int = 10) -> Dict:
        """
        Search academic papers using arXiv API.
        
        Args:
            query: Search query
            limit: Number of results (max 50)
            
        Returns:
            Academic paper search results
        """
        try:
            # arXiv API — URL-encode the query for multi-word topics
            encoded_query = quote_plus(query)
            arxiv_url = f"https://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={min(limit, 50)}"
            
            async with httpx.AsyncClient(timeout=self.timeout, verify=False, follow_redirects=True) as client:
                response = await client.get(arxiv_url)
                response.raise_for_status()
            
            # Parse arXiv XML response
            papers = self._parse_arxiv_response(response.text)
            
            return {
                "success": True,
                "source": "arXiv",
                "query": query,
                "count": len(papers),
                "papers": papers
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def search_patents(self, query: str, limit: int = 10) -> Dict:
        """
        Search patents using Google Patents (public data).
        
        Args:
            query: Search query
            limit: Number of results
            
        Returns:
            Patent search results
        """
        # Note: This is a placeholder. In production, integrate with:
        # - USPTO API (https://developer.uspto.gov/)
        # - Google Patents Public Data
        # - EPO Open Patent Services
        
        return {
            "success": True,
            "source": "patents",
            "query": query,
            "count": 0,
            "patents": [],
            "note": "Patent search requires USPTO API or Google Patents integration"
        }
    
    async def search_technical_docs(self, query: str, sources: Optional[List[str]] = None) -> Dict:
        """
        Search technical documentation.
        
        Args:
            query: Search query
            sources: Specific sources to search ['github', 'stackoverflow', 'docs']
            
        Returns:
            Technical documentation results
        """
        if sources is None:
            sources = ['github', 'stackoverflow']
        
        results = {}
        
        # Search each source
        tasks = []
        if 'github' in sources:
            tasks.append(self._search_github(query))
        if 'stackoverflow' in sources:
            tasks.append(self._search_stackoverflow(query))
        
        source_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results — flatten so github/stackoverflow are top-level keys
        for i, source in enumerate(sources):
            if i < len(source_results) and not isinstance(source_results[i], Exception):
                results[source] = source_results[i]
        
        return {
            "success": True,
            "query": query,
            **results,  # github and stackoverflow as top-level keys
            "total_results": sum(r.get('count', 0) for r in results.values())
        }
    
    def _parse_arxiv_response(self, xml_text: str) -> List[Dict]:
        """Parse arXiv XML response."""
        import xml.etree.ElementTree as ET
        
        papers = []
        
        try:
            root = ET.fromstring(xml_text)
            
            # arXiv namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                paper = {
                    'title': self._get_text(entry, 'atom:title', ns),
                    'summary': self._get_text(entry, 'atom:summary', ns)[:500],  # Truncate
                    'authors': [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
                    'published': self._get_text(entry, 'atom:published', ns),
                    'updated': self._get_text(entry, 'atom:updated', ns),
                    'pdf_url': None,
                    'arxiv_url': None
                }
                
                # Get links
                for link in entry.findall('atom:link', ns):
                    if link.get('title') == 'pdf':
                        paper['pdf_url'] = link.get('href')
                    elif link.get('type') == 'text/html':
                        paper['arxiv_url'] = link.get('href')
                
                # Get categories
                categories = [cat.get('term') for cat in entry.findall('atom:category', ns)]
                paper['categories'] = categories
                
                papers.append(paper)
        
        except Exception:
            pass
        
        return papers
    
    def _get_text(self, element, path: str, namespace: Dict) -> str:
        """Safely get text from XML element."""
        try:
            found = element.find(path, namespace)
            return found.text.strip() if found is not None and found.text else ""
        except Exception:
            return ""
    
    async def _search_github(self, query: str) -> Dict:
        """
        Search GitHub repositories.
        Uses GitHub's public search API (rate limited without auth).
        """
        try:
            search_url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page=10"
            
            async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers, verify=False) as client:
                response = await client.get(search_url)
                response.raise_for_status()
                data = response.json()
            
            repos = []
            for item in data.get('items', []):
                repos.append({
                    'name': item.get('full_name'),
                    'description': item.get('description', ''),
                    'stars': item.get('stargazers_count', 0),
                    'language': item.get('language', 'Unknown'),
                    'url': item.get('html_url'),
                    'updated': item.get('updated_at')
                })
            
            return {
                "success": True,
                "count": len(repos),
                "repositories": repos
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _search_stackoverflow(self, query: str) -> Dict:
        """
        Search StackOverflow questions.
        Uses StackExchange API (no auth required for basic search).
        """
        try:
            search_url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&q={query}&site=stackoverflow"
            
            async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
                response = await client.get(search_url)
                response.raise_for_status()
                data = response.json()
            
            questions = []
            for item in data.get('items', [])[:10]:
                questions.append({
                    'title': item.get('title'),
                    'score': item.get('score', 0),
                    'answer_count': item.get('answer_count', 0),
                    'is_answered': item.get('is_answered', False),
                    'link': item.get('link'),
                    'tags': item.get('tags', [])
                })
            
            return {
                "success": True,
                "count": len(questions),
                "questions": questions
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def comprehensive_research(self, topic: str) -> Dict:
        """
        Perform comprehensive research across all sources.
        
        Args:
            topic: Research topic
            
        Returns:
            Aggregated research results
        """
        # Search all sources in parallel
        papers_task = self.search_papers(topic, limit=5)
        patents_task = self.search_patents(topic, limit=5)
        tech_docs_task = self.search_technical_docs(topic)
        
        papers, patents, tech_docs = await asyncio.gather(
            papers_task, patents_task, tech_docs_task,
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(papers, Exception):
            papers = {"success": False, "error": str(papers)}
        if isinstance(patents, Exception):
            patents = {"success": False, "error": str(patents)}
        if isinstance(tech_docs, Exception):
            tech_docs = {"success": False, "error": str(tech_docs)}
        
        # Aggregate insights
        total_sources = sum([
            papers.get('count', 0),
            patents.get('count', 0),
            tech_docs.get('total_results', 0)
        ])
        
        return {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "academic_papers": papers,
            "patents": patents,
            "technical_docs": tech_docs,
            "summary": {
                "total_sources_found": total_sources,
                "papers_found": papers.get('count', 0),
                "patents_found": patents.get('count', 0),
                "tech_docs_found": tech_docs.get('total_results', 0)
            }
        }


async def search_academic_papers(query: str, limit: int = 10) -> Dict:
    """
    Search academic papers.
    
    Args:
        query: Search query
        limit: Number of results
        
    Returns:
        Academic paper results
    """
    assistant = ResearchAssistant()
    result = await assistant.search_papers(query, limit)
    
    await log_tool_execution(
        tool_name="research_assistant",
        params={"query": query, "type": "papers"},
        result={"count": result.get('count', 0)}
    )
    
    return result


async def comprehensive_research(topic: str) -> Dict:
    """
    Perform comprehensive research.
    
    Args:
        topic: Research topic
        
    Returns:
        Comprehensive research results
    """
    assistant = ResearchAssistant()
    result = await assistant.comprehensive_research(topic)
    
    await log_tool_execution(
        tool_name="research_assistant",
        params={"topic": topic, "type": "comprehensive"},
        result={"total_sources": result['summary']['total_sources_found']}
    )
    
    return result
