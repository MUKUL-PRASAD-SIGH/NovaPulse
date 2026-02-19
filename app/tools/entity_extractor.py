"""Entity Extractor Tool - Extract and analyze entities from text."""
import re
from typing import Dict, List, Set
from collections import Counter, defaultdict
import json
from app.memory.store import log_tool_execution


class EntityExtractor:
    """Extract named entities and build knowledge graphs."""
    
    def __init__(self):
        # Common entity patterns
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)',
            'phone': r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            'money': r'\$\s?\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s?(?:USD|EUR|GBP|INR|billion|million|trillion)',
            'percentage': r'\b\d+(?:\.\d+)?%',
            'date': r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
        }
        
        # Common organization indicators
        self.org_indicators = {
            'Inc', 'Corp', 'LLC', 'Ltd', 'Co', 'Company', 'Corporation',
            'Group', 'Holdings', 'Partners', 'Associates', 'Technologies'
        }
        
        # Common person title indicators
        self.person_titles = {
            'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'CEO', 'President', 'Director',
            'Minister', 'Senator', 'Representative', 'Judge', 'Chief'
        }
        
        # Location indicators
        self.location_indicators = {
            'City', 'State', 'Country', 'County', 'Province', 'District',
            'Street', 'Avenue', 'Boulevard', 'Road'
        }
    
    def extract_entities(self, text: str) -> Dict:
        """
        Extract all entities from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with categorized entities
        """
        entities = {
            'people': self._extract_people(text),
            'organizations': self._extract_organizations(text),
            'locations': self._extract_locations(text),
            'emails': self._extract_pattern(text, 'email'),
            'urls': self._extract_pattern(text, 'url'),
            'phones': self._extract_pattern(text, 'phone'),
            'money': self._extract_pattern(text, 'money'),
            'percentages': self._extract_pattern(text, 'percentage'),
            'dates': self._extract_pattern(text, 'date'),
            'hashtags': self._extract_hashtags(text),
            'mentions': self._extract_mentions(text),
        }
        
        # Build relationships
        relationships = self._build_relationships(text, entities)
        
        # Calculate entity importance
        importance_scores = self._calculate_importance(entities, text)
        
        return {
            'entities': entities,
            'relationships': relationships,
            'importance_scores': importance_scores,
            'entity_count': sum(len(v) for v in entities.values()),
            'unique_entities': self._count_unique_entities(entities)
        }
    
    def _extract_pattern(self, text: str, pattern_name: str) -> List[str]:
        """Extract entities matching a regex pattern."""
        pattern = self.patterns.get(pattern_name)
        if not pattern:
            return []
        
        matches = re.findall(pattern, text, re.IGNORECASE)
        return list(set(matches))  # Remove duplicates
    
    def _extract_people(self, text: str) -> List[Dict]:
        """Extract person names with context."""
        people = []
        
        # Pattern: Title + Capitalized Name
        title_pattern = r'\b(' + '|'.join(self.person_titles) + r')\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        matches = re.findall(title_pattern, text)
        
        for title, name in matches:
            people.append({
                'name': name,
                'title': title,
                'full_mention': f"{title} {name}"
            })
        
        # Pattern: Capitalized names (2-3 words)
        name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        potential_names = re.findall(name_pattern, text)
        
        for name in potential_names:
            # Filter out common false positives
            if not any(word in name for word in ['The', 'This', 'That', 'These', 'Those']):
                if not any(p['name'] == name for p in people):
                    people.append({
                        'name': name,
                        'title': None,
                        'full_mention': name
                    })
        
        return people[:20]  # Limit to top 20
    
    def _extract_organizations(self, text: str) -> List[Dict]:
        """Extract organization names."""
        organizations = []
        
        # Pattern: Name + Org Indicator
        org_pattern = r'\b([A-Z][A-Za-z\s&]+(?:' + '|'.join(self.org_indicators) + r')\.?)\b'
        matches = re.findall(org_pattern, text)
        
        for org in matches:
            org = org.strip()
            if len(org) > 3 and not any(o['name'] == org for o in organizations):
                organizations.append({
                    'name': org,
                    'type': self._classify_organization(org)
                })
        
        # Pattern: All caps organizations (acronyms)
        acronym_pattern = r'\b([A-Z]{2,})\b'
        acronyms = re.findall(acronym_pattern, text)
        
        for acronym in acronyms:
            if len(acronym) >= 2 and acronym not in ['US', 'UK', 'EU', 'UN']:
                if not any(o['name'] == acronym for o in organizations):
                    organizations.append({
                        'name': acronym,
                        'type': 'acronym'
                    })
        
        return organizations[:30]  # Limit to top 30
    
    def _extract_locations(self, text: str) -> List[Dict]:
        """Extract location names."""
        locations = []
        
        # Pattern: Location + Indicator
        loc_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:' + '|'.join(self.location_indicators) + r'))\b'
        matches = re.findall(loc_pattern, text)
        
        for loc in matches:
            locations.append({
                'name': loc.strip(),
                'type': 'explicit'
            })
        
        # Common countries and cities (simplified list)
        known_locations = {
            'India', 'China', 'USA', 'America', 'United States', 'UK', 'Britain',
            'Russia', 'Japan', 'Germany', 'France', 'Canada', 'Australia',
            'New York', 'London', 'Tokyo', 'Paris', 'Beijing', 'Delhi', 'Mumbai'
        }
        
        for loc in known_locations:
            if re.search(r'\b' + re.escape(loc) + r'\b', text, re.IGNORECASE):
                if not any(l['name'].lower() == loc.lower() for l in locations):
                    locations.append({
                        'name': loc,
                        'type': 'known'
                    })
        
        return locations[:25]  # Limit to top 25
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags."""
        return list(set(re.findall(r'#(\w+)', text)))
    
    def _extract_mentions(self, text: str) -> List[str]:
        """Extract @mentions."""
        return list(set(re.findall(r'@(\w+)', text)))
    
    def _classify_organization(self, org_name: str) -> str:
        """Classify organization type."""
        org_lower = org_name.lower()
        
        if any(word in org_lower for word in ['bank', 'capital', 'investment', 'financial']):
            return 'financial'
        elif any(word in org_lower for word in ['tech', 'software', 'digital', 'ai', 'data']):
            return 'technology'
        elif any(word in org_lower for word in ['pharma', 'health', 'medical', 'bio']):
            return 'healthcare'
        elif any(word in org_lower for word in ['energy', 'oil', 'gas', 'power']):
            return 'energy'
        elif any(word in org_lower for word in ['government', 'ministry', 'department']):
            return 'government'
        else:
            return 'general'
    
    def _build_relationships(self, text: str, entities: Dict) -> List[Dict]:
        """Build relationships between entities."""
        relationships = []
        
        # Simple co-occurrence based relationships
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            # Find entities in this sentence
            sentence_entities = defaultdict(list)
            
            for person in entities['people']:
                if person['name'] in sentence:
                    sentence_entities['person'].append(person['name'])
            
            for org in entities['organizations']:
                if org['name'] in sentence:
                    sentence_entities['organization'].append(org['name'])
            
            for loc in entities['locations']:
                if loc['name'] in sentence:
                    sentence_entities['location'].append(loc['name'])
            
            # Create relationships
            if sentence_entities['person'] and sentence_entities['organization']:
                for person in sentence_entities['person']:
                    for org in sentence_entities['organization']:
                        relationships.append({
                            'source': person,
                            'target': org,
                            'type': 'person_org',
                            'context': sentence.strip()[:100]
                        })
            
            if sentence_entities['organization'] and sentence_entities['location']:
                for org in sentence_entities['organization']:
                    for loc in sentence_entities['location']:
                        relationships.append({
                            'source': org,
                            'target': loc,
                            'type': 'org_location',
                            'context': sentence.strip()[:100]
                        })
        
        return relationships[:50]  # Limit relationships
    
    def _calculate_importance(self, entities: Dict, text: str) -> Dict:
        """Calculate importance scores for entities."""
        scores = {}
        
        # Count mentions
        for category, entity_list in entities.items():
            if category in ['emails', 'urls', 'phones', 'money', 'percentages', 'dates', 'hashtags', 'mentions']:
                continue
            
            for entity in entity_list:
                name = entity.get('name', entity)
                if isinstance(name, str):
                    count = len(re.findall(re.escape(name), text, re.IGNORECASE))
                    scores[name] = {
                        'mentions': count,
                        'category': category,
                        'importance': min(count * 10, 100)  # Cap at 100
                    }
        
        return dict(sorted(scores.items(), key=lambda x: x[1]['importance'], reverse=True)[:20])
    
    def _count_unique_entities(self, entities: Dict) -> int:
        """Count total unique entities."""
        unique = set()
        
        for category, entity_list in entities.items():
            for entity in entity_list:
                if isinstance(entity, dict):
                    unique.add(entity.get('name', str(entity)))
                else:
                    unique.add(str(entity))
        
        return len(unique)


async def extract_entities(text: str) -> Dict:
    """
    Extract entities from text.
    
    Args:
        text: Input text
        
    Returns:
        Extracted entities and relationships
    """
    extractor = EntityExtractor()
    result = extractor.extract_entities(text)
    
    await log_tool_execution(
        tool_name="entity_extractor",
        params={"text_length": len(text)},
        result={"entity_count": result['entity_count'], "unique_entities": result['unique_entities']}
    )
    
    return result


async def extract_entities_from_articles(articles: List[Dict]) -> Dict:
    """
    Extract entities from multiple articles.
    
    Args:
        articles: List of article dicts with 'title' and 'content'
        
    Returns:
        Aggregated entity extraction results
    """
    extractor = EntityExtractor()
    
    # Combine all text
    combined_text = ""
    for article in articles:
        combined_text += f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')} "
    
    result = extractor.extract_entities(combined_text)
    
    await log_tool_execution(
        tool_name="entity_extractor",
        params={"article_count": len(articles)},
        result={"entity_count": result['entity_count']}
    )
    
    return result
