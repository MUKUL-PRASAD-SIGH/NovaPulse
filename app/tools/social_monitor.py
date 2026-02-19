"""Social Media Monitor Tool - Track trends across social platforms."""
import asyncio
import re
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import Counter
import httpx
from app.memory.store import log_tool_execution


class SocialMediaMonitor:
    """Monitor social media trends and sentiment."""
    
    def __init__(self):
        self.timeout = 10
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    async def monitor_topic(self, topic: str, platforms: Optional[List[str]] = None) -> Dict:
        """
        Monitor a topic across social media platforms.
        
        Args:
            topic: Topic to monitor
            platforms: List of platforms ['twitter', 'reddit'] (default: all)
            
        Returns:
            Social media intelligence
        """
        if platforms is None:
            platforms = ['twitter', 'reddit']
        
        results = {}
        
        # Monitor each platform
        tasks = []
        if 'twitter' in platforms:
            tasks.append(self._monitor_twitter(topic))
        if 'reddit' in platforms:
            tasks.append(self._monitor_reddit(topic))
        
        platform_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        for i, platform in enumerate(platforms):
            if i < len(platform_results) and not isinstance(platform_results[i], Exception):
                results[platform] = platform_results[i]
        
        # Aggregate insights
        aggregate = self._aggregate_insights(results)
        
        return {
            "topic": topic,
            "platforms": results,
            "aggregate": aggregate,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _monitor_twitter(self, topic: str) -> Dict:
        """
        Monitor Twitter/X for topic.
        Note: Requires Twitter API access in production.
        This is a simulated implementation.
        """
        # In production, integrate with Twitter API v2
        # For now, return simulated data structure
        
        return {
            "platform": "twitter",
            "status": "simulated",
            "note": "Integrate with Twitter API v2 for live data",
            "trending_score": 0,
            "tweet_volume": 0,
            "sentiment": {
                "positive": 0,
                "neutral": 0,
                "negative": 0
            },
            "top_tweets": [],
            "hashtags": [],
            "influencers": [],
            "velocity": {
                "tweets_per_hour": 0,
                "trend": "stable"
            }
        }
    
    async def _monitor_reddit(self, topic: str) -> Dict:
        """
        Monitor Reddit for topic.
        Uses Reddit's public JSON API (no auth required for basic data).
        """
        try:
            # Search Reddit — use encoded topic and week timeframe for better results
            import urllib.parse
            encoded_topic = urllib.parse.quote(topic)
            search_url = f"https://www.reddit.com/search.json?q={encoded_topic}&limit=25&sort=relevance&t=week&raw_json=1"
            
            # Reddit blocks generic User-Agents; use a compliant one
            reddit_headers = {
                'User-Agent': 'NovaIntelligence/1.0 (Windows; NewsMonitor)',
                'Accept': 'application/json'
            }
            
            async with httpx.AsyncClient(timeout=self.timeout, headers=reddit_headers, verify=False, follow_redirects=True) as client:
                response = await client.get(search_url)
                response.raise_for_status()
                data = response.json()
            
            posts = data.get('data', {}).get('children', [])
            
            # Analyze posts
            analysis = self._analyze_reddit_posts(posts, topic)
            
            return {
                "platform": "reddit",
                "status": "live",
                "post_count": len(posts),
                "sentiment": analysis['sentiment'],
                "top_posts": analysis['top_posts'],
                "subreddits": analysis['subreddits'],
                "engagement": analysis['engagement'],
                "velocity": analysis['velocity']
            }
            
        except Exception as e:
            return {
                "platform": "reddit",
                "status": "error",
                "error": str(e)
            }
    
    def _analyze_reddit_posts(self, posts: List[Dict], topic: str) -> Dict:
        """Analyze Reddit posts."""
        if not posts:
            return {
                "sentiment": {"positive": 0, "neutral": 0, "negative": 0},
                "top_posts": [],
                "subreddits": [],
                "engagement": {"total_upvotes": 0, "total_comments": 0},
                "velocity": {"posts_per_day": 0, "trend": "stable"}
            }
        
        # Extract post data
        post_data = []
        subreddits = []
        total_upvotes = 0
        total_comments = 0
        
        for post in posts:
            data = post.get('data', {})
            
            post_info = {
                "title": data.get('title', ''),
                "subreddit": data.get('subreddit', ''),
                "score": data.get('score', 0),
                "num_comments": data.get('num_comments', 0),
                "created_utc": data.get('created_utc', 0),
                "url": f"https://reddit.com{data.get('permalink', '')}",
                "author": data.get('author', '')
            }
            
            post_data.append(post_info)
            subreddits.append(data.get('subreddit', ''))
            total_upvotes += data.get('score', 0)
            total_comments += data.get('num_comments', 0)
        
        # Sentiment analysis (basic keyword-based)
        sentiment = self._analyze_sentiment_simple(post_data)
        
        # Top posts
        top_posts = sorted(post_data, key=lambda x: x['score'], reverse=True)[:5]
        
        # Subreddit distribution
        subreddit_counts = Counter(subreddits)
        top_subreddits = [{"name": sub, "count": count} for sub, count in subreddit_counts.most_common(5)]
        
        # Calculate velocity
        now = datetime.now().timestamp()
        recent_posts = [p for p in post_data if (now - p['created_utc']) < 86400]  # Last 24h
        posts_per_day = len(recent_posts)
        
        return {
            "sentiment": sentiment,
            "top_posts": top_posts,
            "subreddits": top_subreddits,
            "engagement": {
                "total_upvotes": total_upvotes,
                "total_comments": total_comments,
                "avg_upvotes": total_upvotes // len(posts) if posts else 0,
                "avg_comments": total_comments // len(posts) if posts else 0
            },
            "velocity": {
                "posts_per_day": posts_per_day,
                "trend": "rising" if posts_per_day > 10 else "stable"
            }
        }
    
    def _analyze_sentiment_simple(self, posts: List[Dict]) -> Dict:
        """Simple keyword-based sentiment analysis."""
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'awesome', 'love', 'best',
            'fantastic', 'wonderful', 'brilliant', 'positive', 'bullish', 'up'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'worst', 'hate', 'poor', 'negative',
            'bearish', 'down', 'crash', 'fail', 'disaster', 'scam'
        }
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for post in posts:
            title_lower = post['title'].lower()
            
            pos_score = sum(1 for word in positive_words if word in title_lower)
            neg_score = sum(1 for word in negative_words if word in title_lower)
            
            if pos_score > neg_score:
                positive_count += 1
            elif neg_score > pos_score:
                negative_count += 1
            else:
                neutral_count += 1
        
        total = len(posts)
        
        return {
            "positive": round(positive_count / total * 100, 1) if total > 0 else 0,
            "neutral": round(neutral_count / total * 100, 1) if total > 0 else 0,
            "negative": round(negative_count / total * 100, 1) if total > 0 else 0,
            "positive_count": positive_count,
            "neutral_count": neutral_count,
            "negative_count": negative_count
        }
    
    def _aggregate_insights(self, platform_results: Dict) -> Dict:
        """Aggregate insights across platforms."""
        total_mentions = 0
        avg_sentiment = {"positive": 0, "neutral": 0, "negative": 0}
        platforms_active = 0
        
        for platform, data in platform_results.items():
            if data.get('status') == 'live':
                platforms_active += 1
                
                # Aggregate mentions
                if platform == 'reddit':
                    total_mentions += data.get('post_count', 0)
                elif platform == 'twitter':
                    total_mentions += data.get('tweet_volume', 0)
                
                # Aggregate sentiment
                sentiment = data.get('sentiment', {})
                avg_sentiment['positive'] += sentiment.get('positive', 0)
                avg_sentiment['neutral'] += sentiment.get('neutral', 0)
                avg_sentiment['negative'] += sentiment.get('negative', 0)
        
        # Calculate averages
        if platforms_active > 0:
            avg_sentiment = {
                k: round(v / platforms_active, 1) 
                for k, v in avg_sentiment.items()
            }
        
        # Overall sentiment label
        if avg_sentiment['positive'] > 50:
            overall_sentiment = "positive"
        elif avg_sentiment['negative'] > 50:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        return {
            "total_mentions": total_mentions,
            "platforms_monitored": platforms_active,
            "avg_sentiment": avg_sentiment,
            "overall_sentiment": overall_sentiment,
            "social_buzz_level": "high" if total_mentions > 50 else "medium" if total_mentions > 10 else "low"
        }


async def monitor_social_media(topic: str, platforms: Optional[List[str]] = None) -> Dict:
    """
    Monitor social media for a topic.
    
    Args:
        topic: Topic to monitor
        platforms: Platforms to check (default: all)
        
    Returns:
        Social media intelligence
    """
    monitor = SocialMediaMonitor()
    result = await monitor.monitor_topic(topic, platforms)
    
    await log_tool_execution(
        tool_name="social_monitor",
        params={"topic": topic, "platforms": platforms or "all"},
        result={"total_mentions": result['aggregate']['total_mentions']}
    )
    
    return result


async def track_hashtag(hashtag: str) -> Dict:
    """
    Track a specific hashtag across platforms.
    
    Args:
        hashtag: Hashtag to track (with or without #)
        
    Returns:
        Hashtag tracking results
    """
    # Clean hashtag
    hashtag = hashtag.lstrip('#')
    
    monitor = SocialMediaMonitor()
    result = await monitor.monitor_topic(f"#{hashtag}")
    
    await log_tool_execution(
        tool_name="social_monitor",
        params={"hashtag": hashtag},
        result={"mentions": result['aggregate']['total_mentions']}
    )
    
    return result
