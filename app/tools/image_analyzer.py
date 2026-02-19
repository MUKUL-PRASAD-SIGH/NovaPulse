"""Image Analysis Tool - Analyze images from news articles."""
import asyncio
import base64
from typing import Dict, List, Optional
from io import BytesIO
import httpx
from PIL import Image
import json
from app.memory.store import log_tool_execution


class ImageAnalyzer:
    """Analyze images using Amazon Nova Vision and basic image processing."""
    
    def __init__(self):
        self.timeout = 15
        self.max_image_size = 5 * 1024 * 1024  # 5MB
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    async def analyze_image(self, image_url: str, context: Optional[str] = None) -> Dict:
        """
        Analyze a single image.
        
        Args:
            image_url: URL of the image to analyze
            context: Optional context about the image (article title, etc.)
            
        Returns:
            Analysis results including description, objects, text, metadata
        """
        try:
            # Download image
            image_data = await self._download_image(image_url)
            if not image_data:
                return {"success": False, "error": "Failed to download image"}
            
            # Get basic metadata
            metadata = self._get_image_metadata(image_data)
            
            # Extract text (OCR simulation - in production use AWS Textract or Tesseract)
            extracted_text = await self._extract_text(image_data)
            
            # Analyze with Nova Vision (simulated - in production use Amazon Bedrock Nova Vision)
            vision_analysis = await self._analyze_with_vision(image_data, context)
            
            # Detect image manipulation (basic checks)
            manipulation_score = self._detect_manipulation(image_data, metadata)
            
            return {
                "success": True,
                "url": image_url,
                "metadata": metadata,
                "extracted_text": extracted_text,
                "vision_analysis": vision_analysis,
                "manipulation_score": manipulation_score,
                "context": context
            }
            
        except Exception as e:
            return {
                "success": False,
                "url": image_url,
                "error": str(e)
            }
    
    async def _download_image(self, url: str) -> Optional[bytes]:
        """Download image from URL."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers, verify=False) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Check size
                if len(response.content) > self.max_image_size:
                    return None
                
                return response.content
                
        except Exception:
            return None
    
    def _get_image_metadata(self, image_data: bytes) -> Dict:
        """Extract image metadata."""
        try:
            image = Image.open(BytesIO(image_data))
            
            return {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "width": image.width,
                "height": image.height,
                "aspect_ratio": round(image.width / image.height, 2),
                "file_size_kb": len(image_data) // 1024
            }
        except Exception:
            return {}
    
    async def _extract_text(self, image_data: bytes) -> Dict:
        """
        Extract text from image (OCR).
        In production, use AWS Textract or Tesseract OCR.
        This is a placeholder implementation.
        """
        # Placeholder - in production integrate with AWS Textract or pytesseract
        return {
            "has_text": False,
            "text": "",
            "confidence": 0,
            "note": "OCR integration pending - use AWS Textract or Tesseract"
        }
    
    async def _analyze_with_vision(self, image_data: bytes, context: Optional[str]) -> Dict:
        """
        Analyze image with vision AI.
        In production, use Amazon Bedrock Nova Vision API.
        This is a simulated implementation.
        """
        # Placeholder for Nova Vision integration
        # In production, call Amazon Bedrock with Nova Vision model
        
        try:
            image = Image.open(BytesIO(image_data))
            
            # Basic analysis
            is_photo = image.mode in ['RGB', 'RGBA']
            is_grayscale = image.mode in ['L', 'LA']
            
            # Simulated analysis
            analysis = {
                "description": "Image analysis requires Amazon Bedrock Nova Vision API integration",
                "objects_detected": [],
                "scene_type": "unknown",
                "confidence": 0,
                "is_photo": is_photo,
                "is_grayscale": is_grayscale,
                "dominant_colors": self._get_dominant_colors(image),
                "note": "Integrate with Amazon Bedrock Nova Vision for full analysis"
            }
            
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_dominant_colors(self, image: Image.Image, num_colors: int = 3) -> List[str]:
        """Extract dominant colors from image."""
        try:
            # Resize for faster processing
            image = image.resize((150, 150))
            image = image.convert('RGB')
            
            # Get color histogram
            colors = image.getcolors(image.width * image.height)
            if not colors:
                return []
            
            # Sort by frequency
            sorted_colors = sorted(colors, key=lambda x: x[0], reverse=True)
            
            # Convert to hex
            dominant = []
            for count, color in sorted_colors[:num_colors]:
                hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
                dominant.append(hex_color)
            
            return dominant
            
        except Exception:
            return []
    
    def _detect_manipulation(self, image_data: bytes, metadata: Dict) -> Dict:
        """
        Detect potential image manipulation.
        Basic checks - in production use specialized forensics tools.
        """
        try:
            image = Image.open(BytesIO(image_data))
            
            # Basic checks
            checks = {
                "exif_data_present": bool(image.getexif()),
                "unusual_aspect_ratio": False,
                "suspicious_compression": False,
                "risk_level": "low"
            }
            
            # Check aspect ratio
            aspect_ratio = metadata.get('aspect_ratio', 1)
            if aspect_ratio < 0.3 or aspect_ratio > 3:
                checks["unusual_aspect_ratio"] = True
                checks["risk_level"] = "medium"
            
            # Check file size vs dimensions (compression analysis)
            pixels = metadata.get('width', 0) * metadata.get('height', 0)
            file_size_kb = metadata.get('file_size_kb', 0)
            
            if pixels > 0 and file_size_kb > 0:
                bytes_per_pixel = (file_size_kb * 1024) / pixels
                if bytes_per_pixel < 0.1 or bytes_per_pixel > 10:
                    checks["suspicious_compression"] = True
                    checks["risk_level"] = "medium"
            
            return checks
            
        except Exception:
            return {"error": "Could not analyze manipulation"}


async def analyze_image(image_url: str, context: Optional[str] = None) -> Dict:
    """
    Analyze a single image.
    
    Args:
        image_url: URL of image to analyze
        context: Optional context
        
    Returns:
        Image analysis results
    """
    analyzer = ImageAnalyzer()
    result = await analyzer.analyze_image(image_url, context)
    
    await log_tool_execution(
        tool_name="image_analyzer",
        params={"url": image_url},
        result={"success": result.get("success")}
    )
    
    return result


async def analyze_article_images(articles: List[Dict]) -> Dict:
    """
    Analyze images from multiple articles.
    
    Args:
        articles: List of articles with image URLs or article links
        
    Returns:
        Aggregated image analysis
    """
    analyzer = ImageAnalyzer()
    
    print(f"[IMAGE_ANALYZER] Received {len(articles)} articles")
    for i, a in enumerate(articles[:3]):
        print(f"  [{i}] title={a.get('title','')[:60]}, images={len(a.get('images',[]))}, link={a.get('link','')[:80]}")
    
    # Collect all image URLs — first from explicit 'images' field, then extract from article pages
    image_urls_with_context = []
    
    for article in articles[:8]:  # Limit to 8 articles
        images = article.get('images', [])
        title = article.get('title', '')
        
        if isinstance(images, list) and images:
            for img_url in images[:2]:
                # Accept both direct image URLs and any http URL (og:image URLs may not end in .jpg)
                if img_url and img_url.startswith('http'):
                    image_urls_with_context.append((img_url, title))
    
    print(f"[IMAGE_ANALYZER] Found {len(image_urls_with_context)} direct image URLs")
    
    # If we didn't find direct image URLs, extract og:image from article pages
    if not image_urls_with_context:
        og_tasks = []
        for article in articles[:5]:  # Top 5 articles
            link = article.get('link', '')
            if link:
                og_tasks.append(_extract_og_image(link))
        
        print(f"[IMAGE_ANALYZER] Extracting og:image from {len(og_tasks)} article pages...")
        og_results = await asyncio.gather(*og_tasks, return_exceptions=True)
        
        for i, result in enumerate(og_results):
            if isinstance(result, str) and result:
                title = articles[i].get('title', '') if i < len(articles) else ''
                image_urls_with_context.append((result, title))
                print(f"  [OK] og:image from article {i}: {result[:80]}")
            elif isinstance(result, Exception):
                print(f"  [FAIL] og:image error for article {i}: {result}")
            else:
                print(f"  [FAIL] og:image not found for article {i}")
    
    print(f"[IMAGE_ANALYZER] Total images to analyze: {len(image_urls_with_context)}")
    
    # Now analyze the collected images
    image_tasks = []
    for img_url, ctx in image_urls_with_context:
        image_tasks.append(analyzer.analyze_image(img_url, ctx))
    
    # Analyze in parallel
    results = await asyncio.gather(*image_tasks, return_exceptions=True)
    
    # Process results
    successful = [r for r in results if isinstance(r, dict) and r.get('success')]
    failed = len(results) - len(successful)
    
    print(f"[IMAGE_ANALYZER] Results: {len(successful)} successful, {failed} failed")
    
    # Extract insights
    total_text_found = sum(1 for r in successful if r.get('extracted_text', {}).get('has_text'))
    manipulation_risks = [r for r in successful if r.get('manipulation_score', {}).get('risk_level') in ['medium', 'high']]
    
    await log_tool_execution(
        tool_name="image_analyzer",
        params={"article_count": len(articles), "total_images": len(image_tasks)},
        result={"successful": len(successful), "failed": failed}
    )
    
    return {
        "total_images": len(results),
        "successful": len(successful),
        "failed": failed,
        "images_with_text": total_text_found,
        "manipulation_risks": len(manipulation_risks),
        "detailed_results": successful[:10]  # Return top 10 detailed results
    }


async def _extract_og_image(url: str) -> Optional[str]:
    """Extract og:image meta tag from an article URL."""
    try:
        async with httpx.AsyncClient(timeout=6, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }, verify=False, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text[:50000]  # Only check first 50KB
            
            # Extract og:image using simple string search (avoid full HTML parse)
            import re
            # Try og:image
            match = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\'](https?://[^"\']+)["\']', html, re.IGNORECASE)
            if not match:
                match = re.search(r'<meta[^>]+content=["\'](https?://[^"\']+)["\'][^>]+property=["\']og:image["\']', html, re.IGNORECASE)
            if match:
                return match.group(1)
            
            # Try twitter:image
            match = re.search(r'<meta[^>]+(?:name|property)=["\']twitter:image["\'][^>]+content=["\'](https?://[^"\']+)["\']', html, re.IGNORECASE)
            if not match:
                match = re.search(r'<meta[^>]+content=["\'](https?://[^"\']+)["\'][^>]+(?:name|property)=["\']twitter:image["\']', html, re.IGNORECASE)
            if match:
                return match.group(1)
                
    except Exception as e:
        print(f"[OG_IMAGE] Error fetching {url[:60]}: {e}")
    return None
