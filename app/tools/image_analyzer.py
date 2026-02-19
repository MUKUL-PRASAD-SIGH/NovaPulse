"""Image Analysis Tool - Analyze images from news articles with AI vision."""
import asyncio
import base64
import os
import json
import re
from typing import Dict, List, Optional
from io import BytesIO
import httpx
from PIL import Image
import boto3
from dotenv import load_dotenv
from app.memory.store import log_tool_execution

load_dotenv()


class ImageAnalyzer:
    """Analyze images using Amazon Nova Vision and intelligent image processing."""
    
    def __init__(self):
        self.timeout = 15
        self.max_image_size = 5 * 1024 * 1024  # 5MB
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    async def analyze_image(self, image_url: str, context: Optional[str] = None) -> Dict:
        """
        Analyze a single image with AI vision + metadata forensics.
        
        Args:
            image_url: URL of the image to analyze
            context: Optional context about the image (article title, etc.)
            
        Returns:
            Analysis results including AI description, objects, metadata, manipulation score
        """
        try:
            # Download image
            image_data = await self._download_image(image_url)
            if not image_data:
                return {"success": False, "error": "Failed to download image"}
            
            # Get basic metadata
            metadata = self._get_image_metadata(image_data)
            
            # If we can't even get dimensions, the file is not a valid image
            if not metadata.get("width") or not metadata.get("height"):
                return {"success": False, "url": image_url, "error": "Not a valid image format"}
            
            # Classify image type (photo/graphic/chart/screenshot)
            image_type = self._classify_image_type(image_data)
            
            # AI Vision analysis (Nova Bedrock or smart local fallback)
            vision_analysis = await self._analyze_with_vision(image_data, context)
            
            # Detect image manipulation
            manipulation_score = self._detect_manipulation(image_data, metadata)
            
            return {
                "success": True,
                "url": image_url,
                "metadata": metadata,
                "image_type": image_type,
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
                
                if len(response.content) > self.max_image_size:
                    return None
                
                return response.content
                
        except Exception:
            return None
    
    def _get_image_metadata(self, image_data: bytes) -> Dict:
        """Extract image metadata including EXIF."""
        try:
            image = Image.open(BytesIO(image_data))
            
            meta = {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "width": image.width,
                "height": image.height,
                "aspect_ratio": round(image.width / image.height, 2),
                "file_size_kb": len(image_data) // 1024
            }
            
            # Extract EXIF data if available
            exif = image.getexif()
            if exif:
                exif_tags = {}
                tag_names = {
                    271: "camera_make", 272: "camera_model",
                    274: "orientation", 306: "date_time",
                    305: "software", 37377: "shutter_speed"
                }
                for tag_id, tag_name in tag_names.items():
                    if tag_id in exif:
                        val = exif[tag_id]
                        if isinstance(val, bytes):
                            try:
                                val = val.decode("utf-8", errors="ignore").strip("\x00")
                            except Exception:
                                val = str(val)
                        exif_tags[tag_name] = str(val)
                if exif_tags:
                    meta["exif"] = exif_tags
            
            return meta
        except Exception:
            return {}
    
    def _classify_image_type(self, image_data: bytes) -> Dict:
        """Classify whether image is a photo, graphic, chart, or screenshot."""
        try:
            image = Image.open(BytesIO(image_data)).convert("RGB")
            small = image.resize((100, 100))
            pixels = list(small.getdata())
            
            unique_colors = len(set(pixels))
            total_pixels = len(pixels)
            color_ratio = unique_colors / total_pixels
            
            # Color variance
            r_vals = [p[0] for p in pixels]
            g_vals = [p[1] for p in pixels]
            b_vals = [p[2] for p in pixels]
            avg_r = sum(r_vals) / total_pixels
            avg_g = sum(g_vals) / total_pixels
            avg_b = sum(b_vals) / total_pixels
            
            # Edge detection: count sharp color transitions
            edges = 0
            for i in range(1, len(pixels)):
                diff = abs(pixels[i][0] - pixels[i-1][0]) + abs(pixels[i][1] - pixels[i-1][1]) + abs(pixels[i][2] - pixels[i-1][2])
                if diff > 100:
                    edges += 1
            edge_ratio = edges / total_pixels
            
            # Check for large flat color regions (indicates graphic/logo)
            from collections import Counter
            color_counts = Counter(pixels)
            top_color_pct = color_counts.most_common(1)[0][1] / total_pixels if color_counts else 0
            
            if color_ratio < 0.15 and top_color_pct > 0.3:
                img_type = "graphic"
                confidence = 0.85
            elif color_ratio < 0.25 and edge_ratio > 0.15:
                img_type = "chart"
                confidence = 0.7
            elif edge_ratio > 0.3 and color_ratio < 0.4:
                img_type = "screenshot"
                confidence = 0.65
            else:
                img_type = "photo"
                confidence = 0.8
            
            return {
                "type": img_type,
                "confidence": round(confidence, 2),
                "color_complexity": round(color_ratio, 3),
                "edge_density": round(edge_ratio, 3)
            }
        except Exception:
            return {"type": "unknown", "confidence": 0}
    
    async def _analyze_with_vision(self, image_data: bytes, context: Optional[str]) -> Dict:
        """
        Analyze image with Amazon Nova Vision via Bedrock.
        Always tries AI first (independent of USE_MOCK_PLANNER).
        Set IMAGE_AI=false to disable Nova Vision calls.
        """
        image_ai = os.getenv("IMAGE_AI", "true").lower() != "false"
        
        if image_ai:
            try:
                return await self._nova_vision_analyze(image_data, context)
            except Exception as e:
                print(f"[IMAGE_ANALYZER] Nova Vision failed, using smart local analysis: {e}")
        
        return self._local_vision_analysis(image_data, context)
    
    async def _nova_vision_analyze(self, image_data: bytes, context: Optional[str]) -> Dict:
        """Call Amazon Bedrock Nova Vision for real AI image understanding."""
        region = os.getenv("AWS_REGION", "us-east-1")
        client = boto3.client("bedrock-runtime", region_name=region)
        
        # Encode image to base64
        b64_image = base64.b64encode(image_data).decode("utf-8")
        
        # Detect format for media type
        try:
            img = Image.open(BytesIO(image_data))
            fmt = (img.format or "JPEG").lower()
            if fmt == "jpg":
                fmt = "jpeg"
        except Exception:
            fmt = "jpeg"
        
        context_hint = f" This image is from a news article titled: '{context}'." if context else ""
        
        prompt = f"""Analyze this news article image.{context_hint}

Return ONLY valid JSON with these fields:
{{
  "description": "One sentence describing what the image shows",
  "objects": ["list", "of", "key", "objects", "or", "people"],
  "scene_type": "one of: portrait, group, event, infographic, product, nature, building, abstract",
  "mood": "one of: positive, negative, neutral, dramatic, professional",
  "relevance": "one of: high, medium, low - how relevant is this image to the article title",
  "news_value": "One sentence on what this image adds to the news story"
}}"""

        body = {
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": fmt,
                            "source": {"bytes": b64_image}
                        }
                    },
                    {"text": prompt}
                ]
            }],
            "inferenceConfig": {"maxTokens": 300, "temperature": 0.3}
        }
        
        # Run synchronous boto3 call in executor to not block async loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: client.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            body=json.dumps(body),
            contentType="application/json"
        ))
        
        result = json.loads(response["body"].read())
        output_text = result.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "{}")
        output_text = re.sub(r"```json\s*|```\s*", "", output_text)
        
        parsed = json.loads(output_text.strip())
        
        # Add dominant colors from PIL (Nova doesn't return colors)
        try:
            image = Image.open(BytesIO(image_data))
            parsed["dominant_colors"] = self._get_dominant_colors(image)
        except Exception:
            parsed["dominant_colors"] = []
        
        parsed["source"] = "nova_vision"
        return parsed
    
    def _local_vision_analysis(self, image_data: bytes, context: Optional[str]) -> Dict:
        """Context-aware local analysis using image properties + article title."""
        try:
            image = Image.open(BytesIO(image_data))
            
            is_photo = image.mode in ['RGB', 'RGBA']
            is_grayscale = image.mode in ['L', 'LA']
            dominant_colors = self._get_dominant_colors(image)
            
            w, h = image.size
            aspect = w / h if h > 0 else 1
            
            # Scene classification from aspect ratio + resolution
            if aspect > 1.7:
                scene_type = "banner"
                layout_desc = "wide banner-format"
            elif aspect < 0.7:
                scene_type = "portrait"
                layout_desc = "vertical portrait-format"
            elif w > 1200 and h > 1200:
                scene_type = "high-res photo"
                layout_desc = "high-resolution"
            elif w < 400 or h < 300:
                scene_type = "thumbnail"
                layout_desc = "small thumbnail"
            else:
                scene_type = "standard"
                layout_desc = "standard-format"
            
            # Brightness analysis
            gray = image.convert("L")
            pixels = list(gray.getdata())
            avg_brightness = sum(pixels) / len(pixels)
            
            if avg_brightness < 60:
                mood = "dark/dramatic"
                brightness_desc = "dark-toned"
            elif avg_brightness > 200:
                mood = "bright/positive"
                brightness_desc = "bright, well-lit"
            elif avg_brightness > 140:
                mood = "neutral"
                brightness_desc = "naturally lit"
            else:
                mood = "neutral"
                brightness_desc = "moderately lit"
            
            # Color temperature analysis
            rgb_image = image.convert("RGB").resize((50, 50))
            rgb_pixels = list(rgb_image.getdata())
            avg_r = sum(p[0] for p in rgb_pixels) / len(rgb_pixels)
            avg_b = sum(p[2] for p in rgb_pixels) / len(rgb_pixels)
            
            if avg_r > avg_b + 30:
                color_temp = "warm-toned"
            elif avg_b > avg_r + 30:
                color_temp = "cool-toned"
            else:
                color_temp = "balanced color"
            
            # Contrast analysis
            min_px = min(pixels)
            max_px = max(pixels)
            contrast_range = max_px - min_px
            if contrast_range > 200:
                contrast_desc = "high contrast"
            elif contrast_range < 80:
                contrast_desc = "low contrast"
            else:
                contrast_desc = "moderate contrast"
            
            # Build context-aware description
            if context:
                # Use article title to make the description meaningful
                topic = context[:80]
                if is_grayscale:
                    description = f"Grayscale {layout_desc} news image related to: {topic}"
                else:
                    description = f"{brightness_desc.capitalize()}, {color_temp} {layout_desc} news photo related to: {topic}"
                
                news_value = f"Visual accompaniment for coverage on {topic}"
                relevance = "high"
            else:
                if is_grayscale:
                    description = f"Grayscale {layout_desc} image ({w}x{h}), {contrast_desc}"
                else:
                    description = f"{brightness_desc.capitalize()}, {color_temp} {layout_desc} image ({w}x{h}), {contrast_desc}"
                
                news_value = "News imagery without article context"
                relevance = "medium"
            
            # Infer likely content from image properties
            inferred_objects = []
            if scene_type == "portrait" and not is_grayscale:
                inferred_objects.append("person (likely)")
            if contrast_range > 200 and avg_brightness < 100:
                inferred_objects.append("high-contrast scene")
            if len(dominant_colors) >= 3:
                inferred_objects.append(f"{len(dominant_colors)} distinct color regions")
            
            return {
                "description": description,
                "objects": inferred_objects,
                "scene_type": scene_type,
                "mood": mood,
                "relevance": relevance,
                "news_value": news_value,
                "is_photo": is_photo,
                "is_grayscale": is_grayscale,
                "avg_brightness": round(avg_brightness),
                "color_temperature": color_temp,
                "contrast": contrast_desc,
                "dominant_colors": dominant_colors,
                "source": "local_analysis"
            }
            
        except Exception as e:
            return {"error": str(e), "source": "local_analysis"}
    
    def _get_dominant_colors(self, image: Image.Image, num_colors: int = 5) -> List[str]:
        """Extract dominant colors from image."""
        try:
            image = image.resize((150, 150))
            image = image.convert('RGB')
            
            colors = image.getcolors(image.width * image.height)
            if not colors:
                return []
            
            sorted_colors = sorted(colors, key=lambda x: x[0], reverse=True)
            
            dominant = []
            for count, color in sorted_colors[:num_colors]:
                hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
                dominant.append(hex_color)
            
            return dominant
            
        except Exception:
            return []
    
    def _detect_manipulation(self, image_data: bytes, metadata: Dict) -> Dict:
        """
        Detect potential image manipulation with enhanced heuristics.
        """
        try:
            image = Image.open(BytesIO(image_data))
            
            checks = {
                "exif_data_present": bool(image.getexif()),
                "unusual_aspect_ratio": False,
                "suspicious_compression": False,
                "risk_level": "low",
                "flags": []
            }
            
            # Check aspect ratio
            aspect_ratio = metadata.get('aspect_ratio', 1)
            if aspect_ratio < 0.3 or aspect_ratio > 3:
                checks["unusual_aspect_ratio"] = True
                checks["flags"].append("Unusual aspect ratio")
            
            # Check file size vs dimensions (compression analysis)
            pixels = metadata.get('width', 0) * metadata.get('height', 0)
            file_size_kb = metadata.get('file_size_kb', 0)
            
            if pixels > 0 and file_size_kb > 0:
                bytes_per_pixel = (file_size_kb * 1024) / pixels
                if bytes_per_pixel < 0.05:
                    checks["suspicious_compression"] = True
                    checks["flags"].append("Heavy compression detected")
                elif bytes_per_pixel > 10:
                    checks["flags"].append("Unusually large file size")
            
            # Check for uniform borders (possible crop/paste)
            try:
                rgb = image.convert("RGB")
                w, h = rgb.size
                if w > 20 and h > 20:
                    # Check top and bottom 5px strips
                    top_strip = list(rgb.crop((0, 0, w, 5)).getdata())
                    bottom_strip = list(rgb.crop((0, h-5, w, h)).getdata())
                    
                    top_unique = len(set(top_strip))
                    bottom_unique = len(set(bottom_strip))
                    
                    if top_unique < 3 or bottom_unique < 3:
                        checks["flags"].append("Uniform border detected (possible composite)")
            except Exception:
                pass
            
            # EXIF software check
            exif_data = metadata.get("exif", {})
            software = exif_data.get("software", "").lower()
            if software and any(tool in software for tool in ["photoshop", "gimp", "paint"]):
                checks["flags"].append(f"Editing software in EXIF: {exif_data['software']}")
            
            # Calculate risk level
            flag_count = len(checks["flags"])
            if flag_count >= 3:
                checks["risk_level"] = "high"
            elif flag_count >= 1 or checks["suspicious_compression"] or checks["unusual_aspect_ratio"]:
                checks["risk_level"] = "medium"
            else:
                checks["risk_level"] = "low"
            
            return checks
            
        except Exception:
            return {"error": "Could not analyze manipulation", "risk_level": "unknown"}


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
        Aggregated image analysis with AI insights
    """
    analyzer = ImageAnalyzer()
    
    print(f"[IMAGE_ANALYZER] Received {len(articles)} articles")
    for i, a in enumerate(articles[:3]):
        print(f"  [{i}] title={a.get('title','')[:60]}, images={len(a.get('images',[]))}, link={a.get('link','')[:80]}")
    
    # Collect all image URLs (deduplicated)
    image_urls_with_context = []
    seen_urls = set()
    
    def _add_image(url, title):
        """Add image URL if not already seen and looks like a real image."""
        if not url or not url.startswith('http'):
            return
        # Normalize URL for dedup
        clean_url = url.split('?')[0].split('#')[0]
        if clean_url in seen_urls:
            return
        # Skip tracking pixels, social icons, and tiny placeholders
        skip_patterns = ['/favicon', '/icon', '/logo-small', 'tracking', '1x1', 'pixel', '.svg']
        if any(p in url.lower() for p in skip_patterns):
            return
        seen_urls.add(clean_url)
        image_urls_with_context.append((url, title))
    
    for article in articles[:8]:
        images = article.get('images', [])
        title = article.get('title', '')
        
        # Check for direct image URL from GNews/Tavily
        _add_image(article.get('image', ''), title)
        
        if isinstance(images, list) and images:
            for img_url in images[:2]:
                _add_image(img_url, title)
    
    print(f"[IMAGE_ANALYZER] Found {len(image_urls_with_context)} direct image URLs (deduped)")
    
    # If no direct images, extract og:image from article pages
    if not image_urls_with_context:
        og_tasks = []
        for article in articles[:5]:
            link = article.get('link', '')
            if link:
                og_tasks.append(_extract_og_image(link))
        
        print(f"[IMAGE_ANALYZER] Extracting og:image from {len(og_tasks)} article pages...")
        og_results = await asyncio.gather(*og_tasks, return_exceptions=True)
        
        for i, result in enumerate(og_results):
            if isinstance(result, str) and result:
                title = articles[i].get('title', '') if i < len(articles) else ''
                _add_image(result, title)
                print(f"  [OK] og:image from article {i}: {result[:80]}")
            elif isinstance(result, Exception):
                print(f"  [FAIL] og:image error for article {i}: {result}")
            else:
                print(f"  [FAIL] og:image not found for article {i}")
    
    print(f"[IMAGE_ANALYZER] Total images to analyze: {len(image_urls_with_context)}")
    
    # Analyze in parallel
    image_tasks = []
    for img_url, ctx in image_urls_with_context:
        image_tasks.append(analyzer.analyze_image(img_url, ctx))
    
    results = await asyncio.gather(*image_tasks, return_exceptions=True)
    
    # Process results
    successful = [r for r in results if isinstance(r, dict) and r.get('success')]
    failed = len(results) - len(successful)
    
    print(f"[IMAGE_ANALYZER] Results: {len(successful)} successful, {failed} failed")
    
    # Extract aggregate insights
    manipulation_risks = [r for r in successful if r.get('manipulation_score', {}).get('risk_level') in ['medium', 'high']]
    
    # Count image types
    type_counts = {}
    for r in successful:
        it = r.get("image_type", {}).get("type", "unknown")
        type_counts[it] = type_counts.get(it, 0) + 1
    
    # Collect all detected objects from AI vision
    all_objects = []
    descriptions = []
    for r in successful:
        va = r.get("vision_analysis", {})
        all_objects.extend(va.get("objects", []))
        desc = va.get("description", "")
        if desc:
            descriptions.append(desc)
    
    await log_tool_execution(
        tool_name="image_analyzer",
        params={"article_count": len(articles), "total_images": len(image_tasks)},
        result={"successful": len(successful), "failed": failed}
    )
    
    return {
        "total_images": len(results),
        "successful": len(successful),
        "failed": failed,
        "manipulation_risks": len(manipulation_risks),
        "image_types": type_counts,
        "objects_detected": list(set(all_objects))[:20],
        "descriptions": descriptions[:5],
        "detailed_results": successful[:10]
    }


async def _extract_og_image(url: str) -> Optional[str]:
    """Extract og:image meta tag from an article URL."""
    try:
        async with httpx.AsyncClient(timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }, verify=False, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text[:50000]
            
            import re
            # Try og:image
            match = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\'](https?://[^"\'>]+)["\']', html, re.IGNORECASE)
            if not match:
                match = re.search(r'<meta[^>]+content=["\'](https?://[^"\'>]+)["\'][^>]+property=["\']og:image["\']', html, re.IGNORECASE)
            if match:
                return match.group(1)
            
            # Try twitter:image
            match = re.search(r'<meta[^>]+(?:name|property)=["\']twitter:image["\'][^>]+content=["\'](https?://[^"\'>]+)["\']', html, re.IGNORECASE)
            if not match:
                match = re.search(r'<meta[^>]+content=["\'](https?://[^"\'>]+)["\'][^>]+(?:name|property)=["\']twitter:image["\']', html, re.IGNORECASE)
            if match:
                return match.group(1)
                
    except Exception as e:
        print(f"[OG_IMAGE] Error fetching {url[:60]}: {e}")
    return None

