#!/usr/bin/env python3
"""
Web Intelligence Module for Financial Data Scraping

Using Firecrawl API to extract structured financial information
from web sources for enhanced market analysis.
"""

import re
from typing import Dict, List, Any
from datetime import datetime
import logging

from firecrawl import Firecrawl, FirecrawlApp
from firecrawl.v2.types import Document, DocumentMetadata
from settings import settings

logger = logging.getLogger(__name__)


class FinancialWebIntelligence:
    """Financial web intelligence using Firecrawl API"""

    def __init__(self):
        self.firecrawl = None
        self.firecrawl_app = None
        if settings.firecrawl_api_key:
            try:
                self.firecrawl = Firecrawl(api_key=settings.firecrawl_api_key)
                self.firecrawl_app = FirecrawlApp(api_key=settings.firecrawl_api_key)
                logger.info("Firecrawl client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Firecrawl client: {e}")
                self.firecrawl = None
                self.firecrawl_app = None
        else:
            logger.warning("Firecrawl API key not configured")

    def analyze_web_content(
        self,
        url: str,
        analysis_type: str = "financial_news",
        extract_data_points: List[str] = None,
        include_links: bool = True,
    ) -> Dict[str, Any]:
        """Analyze specific web page for financial content"""

        if not self.firecrawl:
            return {
                "error": "Firecrawl API not configured",
                "url": url,
                "analysis_type": analysis_type,
            }

        if extract_data_points is None:
            extract_data_points = ["sentiment", "key_metrics", "price_targets"]

        try:
            logger.info(f"Analyzing web content: {url}")

            # Scrape the webpage with structured extraction using Firecrawl v2 API
            formats = ["markdown", "html"]
            if include_links:
                formats.append("links")

            scrape_result = self.firecrawl.scrape(
                url=url,
                formats=formats,
            )

            logger.debug(f"Firecrawl scrape result type: {type(scrape_result)}")

            # Extract data from Firecrawl v2 response using proper types
            if not isinstance(scrape_result, Document):
                return {
                    "error": "Unexpected response type from Firecrawl",
                    "url": url,
                    "response_type": str(type(scrape_result)),
                }

            markdown_content = scrape_result.markdown or ""
            metadata_obj: DocumentMetadata = scrape_result.metadata

            # Convert DocumentMetadata to dictionary using model_dump()
            metadata = metadata_obj.model_dump() if metadata_obj else {}

            if not markdown_content and not metadata:
                return {
                    "error": "No content extracted from URL",
                    "url": url,
                    "response_type": str(type(scrape_result)),
                }

            # Analyze the content
            analysis = self._analyze_financial_content(
                markdown_content, metadata, analysis_type, extract_data_points
            )

            # Structure the response
            result = {
                "url": url,
                "title": metadata.get("title", ""),
                "content_type": self._classify_content_type(markdown_content, metadata),
                "sentiment_score": analysis.get("sentiment_score", 0.0),
                "key_financial_metrics": analysis.get("financial_metrics", {}),
                "extracted_data": analysis.get("extracted_data", {}),
                "market_impact_assessment": analysis.get("market_impact", "neutral"),
                "credibility_score": self._assess_source_credibility(url, metadata),
                "source_quality": analysis.get("source_quality", "medium"),
                "timestamp": datetime.now().isoformat(),
                "raw_markdown": markdown_content[:1000] + "..."
                if len(markdown_content) > 1000
                else markdown_content,
                "metadata": metadata,
            }

            if include_links:
                # Handle links extraction using Document type
                links = scrape_result.links or []
                if links:
                    result["related_links"] = links[:10]  # Top 10 links

            logger.info(
                f"Successfully analyzed {url}: {analysis.get('sentiment_score', 0.0):.2f} sentiment"
            )
            return result

        except Exception as e:
            logger.error(f"Error analyzing web content {url}: {e}")
            return {
                "error": str(e),
                "url": url,
                "analysis_type": analysis_type,
            }

    def research_financial_topic(
        self,
        search_query: str,
        research_depth: str = "comprehensive",
        max_sources: int = 10,
        include_news: bool = True,
        include_analyst_reports: bool = True,
        extract_financial_data: bool = True,
        time_range: str = "1_week",
    ) -> Dict[str, Any]:
        """Comprehensive web research on financial topics"""

        if not self.firecrawl_app:
            return {
                "error": "Firecrawl API not configured",
                "search_query": search_query,
            }

        try:
            logger.info(f"Researching financial topic: {search_query}")

            # Use Firecrawl search functionality with FirecrawlApp (v3.4.0)
            # Limit to 5 sources max to prevent payload overflow and performance issues
            search_limit = min(max_sources, 5)

            # Use FirecrawlApp.search following official documentation
            search_result = self.firecrawl_app.search(
                query=search_query,
                limit=search_limit,
            )

            logger.debug(f"Firecrawl search result type: {type(search_result)}")

            # Extract search results from SearchData object (v3.4.0 format)
            # SearchData has direct attributes: web, news, images
            web_results = []
            if hasattr(search_result, "web") and search_result.web:
                web_results = search_result.web
            elif hasattr(search_result, "data") and hasattr(search_result.data, "web"):
                web_results = search_result.data.web
            elif isinstance(search_result, dict):
                web_results = search_result.get("web", [])

            if not web_results:
                return {
                    "error": "No search results found",
                    "search_query": search_query,
                    "sources_analyzed": 0,
                }

            # Analyze each source
            analyzed_sources = []
            overall_sentiments = []
            key_findings = []
            risk_factors = []
            opportunities = []
            financial_data = {}

            for i, result in enumerate(web_results[:max_sources]):
                try:
                    # Extract from SearchResultWeb object (v3.4.0) or dictionary format
                    if hasattr(result, "url"):
                        url = getattr(result, "url", "")
                        title = getattr(result, "title", "")
                    elif isinstance(result, dict):
                        url = result.get("url", "")
                        title = result.get("title", "")
                    else:
                        url = ""
                        title = ""

                    if not url:
                        continue

                    # Filter by content type if specified
                    if include_news and not include_analyst_reports:
                        if not self._is_news_source(url, title):
                            continue
                    elif include_analyst_reports and not include_news:
                        if not self._is_analyst_report(url, title):
                            continue

                    # Analyze this source with timeout protection
                    logger.debug(
                        f"Analyzing source {i+1}/{len(web_results[:max_sources])}: {url}"
                    )

                    source_analysis = self.analyze_web_content(
                        url=url,
                        analysis_type="financial_research",
                        extract_data_points=[
                            "sentiment",
                            "key_metrics",
                            "risks",
                            "opportunities",
                        ],
                        include_links=False,
                    )

                    if "error" not in source_analysis:
                        analyzed_sources.append(source_analysis)
                        overall_sentiments.append(
                            source_analysis.get("sentiment_score", 0.0)
                        )

                        # Extract insights
                        extracted_data = source_analysis.get("extracted_data", {})
                        if "key_findings" in extracted_data:
                            key_findings.extend(extracted_data["key_findings"])
                        if "risks" in extracted_data:
                            risk_factors.extend(extracted_data["risks"])
                        if "opportunities" in extracted_data:
                            opportunities.extend(extracted_data["opportunities"])

                        # Aggregate financial metrics
                        metrics = source_analysis.get("key_financial_metrics", {})
                        for metric, value in metrics.items():
                            if metric not in financial_data:
                                financial_data[metric] = []
                            financial_data[metric].append(value)

                except Exception as source_error:
                    logger.warning(f"Error analyzing source {i+1}: {source_error}")
                    continue

            # Calculate summary statistics
            sources_analyzed = len(analyzed_sources)
            overall_sentiment = (
                sum(overall_sentiments) / len(overall_sentiments)
                if overall_sentiments
                else 0.0
            )

            # Create consensus from financial data
            financial_consensus = {}
            for metric, values in financial_data.items():
                if isinstance(values[0], (int, float)):
                    financial_consensus[metric] = {
                        "average": sum(values) / len(values),
                        "range": [min(values), max(values)],
                        "consensus": "bullish"
                        if sum(values) / len(values) > 0
                        else "bearish",
                    }

            # Source quality distribution
            quality_distribution = {"high": 0, "medium": 0, "low": 0}
            for source in analyzed_sources:
                quality = source.get("source_quality", "medium")
                quality_distribution[quality] += 1

            result = {
                "search_query": search_query,
                "sources_analyzed": sources_analyzed,
                "overall_sentiment": overall_sentiment,
                "key_findings": list(set(key_findings))[:20],  # Top 20 unique findings
                "financial_consensus": financial_consensus,
                "risk_factors": list(set(risk_factors))[:15],  # Top 15 unique risks
                "opportunities": list(set(opportunities))[
                    :15
                ],  # Top 15 unique opportunities
                "source_quality_distribution": quality_distribution,
                "sentiment_range": [min(overall_sentiments), max(overall_sentiments)]
                if overall_sentiments
                else [0, 0],
                "research_depth": research_depth,
                "time_range": time_range,
                "analyzed_sources": analyzed_sources[
                    :5
                ],  # Include full analysis for top 5 sources
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"Research complete: {sources_analyzed} sources, {overall_sentiment:.2f} sentiment"
            )
            return result

        except Exception as e:
            logger.error(f"Error researching topic '{search_query}': {e}")
            return {
                "error": str(e),
                "search_query": search_query,
                "sources_analyzed": 0,
            }

    def _analyze_financial_content(
        self,
        content: str,
        metadata: Dict[str, Any],
        analysis_type: str,
        extract_data_points: List[str],
    ) -> Dict[str, Any]:
        """Analyze financial content and extract structured data"""

        analysis = {
            "sentiment_score": 0.0,
            "financial_metrics": {},
            "extracted_data": {},
            "market_impact": "neutral",
            "source_quality": "medium",
        }

        if not content:
            return analysis

        # Sentiment analysis
        if "sentiment" in extract_data_points:
            analysis["sentiment_score"] = self._calculate_sentiment_score(content)

        # Extract financial metrics
        if "key_metrics" in extract_data_points:
            analysis["financial_metrics"] = self._extract_financial_metrics(content)

        # Extract price targets
        if "price_targets" in extract_data_points:
            price_targets = self._extract_price_targets(content)
            if price_targets:
                analysis["extracted_data"]["price_targets"] = price_targets

        # Extract other data points
        if "risks" in extract_data_points:
            analysis["extracted_data"]["risks"] = self._extract_risk_factors(content)

        if "opportunities" in extract_data_points:
            analysis["extracted_data"]["opportunities"] = self._extract_opportunities(
                content
            )

        if "key_findings" in extract_data_points:
            analysis["extracted_data"]["key_findings"] = self._extract_key_findings(
                content
            )

        # Assess market impact
        analysis["market_impact"] = self._assess_market_impact(content, metadata)

        # Assess source quality
        url = metadata.get("sourceURL", "")
        analysis["source_quality"] = self._get_source_quality_from_url(url)

        return analysis

    def _calculate_sentiment_score(self, content: str) -> float:
        """Calculate financial sentiment score (-1.0 to 1.0)"""

        # Enhanced financial sentiment keywords
        positive_keywords = {
            "bullish": 0.8,
            "buy": 0.6,
            "strong": 0.5,
            "growth": 0.6,
            "profit": 0.7,
            "earnings beat": 0.9,
            "outperform": 0.7,
            "upgrade": 0.8,
            "rally": 0.7,
            "surge": 0.8,
            "soar": 0.9,
            "breakout": 0.7,
            "momentum": 0.6,
            "optimistic": 0.6,
            "revenue growth": 0.8,
            "margin expansion": 0.7,
            "dividend increase": 0.6,
            "share buyback": 0.5,
            "market share": 0.5,
            "competitive advantage": 0.6,
        }

        negative_keywords = {
            "bearish": -0.8,
            "sell": -0.6,
            "weak": -0.5,
            "decline": -0.6,
            "loss": -0.7,
            "earnings miss": -0.9,
            "underperform": -0.7,
            "downgrade": -0.8,
            "crash": -0.9,
            "plunge": -0.8,
            "fall": -0.5,
            "pessimistic": -0.6,
            "risk": -0.4,
            "revenue decline": -0.8,
            "margin compression": -0.7,
            "dividend cut": -0.8,
            "layoffs": -0.6,
            "investigation": -0.7,
            "lawsuit": -0.6,
            "bankruptcy": -1.0,
        }

        content_lower = content.lower()
        sentiment_scores = []

        # Score positive keywords
        for keyword, weight in positive_keywords.items():
            count = content_lower.count(keyword)
            if count > 0:
                sentiment_scores.extend([weight] * count)

        # Score negative keywords
        for keyword, weight in negative_keywords.items():
            count = content_lower.count(keyword)
            if count > 0:
                sentiment_scores.extend([weight] * count)

        if not sentiment_scores:
            return 0.0

        # Calculate weighted average and normalize
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        return max(-1.0, min(1.0, avg_sentiment))

    def _extract_financial_metrics(self, content: str) -> Dict[str, Any]:
        """Extract financial metrics from content"""

        metrics = {}

        # Revenue patterns
        revenue_patterns = [
            r"revenue[:\s]+\$?([\d,\.]+)\s*([bmk]illion)?",
            r"sales[:\s]+\$?([\d,\.]+)\s*([bmk]illion)?",
            r"\$?([\d,\.]+)\s*([bmk]illion)?\s+in revenue",
        ]

        # Earnings patterns
        earnings_patterns = [
            r"earnings[:\s]+\$?([\d,\.]+)\s*per share",
            r"eps[:\s]+\$?([\d,\.]+)",
            r"profit[:\s]+\$?([\d,\.]+)\s*([bmk]illion)?",
        ]

        # Price target patterns
        price_patterns = [
            r"price target[:\s]+\$?([\d,\.]+)",
            r"target price[:\s]+\$?([\d,\.]+)",
            r"fair value[:\s]+\$?([\d,\.]+)",
        ]

        # Market cap patterns
        market_cap_patterns = [
            r"market cap[:\s]+\$?([\d,\.]+)\s*([bmk]illion)?",
            r"valuation[:\s]+\$?([\d,\.]+)\s*([bmk]illion)?",
        ]

        # Extract metrics using regex
        for pattern in revenue_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                metrics["revenue"] = self._parse_financial_number(matches[0])
                break

        for pattern in earnings_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                metrics["earnings_per_share"] = self._parse_financial_number(matches[0])
                break

        for pattern in price_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                metrics["price_target"] = float(matches[0].replace(",", ""))
                break

        for pattern in market_cap_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                metrics["market_cap"] = self._parse_financial_number(matches[0])
                break

        return metrics

    def _extract_price_targets(self, content: str) -> List[Dict[str, Any]]:
        """Extract analyst price targets"""

        targets = []

        # Price target patterns with analyst info
        patterns = [
            r"(\w+\s+\w+|\w+)\s+raised.*?price target.*?\$?([\d,\.]+)",
            r"(\w+\s+\w+|\w+)\s+lowered.*?price target.*?\$?([\d,\.]+)",
            r"(\w+\s+\w+|\w+)\s+set.*?price target.*?\$?([\d,\.]+)",
            r"price target.*?\$?([\d,\.]+).*?(\w+\s+\w+|\w+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    analyst, price = match
                    try:
                        targets.append(
                            {
                                "analyst": analyst.strip(),
                                "price_target": float(price.replace(",", "")),
                                "action": "raised" if "raised" in pattern else "set",
                            }
                        )
                    except ValueError:
                        continue

        return targets[:5]  # Return top 5

    def _extract_risk_factors(self, content: str) -> List[str]:
        """Extract risk factors from content"""

        risk_keywords = [
            "risk",
            "concern",
            "challenge",
            "threat",
            "downside",
            "weakness",
            "volatility",
            "uncertainty",
            "headwind",
            "pressure",
            "decline",
        ]

        risks = []
        sentences = content.split(".")

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in risk_keywords):
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 20 and len(clean_sentence) < 200:
                    risks.append(clean_sentence)

        return risks[:10]  # Top 10 risks

    def _extract_opportunities(self, content: str) -> List[str]:
        """Extract opportunities from content"""

        opportunity_keywords = [
            "opportunity",
            "growth",
            "expansion",
            "upside",
            "potential",
            "catalyst",
            "breakthrough",
            "innovation",
            "competitive advantage",
            "market share",
        ]

        opportunities = []
        sentences = content.split(".")

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in opportunity_keywords):
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 20 and len(clean_sentence) < 200:
                    opportunities.append(clean_sentence)

        return opportunities[:10]  # Top 10 opportunities

    def _extract_key_findings(self, content: str) -> List[str]:
        """Extract key findings and insights"""

        # Look for sentences with key information indicators
        key_indicators = [
            "announced",
            "reported",
            "revealed",
            "disclosed",
            "confirmed",
            "expects",
            "forecasts",
            "projects",
            "estimates",
            "guidance",
            "increased",
            "decreased",
            "grew",
            "fell",
            "rose",
            "dropped",
        ]

        findings = []
        sentences = content.split(".")

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in key_indicators):
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 30 and len(clean_sentence) < 250:
                    findings.append(clean_sentence)

        return findings[:15]  # Top 15 findings

    def _parse_financial_number(self, match_tuple) -> float:
        """Parse financial number with billion/million/thousand suffixes"""

        if isinstance(match_tuple, tuple):
            number_str = match_tuple[0]
            suffix = match_tuple[1] if len(match_tuple) > 1 else ""
        else:
            number_str = str(match_tuple)
            suffix = ""

        try:
            number = float(number_str.replace(",", ""))

            suffix_lower = suffix.lower() if suffix else ""
            if "billion" in suffix_lower or suffix_lower == "b":
                number *= 1_000_000_000
            elif "million" in suffix_lower or suffix_lower == "m":
                number *= 1_000_000
            elif "thousand" in suffix_lower or suffix_lower == "k":
                number *= 1_000

            return number
        except (ValueError, AttributeError):
            return 0.0

    def _classify_content_type(self, content: str, metadata: Dict[str, Any]) -> str:
        """Classify the type of financial content"""

        title = metadata.get("title", "").lower()
        url = metadata.get("sourceURL", "").lower()
        content_lower = content.lower()

        # Check for earnings report
        if any(
            term in title or term in content_lower
            for term in ["earnings", "quarterly results", "q1", "q2", "q3", "q4"]
        ):
            return "earnings_report"

        # Check for analyst report
        if any(
            term in title or term in content_lower
            for term in ["rating", "upgrade", "downgrade", "analyst", "recommendation"]
        ):
            return "analyst_report"

        # Check for SEC filing
        if any(term in url for term in ["sec.gov", "edgar", "10-k", "10-q", "8-k"]):
            return "sec_filing"

        # Check for news
        if any(
            term in url
            for term in ["reuters", "bloomberg", "cnbc", "marketwatch", "wsj", "ft.com"]
        ):
            return "financial_news"

        # Default
        return "market_analysis"

    def _assess_market_impact(self, content: str, metadata: Dict[str, Any]) -> str:
        """Assess potential market impact of the content"""

        content_lower = content.lower()

        high_impact_terms = [
            "acquisition",
            "merger",
            "bankruptcy",
            "earnings beat",
            "earnings miss",
            "fda approval",
            "product recall",
            "ceo",
            "layoffs",
            "dividend",
            "stock split",
            "guidance",
            "investigation",
            "lawsuit",
        ]

        medium_impact_terms = [
            "partnership",
            "contract",
            "expansion",
            "new product",
            "analyst upgrade",
            "analyst downgrade",
            "rating change",
            "price target",
            "revenue growth",
        ]

        # Count impact terms
        high_impact_count = sum(
            1 for term in high_impact_terms if term in content_lower
        )
        medium_impact_count = sum(
            1 for term in medium_impact_terms if term in content_lower
        )

        if high_impact_count >= 2:
            return "high"
        elif high_impact_count >= 1 or medium_impact_count >= 3:
            return "medium"
        else:
            return "low"

    def _assess_source_credibility(self, url: str, metadata: Dict[str, Any]) -> float:
        """Assess source credibility (0.0 to 1.0)"""

        url_lower = url.lower()

        # High credibility sources
        if any(
            source in url_lower
            for source in [
                "reuters.com",
                "bloomberg.com",
                "wsj.com",
                "ft.com",
                "sec.gov",
            ]
        ):
            return 0.95

        # Medium-high credibility
        if any(
            source in url_lower
            for source in [
                "cnbc.com",
                "marketwatch.com",
                "yahoo.com",
                "investopedia.com",
            ]
        ):
            return 0.8

        # Medium credibility
        if any(
            source in url_lower
            for source in ["seekingalpha.com", "fool.com", "benzinga.com"]
        ):
            return 0.6

        # Check if it's an official company website
        if any(
            indicator in url_lower for indicator in ["investor", "ir.", "company.com"]
        ):
            return 0.85

        # Default for unknown sources
        return 0.5

    def _get_source_quality_from_url(self, url: str) -> str:
        """Get source quality category from URL"""

        credibility = self._assess_source_credibility(url, {})

        if credibility >= 0.9:
            return "high"
        elif credibility >= 0.7:
            return "medium"
        else:
            return "low"

    def _is_news_source(self, url: str, title: str) -> bool:
        """Check if source is a news website"""

        news_domains = [
            "reuters.com",
            "bloomberg.com",
            "cnbc.com",
            "marketwatch.com",
            "wsj.com",
            "ft.com",
            "yahoo.com",
            "benzinga.com",
        ]

        return any(domain in url.lower() for domain in news_domains)

    def _is_analyst_report(self, url: str, title: str) -> bool:
        """Check if source is an analyst report"""

        analyst_indicators = [
            "rating",
            "upgrade",
            "downgrade",
            "analyst",
            "recommendation",
            "price target",
            "research",
            "coverage",
        ]

        title_lower = title.lower()
        return any(indicator in title_lower for indicator in analyst_indicators)


# Global instance
web_intelligence = FinancialWebIntelligence()


# Tool functions for SGR integration
def analyze_web_content(
    url: str,
    analysis_type: str = "financial_news",
    extract_data_points: List[str] = None,
    include_links: bool = True,
) -> Dict[str, Any]:
    """Tool function for web content analysis"""
    return web_intelligence.analyze_web_content(
        url, analysis_type, extract_data_points, include_links
    )


def research_financial_topic(
    search_query: str,
    research_depth: str = "comprehensive",
    max_sources: int = 10,
    include_news: bool = True,
    include_analyst_reports: bool = True,
    extract_financial_data: bool = True,
    time_range: str = "1_week",
) -> Dict[str, Any]:
    """Tool function for comprehensive financial research"""
    return web_intelligence.research_financial_topic(
        search_query,
        research_depth,
        max_sources,
        include_news,
        include_analyst_reports,
        extract_financial_data,
        time_range,
    )


if __name__ == "__main__":
    # Test the web intelligence module
    print("Financial Web Intelligence Module")
    print("=" * 40)

    if web_intelligence.firecrawl:
        print("✓ Firecrawl client initialized")

        # Test URL analysis
        test_url = "https://finance.yahoo.com/news/apple-quarterly-earnings-beat-expectations-140000123.html"

        print(f"\nTesting web analysis on: {test_url}")
        result = analyze_web_content(
            url=test_url,
            analysis_type="earnings_report",
            extract_data_points=["sentiment", "key_metrics", "price_targets"],
        )

        if "error" not in result:
            print("✓ Analysis successful")
            print(f"  Sentiment: {result.get('sentiment_score', 0.0):.2f}")
            print(f"  Content type: {result.get('content_type', 'unknown')}")
            print(f"  Source quality: {result.get('source_quality', 'unknown')}")
        else:
            print(f"✗ Analysis failed: {result['error']}")

    else:
        print("✗ Firecrawl client not initialized")
        print("Check FIRECRAWL_API_KEY environment variable")
