#!/usr/bin/env python3
"""
Agentic Research Assistant with Multi-Agent Architecture
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import re # Added for regex in _calculate_relevance

from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchPhase(Enum):
    """Research phases enumeration"""
    PLANNING = "planning"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    WRITING = "writing"
    EDITING = "editing"
    COMPLETE = "complete"

@dataclass
class LiteratureSource:
    """Literature source data structure"""
    title: str
    authors: str
    year: int
    journal: str
    summary: str
    relevance_score: float
    url: str
    doi: str
    citation_count: int
    abstract: str

@dataclass
class DataInsight:
    """Data insight structure"""
    category: str
    value: str
    significance: str
    source: str
    confidence: float
    trend: str

@dataclass
class ResearchMetrics:
    """Research performance metrics with comprehensive scoring"""
    # Core metrics
    total_sources: int
    avg_relevance: float
    data_points: int
    paper_length: int
    completion_time: float
    quality_score: float
    
    # Source quality metrics
    high_relevance_sources: int  # Sources with relevance >= 8.0
    medium_relevance_sources: int  # Sources with relevance 5.0-7.9
    low_relevance_sources: int  # Sources with relevance < 5.0
    relevance_consistency: float  # Standard deviation of relevance scores
    source_diversity_score: float  # 0-10 score based on source variety
    
    # Citation metrics
    total_citations: int
    avg_citations_per_source: float
    highly_cited_sources: int  # Sources with >= 100 citations
    citation_impact_score: float  # 0-10 score based on citation patterns
    
    # Temporal metrics
    recent_sources: int  # Sources from last 3 years
    current_sources: int  # Sources from last 5 years
    outdated_sources: int  # Sources over 10 years old
    temporal_freshness_score: float  # 0-10 score based on recency
    
    # Data insight metrics
    high_confidence_insights: int  # Insights with confidence >= 0.8
    medium_confidence_insights: int  # Insights with confidence 0.6-0.79
    low_confidence_insights: int  # Insights with confidence < 0.6
    insight_significance_score: float  # 0-10 score based on significance levels
    insight_diversity_score: float  # 0-10 score based on insight variety
    
    # Paper quality metrics
    content_structure_score: float  # 0-10 score based on paper sections
    length_adequacy_score: float  # 0-10 score based on target length
    writing_quality_score: float  # 0-10 score based on content indicators
    
    # Research depth metrics
    methodology_coverage_score: float  # 0-10 score based on method diversity
    theoretical_framework_score: float  # 0-10 score based on framework coverage
    empirical_evidence_score: float  # 0-10 score based on data quality
    
    # Efficiency metrics
    research_efficiency_score: float  # 0-10 score based on time vs. quality
    phase_completion_scores: Dict[str, float]  # Individual phase scores
    
    # Overall assessment scores
    comprehensiveness_score: float  # 0-10 overall research coverage
    rigor_score: float  # 0-10 methodological rigor
    innovation_score: float  # 0-10 novelty and originality
    impact_potential_score: float  # 0-10 potential research impact

class AgenticResearchAssistant:
    """Agentic research assistant with improved architecture and features"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the research assistant"""
        self.config = config or self._default_config()
        self.current_phase = ResearchPhase.PLANNING
        self.completion_percentage = 0.0
        self.start_time = None
        self.end_time = None
        
        # Research data
        self.literature_sources: List[LiteratureSource] = []
        self.data_insights: List[DataInsight] = []
        self.paper_draft = ""
        self.final_paper = ""
        self.agent_messages: List[str] = []
        self.research_notes: List[str] = []
        
        # Initialize LLM and tools
        self._initialize_llm()
        self._initialize_tools()
        
        # Performance tracking
        self.metrics = ResearchMetrics(
            # Core metrics
            total_sources=0,
            avg_relevance=0.0,
            data_points=0,
            paper_length=0,
            completion_time=0.0,
            quality_score=0.0,
            
            # Source quality metrics
            high_relevance_sources=0,
            medium_relevance_sources=0,
            low_relevance_sources=0,
            relevance_consistency=0.0,
            source_diversity_score=0.0,
            
            # Citation metrics
            total_citations=0,
            avg_citations_per_source=0.0,
            highly_cited_sources=0,
            citation_impact_score=0.0,
            
            # Temporal metrics
            recent_sources=0,
            current_sources=0,
            outdated_sources=0,
            temporal_freshness_score=0.0,
            
            # Data insight metrics
            high_confidence_insights=0,
            medium_confidence_insights=0,
            low_confidence_insights=0,
            insight_significance_score=0.0,
            insight_diversity_score=0.0,
            
            # Paper quality metrics
            content_structure_score=0.0,
            length_adequacy_score=0.0,
            writing_quality_score=0.0,
            
            # Research depth metrics
            methodology_coverage_score=0.0,
            theoretical_framework_score=0.0,
            empirical_evidence_score=0.0,
            
            # Efficiency metrics
            research_efficiency_score=0.0,
            phase_completion_scores={},
            
            # Overall assessment scores
            comprehensiveness_score=0.0,
            rigor_score=0.0,
            innovation_score=0.0,
            impact_potential_score=0.0
        )
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "max_sources": 10,
            "min_relevance_score": 7.0,
            "paper_target_length": 15000,
            "enable_citations": True,
            "enable_plagiarism_check": True,
            "research_depth": "comprehensive"
        }
    
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            self.llm = ChatGroq(
                groq_api_key=api_key,
                model_name="llama-3.1-8b-instant"
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _initialize_tools(self):
        """Initialize research tools"""
        try:
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                logger.warning("TAVILY_API_KEY not found, search functionality limited")
                self.search_tool = None
            else:
                self.search_tool = TavilySearchResults(max_results=self.config["max_sources"])
                logger.info("Search tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize search tool: {e}")
            self.search_tool = None
    
    def research_director(self, topic: str) -> str:
        """Research Director - Strategic planning and coordination"""
        self.current_phase = ResearchPhase.RESEARCH
        self.completion_percentage = 10.0
        self.start_time = datetime.now()
        
        # Create research plan
        plan_prompt = f"""Create a comprehensive research plan for: {topic}
        
        Include:
        1. Research objectives
        2. Key research questions
        3. Methodology approach
        4. Expected outcomes
        5. Timeline and milestones
        
        Provide a structured, actionable plan."""
        
        try:
            response = self.llm.invoke([HumanMessage(content=plan_prompt)])
            research_plan = response.content
            
            message = f"🎯 Research Director: Strategic plan created. Initiating research phase."
            self.agent_messages.append(message)
            self.research_notes.append(f"Research Plan: {research_plan[:200]}...")
            
            return message
        except Exception as e:
            error_msg = f"Research Director failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def literature_agent(self, topic: str) -> str:
        """Literature Agent - Advanced literature search and analysis"""
        try:
            if not self.search_tool:
                # Create mock sources for demonstration
                self.literature_sources = self._create_mock_sources(topic)
            else:
                # Perform actual search
                search_query = f"academic research papers {topic} recent studies 2020-2024"
                search_results = self.search_tool.invoke(search_query)
                self.literature_sources = self._process_search_results(search_results, topic)
            
            # Analyze and rank sources
            self._analyze_sources()
            
            self.completion_percentage = 30.0
            message = f"🔍 Literature Agent: Found and analyzed {len(self.literature_sources)} relevant sources."
            self.agent_messages.append(message)
            
            return message
            
        except Exception as e:
            error_msg = f"Literature Agent failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _create_mock_sources(self, topic: str) -> List[LiteratureSource]:
        """Create mock literature sources for demonstration"""
        sources = []
        for i in range(5):
            source = LiteratureSource(
                title=f"Research on {topic} - Study {i+1}",
                authors=f"Author {i+1}, et al.",
                year=2023 - i,
                journal=f"Journal of {topic.split()[0]} Research",
                summary=f"Comprehensive study on {topic} with innovative methodology",
                relevance_score=9.0 - i * 0.5,
                url=f"https://example.com/paper{i+1}",
                doi=f"10.1000/example{i+1}",
                citation_count=50 - i * 10,
                abstract=f"This study investigates {topic} using advanced analytical methods..."
            )
            sources.append(source)
        return sources
    
    def _process_search_results(self, results: List, topic: str) -> List[LiteratureSource]:
        """Process and structure search results"""
        sources = []
        for result in results:
            try:
                source = LiteratureSource(
                    title=result.get('title', 'Unknown Title'),
                    authors=result.get('authors', 'Unknown Authors'),
                    year=int(result.get('year', 2023)),
                    journal=result.get('journal', 'Unknown Journal'),
                    summary=result.get('content', '')[:200] + "...",
                    relevance_score=self._calculate_relevance(result, topic),
                    url=result.get('url', ''),
                    doi=result.get('doi', ''),
                    citation_count=int(result.get('citation_count', 0)),
                    abstract=result.get('abstract', '')
                )
                sources.append(source)
            except Exception as e:
                logger.warning(f"Failed to process search result: {e}")
                continue
        return sources
    
    def _calculate_relevance(self, result: Dict, topic: str) -> float:
        """Calculate semantic relevance score using LLM analysis"""
        try:
            # Extract content for analysis
            title = result.get('title', '')
            abstract = result.get('abstract', '')
            content = result.get('content', '')
            
            # Combine all available text for comprehensive analysis
            full_text = f"Title: {title}\nAbstract: {abstract}\nContent: {content}"
            
            if not full_text.strip():
                return 0.0
            
            # Create semantic relevance analysis prompt
            relevance_prompt = f"""Analyze the relevance of this research source to the topic: "{topic}"

Source Information:
{full_text}

Please evaluate the relevance on a scale of 0-10 based on:
1. **Topical Alignment** (0-3 points): How directly does this source address the research topic?
2. **Methodological Relevance** (0-3 points): Does the source provide useful methods/approaches for the topic?
3. **Contextual Value** (0-2 points): Does it offer valuable context, background, or related insights?
4. **Current Relevance** (0-2 points): Is the information current and applicable to modern research?

Provide your analysis in this exact format:
Score: [0-10]
Reasoning: [Brief explanation of your scoring]

Focus on semantic meaning and research value, not just keyword matches."""
            
            # Get LLM-based relevance assessment
            response = self.llm.invoke([HumanMessage(content=relevance_prompt)])
            response_text = response.content.strip()
            
            # Extract score from response
            score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', response_text)
            if score_match:
                score = float(score_match.group(1))
                # Ensure score is within valid range
                return max(0.0, min(10.0, score))
            else:
                # Fallback to simple keyword matching if LLM parsing fails
                logger.warning("LLM relevance parsing failed, falling back to keyword matching")
                return self._fallback_relevance_calculation(result, topic)
                
        except Exception as e:
            logger.warning(f"LLM-based relevance calculation failed: {e}, using fallback")
            return self._fallback_relevance_calculation(result, topic)
    
    def _fallback_relevance_calculation(self, result: Dict, topic: str) -> float:
        """Fallback relevance calculation using keyword matching"""
        content = result.get('content', '').lower()
        topic_words = topic.lower().split()
        matches = sum(1 for word in topic_words if word in content)
        return min(10.0, (matches / len(topic_words)) * 10)
    
    def _analyze_sources(self):
        """Analyze and rank literature sources"""
        if not self.literature_sources:
            return
        
        # Sort by relevance and citation count
        self.literature_sources.sort(
            key=lambda x: (x.relevance_score, x.citation_count), 
            reverse=True
        )
        
        # Update basic metrics
        self.metrics.total_sources = len(self.literature_sources)
        self.metrics.avg_relevance = sum(s.relevance_score for s in self.literature_sources) / len(self.literature_sources)
        
        # Calculate comprehensive metrics
        self._calculate_comprehensive_metrics()
        
        # Update quality score after comprehensive analysis
        self.metrics.quality_score = self._calculate_quality_score()
    
    def data_agent(self, topic: str) -> str:
        """Data Agent - Advanced data collection and analysis"""
        try:
            # Generate dynamic insights from actual research data
            self.data_insights = self._generate_dynamic_insights(topic)
            
            self.completion_percentage = 50.0
            self.metrics.data_points = len(self.data_insights)
            
            # Calculate comprehensive metrics after insights generation
            self._calculate_comprehensive_metrics()
            
            # Update quality score after generating insights
            self.metrics.quality_score = self._calculate_quality_score()
            
            message = f"📊 Data Agent: Generated {len(self.data_insights)} dynamic insights from research analysis."
            self.agent_messages.append(message)
            
            return message
            
        except Exception as e:
            error_msg = f"Data Agent failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _generate_dynamic_insights(self, topic: str) -> List[DataInsight]:
        """Generate data insights from actual research data and literature analysis"""
        insights = []
        
        try:
            # Insight 1: Research Coverage Analysis
            if self.literature_sources:
                coverage_insight = self._analyze_research_coverage(topic)
                insights.append(coverage_insight)
            
            # Insight 2: Temporal Trends Analysis
            if self.literature_sources:
                temporal_insight = self._analyze_temporal_trends()
                insights.append(temporal_insight)
            
            # Insight 3: Methodological Patterns
            if self.literature_sources:
                method_insight = self._analyze_methodological_patterns()
                insights.append(method_insight)
            
            # Insight 4: Citation Impact Analysis
            if self.literature_sources:
                citation_insight = self._analyze_citation_impact()
                insights.append(citation_insight)
            
            # Insight 5: Research Gaps Identification
            if self.literature_sources:
                gaps_insight = self._identify_research_gaps(topic)
                insights.append(gaps_insight)
            
            # Insight 6: Quality Distribution Analysis
            if self.literature_sources:
                quality_insight = self._analyze_quality_distribution()
                insights.append(quality_insight)
            
        except Exception as e:
            logger.warning(f"Dynamic insight generation failed: {e}, using fallback insights")
            # Fallback to basic insights if dynamic generation fails
            insights = self._create_fallback_insights()
        
        return insights
    
    def _analyze_research_coverage(self, topic: str) -> DataInsight:
        """Analyze research coverage and comprehensiveness"""
        try:
            # Analyze source diversity and coverage
            total_sources = len(self.literature_sources)
            high_relevance_sources = len([s for s in self.literature_sources if s.relevance_score >= 8.0])
            coverage_percentage = (high_relevance_sources / total_sources * 100) if total_sources > 0 else 0
            
            # Determine significance based on coverage
            if coverage_percentage >= 80:
                significance = "High"
                confidence = 0.9
            elif coverage_percentage >= 60:
                significance = "Medium"
                confidence = 0.75
            else:
                significance = "Low"
                confidence = 0.6
            
            return DataInsight(
                category="Research Coverage",
                value=f"{coverage_percentage:.1f}% of sources have high relevance (≥8.0/10)",
                significance=significance,
                source=f"Analysis of {total_sources} literature sources",
                confidence=confidence,
                trend="Comprehensive" if coverage_percentage >= 70 else "Limited"
            )
        except Exception as e:
            logger.warning(f"Coverage analysis failed: {e}")
            return self._create_default_insight("Research Coverage", "Analysis unavailable", "Low", 0.5)
    
    def _analyze_temporal_trends(self) -> DataInsight:
        """Analyze temporal distribution and trends in research"""
        try:
            years = [s.year for s in self.literature_sources if s.year]
            if not years:
                return self._create_default_insight("Temporal Trends", "Year data unavailable", "Low", 0.5)
            
            current_year = datetime.now().year
            recent_sources = len([y for y in years if y >= current_year - 3])
            old_sources = len([y for y in years if y < current_year - 10])
            
            recent_percentage = (recent_sources / len(years)) * 100
            old_percentage = (old_sources / len(years)) * 100
            
            if recent_percentage >= 60:
                trend = "Current"
                significance = "High"
                confidence = 0.85
            elif recent_percentage >= 40:
                trend = "Recent"
                significance = "Medium"
                confidence = 0.7
            else:
                trend = "Outdated"
                significance = "Low"
                confidence = 0.6
            
            return DataInsight(
                category="Temporal Trends",
                value=f"{recent_percentage:.1f}% of sources are from last 3 years, {old_percentage:.1f}% are over 10 years old",
                significance=significance,
                source=f"Analysis of {len(years)} dated sources",
                confidence=confidence,
                trend=trend
            )
        except Exception as e:
            logger.warning(f"Temporal analysis failed: {e}")
            return self._create_default_insight("Temporal Trends", "Analysis unavailable", "Low", 0.5)
    
    def _analyze_methodological_patterns(self) -> DataInsight:
        """Analyze methodological approaches and patterns"""
        try:
            # Analyze source types and methodologies
            high_quality_sources = [s for s in self.literature_sources if s.relevance_score >= 7.0]
            
            if not high_quality_sources:
                return self._create_default_insight("Methodological Patterns", "No high-quality sources available", "Low", 0.5)
            
            # Analyze citation patterns as proxy for methodology quality
            citation_counts = [s.citation_count for s in high_quality_sources if s.citation_count > 0]
            
            if citation_counts:
                avg_citations = sum(citation_counts) / len(citation_counts)
                max_citations = max(citation_counts)
                
                if avg_citations >= 100:
                    methodology_quality = "High-impact"
                    significance = "High"
                    confidence = 0.9
                elif avg_citations >= 50:
                    methodology_quality = "Well-cited"
                    significance = "Medium"
                    confidence = 0.75
                else:
                    methodology_quality = "Emerging"
                    significance = "Medium"
                    confidence = 0.65
                
                return DataInsight(
                    category="Methodological Patterns",
                    value=f"High-quality sources average {avg_citations:.1f} citations (max: {max_citations})",
                    significance=significance,
                    source=f"Analysis of {len(high_quality_sources)} high-quality sources",
                    confidence=confidence,
                    trend=methodology_quality
                )
            else:
                return self._create_default_insight("Methodological Patterns", "Citation data unavailable", "Medium", 0.6)
                
        except Exception as e:
            logger.warning(f"Methodological analysis failed: {e}")
            return self._create_default_insight("Methodological Patterns", "Analysis unavailable", "Low", 0.5)
    
    def _analyze_citation_impact(self) -> DataInsight:
        """Analyze citation impact and influence patterns"""
        try:
            citation_counts = [s.citation_count for s in self.literature_sources if s.citation_count > 0]
            
            if not citation_counts:
                return self._create_default_insight("Citation Impact", "No citation data available", "Low", 0.5)
            
            total_citations = sum(citation_counts)
            avg_citations = total_citations / len(citation_counts)
            highly_cited = len([c for c in citation_counts if c >= 100])
            
            if highly_cited >= len(citation_counts) * 0.3:
                impact_level = "High-impact"
                significance = "High"
                confidence = 0.9
            elif highly_cited >= len(citation_counts) * 0.1:
                impact_level = "Moderate-impact"
                significance = "Medium"
                confidence = 0.75
            else:
                impact_level = "Emerging"
                significance = "Medium"
                confidence = 0.65
            
            return DataInsight(
                category="Citation Impact",
                value=f"Total {total_citations:,} citations, {highly_cited} highly-cited sources (≥100 citations)",
                significance=significance,
                source=f"Analysis of {len(citation_counts)} cited sources",
                confidence=confidence,
                trend=impact_level
            )
        except Exception as e:
            logger.warning(f"Citation analysis failed: {e}")
            return self._create_default_insight("Citation Impact", "Analysis unavailable", "Low", 0.5)
    
    def _identify_research_gaps(self, topic: str) -> DataInsight:
        """Identify potential research gaps and opportunities"""
        try:
            # Analyze relevance score distribution to identify gaps
            relevance_scores = [s.relevance_score for s in self.literature_sources]
            
            if not relevance_scores:
                return self._create_default_insight("Research Gaps", "No relevance data available", "Low", 0.5)
            
            low_relevance = len([s for s in relevance_scores if s < 5.0])
            high_relevance = len([s for s in relevance_scores if s >= 8.0])
            
            if low_relevance > high_relevance:
                gap_status = "Significant gaps identified"
                significance = "High"
                confidence = 0.8
                trend = "Research needed"
            elif low_relevance > 0:
                gap_status = "Some gaps identified"
                significance = "Medium"
                confidence = 0.7
                trend = "Partial coverage"
            else:
                gap_status = "Well-covered topic"
                significance = "Medium"
                confidence = 0.75
                trend = "Comprehensive"
            
            return DataInsight(
                category="Research Gaps",
                value=f"{low_relevance} low-relevance sources vs {high_relevance} high-relevance sources",
                significance=significance,
                source=f"Relevance analysis of {len(relevance_scores)} sources",
                confidence=confidence,
                trend=trend
            )
        except Exception as e:
            logger.warning(f"Gap analysis failed: {e}")
            return self._create_default_insight("Research Gaps", "Analysis unavailable", "Low", 0.5)
    
    def _analyze_quality_distribution(self) -> DataInsight:
        """Analyze overall quality distribution of sources"""
        try:
            relevance_scores = [s.relevance_score for s in self.literature_sources]
            
            if not relevance_scores:
                return self._create_default_insight("Quality Distribution", "No quality data available", "Low", 0.5)
            
            avg_quality = sum(relevance_scores) / len(relevance_scores)
            quality_std = (sum((x - avg_quality) ** 2 for x in relevance_scores) / len(relevance_scores)) ** 0.5
            
            if avg_quality >= 7.5:
                quality_level = "High-quality"
                significance = "High"
                confidence = 0.9
            elif avg_quality >= 6.0:
                quality_level = "Good-quality"
                significance = "Medium"
                confidence = 0.75
            else:
                quality_level = "Variable-quality"
                significance = "Medium"
                confidence = 0.65
            
            return DataInsight(
                category="Quality Distribution",
                value=f"Average quality: {avg_quality:.1f}/10 (std: {quality_std:.1f})",
                significance=significance,
                source=f"Quality analysis of {len(relevance_scores)} sources",
                confidence=confidence,
                trend=quality_level
            )
        except Exception as e:
            logger.warning(f"Quality distribution analysis failed: {e}")
            return self._create_default_insight("Quality Distribution", "Analysis unavailable", "Low", 0.5)
    
    def _create_default_insight(self, category: str, value: str, significance: str, confidence: float) -> DataInsight:
        """Create a default insight when analysis fails"""
        return DataInsight(
            category=category,
            value=value,
            significance=significance,
            source="Fallback analysis",
            confidence=confidence,
            trend="Unknown"
        )
    
    def _create_fallback_insights(self) -> List[DataInsight]:
        """Create fallback insights when dynamic generation fails"""
        return [
            DataInsight(
                category="Research Coverage",
                value="Analysis temporarily unavailable",
                significance="Low",
                source="Fallback system",
                confidence=0.3,
                trend="Unknown"
            ),
            DataInsight(
                category="Data Quality",
                value="Using fallback data generation",
                significance="Low",
                source="Fallback system",
                confidence=0.3,
                trend="Unknown"
            )
        ]
    
    def writer_agent(self, topic: str) -> str:
        """Writer Agent - Advanced research paper generation"""
        try:
            # Create comprehensive paper outline
            outline = self._create_paper_outline(topic)
            
            # Generate paper content
            writing_prompt = self._create_writing_prompt(topic, outline)
            paper_response = self.llm.invoke([HumanMessage(content=writing_prompt)])
            self.paper_draft = paper_response.content
            
            # Update metrics
            self.metrics.paper_length = len(self.paper_draft)
            self.completion_percentage = 75.0
            
            # Calculate comprehensive metrics after paper generation
            self._calculate_comprehensive_metrics()
            
            # Update quality score after paper generation
            self.metrics.quality_score = self._calculate_quality_score()
            
            message = f"✍️ Writer Agent: Research paper draft complete. {len(self.paper_draft)} characters written."
            self.agent_messages.append(message)
            
            return message
            
        except Exception as e:
            error_msg = f"Writer Agent failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _create_paper_outline(self, topic: str) -> str:
        """Create comprehensive paper outline"""
        return f"""
1. Executive Summary
   - Research overview and key findings
   - Methodology highlights
   - Main conclusions

2. Introduction
   - Background and context
   - Research objectives
   - Scope and limitations

3. Literature Review
   - Current state of knowledge
   - Key theoretical frameworks
   - Research gaps identified

4. Methodology
   - Research design
   - Data collection methods
   - Analysis framework

5. Results and Analysis
   - Key findings
   - Statistical analysis
   - Data interpretation

6. Discussion
   - Implications of findings
   - Comparison with existing research
   - Theoretical contributions

7. Conclusion
   - Summary of contributions
   - Practical implications
   - Future research directions

8. References
   - Academic sources
   - Data sources
   - Additional resources
"""
    
    def _create_writing_prompt(self, topic: str, outline: str) -> str:
        """Create comprehensive writing prompt"""
        return f"""Write a comprehensive, academic research paper on: {topic}

Paper Outline:
{outline}

Literature Sources: {len(self.literature_sources)} sources analyzed
Data Insights: {len(self.data_insights)} key insights

Requirements:
1. Follow academic writing standards
2. Integrate all literature sources with proper citations
3. Include all data insights with analysis
4. Maintain logical flow and structure
5. Target length: {self.config['paper_target_length']} words
6. Use clear, professional language
7. Include relevant statistics and examples

Create a well-structured, comprehensive research paper that demonstrates deep understanding of the topic and provides valuable insights."""
    
    def editor_agent(self, topic: str) -> str:
        """Editor Agent - Advanced editing and quality assurance"""
        try:
            # Quality check and refinement
            quality_prompt = f"""You are an expert academic editor. Review and improve this research paper:

Paper: {self.paper_draft}

Your task is to:
1. Fix any grammar, spelling, and punctuation errors
2. Improve academic writing style and tone
3. Enhance logical structure and flow
4. Ensure citation accuracy and formatting
5. Verify data consistency and accuracy
6. Improve clarity and readability
7. Apply professional formatting

IMPORTANT: Return ONLY the improved, final version of the paper. Do NOT include any quality check reports, feedback, or analysis. Return the complete, polished research paper as the final output."""

            final_response = self.llm.invoke([HumanMessage(content=quality_prompt)])
            
            # Use the improved paper content directly
            improved_paper = final_response.content.strip()
            
            # Fallback: if the LLM didn't return the paper content properly, use the original draft
            if len(improved_paper) < len(self.paper_draft) * 0.5:  # If response is too short
                improved_paper = self.paper_draft
                logger.warning("LLM response too short, using original paper draft")
            
            # Create final formatted paper
            self.final_paper = self._format_final_paper(topic, improved_paper)
            
            # Calculate quality score
            self.metrics.quality_score = self._calculate_quality_score()
            
            # Update completion
            self.completion_percentage = 100.0
            self.current_phase = ResearchPhase.COMPLETE
            self.end_time = datetime.now()
            
            if self.start_time and self.end_time:
                self.metrics.completion_time = (self.end_time - self.start_time).total_seconds()
            
            message = f"✏️ Editor Agent: Final paper complete! Quality score: {self.metrics.quality_score:.1f}/10"
            self.agent_messages.append(message)
            
            return message
            
        except Exception as e:
            error_msg = f"Editor Agent failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _format_final_paper(self, topic: str, content: str) -> str:
        """Format the final research paper"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Clean up the content - remove any quality check language
        cleaned_content = content
        if "quality check" in content.lower() or "comprehensive quality check report" in content:
            # Extract only the paper content, not the quality report
            if "**Polished Version**" in content:
                parts = content.split("**Polished Version**")
                if len(parts) > 1:
                    cleaned_content = parts[1].strip()
            elif "Polished Version" in content:
                parts = content.split("Polished Version")
                if len(parts) > 1:
                    cleaned_content = parts[1].strip()
        
        return f"""
{'='*80}
        📄 AGENTIC RESEARCH PAPER
{'='*80}
Topic: {topic}
Generated: {timestamp}
Completion Time: {self.metrics.completion_time:.1f} seconds
Quality Score: {self.metrics.quality_score:.1f}/10
{'='*80}

{cleaned_content}

{'='*80}
📊 RESEARCH METRICS
{'='*80}
• Literature Sources: {self.metrics.total_sources}
• Average Relevance: {self.metrics.avg_relevance:.1f}/10
• Data Insights: {self.metrics.data_points}
• Paper Length: {len(cleaned_content):,} characters
• Completion Time: {self.metrics.completion_time:.1f} seconds

{'='*80}
        Generated by Agentic Research Assistant
Powered by Groq LLM and Advanced AI Agents
{'='*80}
"""
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall quality score with agentic metrics"""
        scores = []
        weights = []
        
        # Source quality (weighted by relevance distribution)
        if self.literature_sources:
            relevance_scores = [s.relevance_score for s in self.literature_sources]
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            
            # Bonus for consistency (low standard deviation indicates consistent quality)
            if len(relevance_scores) > 1:
                std_dev = (sum((x - avg_relevance) ** 2 for x in relevance_scores) / len(relevance_scores)) ** 0.5
                consistency_bonus = max(0, (2.0 - std_dev) / 2.0)  # Up to 1.0 bonus for low std dev
                source_score = min(10.0, avg_relevance + consistency_bonus)
            else:
                source_score = min(10.0, avg_relevance)
            
            scores.append(source_score)
            weights.append(3.0)  # Higher weight for source quality
        
        # Data quality (confidence-weighted insights)
        if self.data_insights:
            # Calculate weighted average of insight confidence and significance
            insight_scores = []
            for insight in self.data_insights:
                # Convert significance to numeric score
                significance_score = {"High": 1.0, "Medium": 0.7, "Low": 0.4}.get(insight.significance, 0.5)
                # Combine confidence and significance
                insight_score = (insight.confidence + significance_score) / 2 * 10
                insight_scores.append(insight_score)
            
            if insight_scores:
                avg_insight_score = sum(insight_scores) / len(insight_scores)
                # Bonus for having multiple high-quality insights
                high_quality_insights = len([s for s in insight_scores if s >= 7.0])
                diversity_bonus = min(1.0, high_quality_insights / len(insight_scores))
                
                data_score = min(10.0, avg_insight_score + diversity_bonus)
                scores.append(data_score)
                weights.append(2.5)  # High weight for data quality
        
        # Paper quality (agentic length and content analysis)
        if self.paper_draft:
            # Base length score
            length_score = min(8.0, len(self.paper_draft) / 2000)  # 16k chars = 8.0 points
            
            # Content quality indicators (simple heuristics)
            content_indicators = 0
            if "introduction" in self.paper_draft.lower():
                content_indicators += 0.5
            if "method" in self.paper_draft.lower() or "methodology" in self.paper_draft.lower():
                content_indicators += 0.5
            if "conclusion" in self.paper_draft.lower():
                content_indicators += 0.5
            if "reference" in self.paper_draft.lower() or "bibliography" in self.paper_draft.lower():
                content_indicators += 0.5
            
            paper_score = min(10.0, length_score + content_indicators)
            scores.append(paper_score)
            weights.append(2.0)  # Medium weight for paper quality
        
        # Completion quality (with phase progression bonus)
        completion_base = self.completion_percentage / 10
        # Bonus for completing more phases
        phase_bonus = min(1.0, self.completion_percentage / 100 * 2)  # Up to 1.0 bonus for 100% completion
        completion_score = min(10.0, completion_base + phase_bonus)
        scores.append(completion_score)
        weights.append(1.5)  # Lower weight for completion
        
        # Research depth bonus (based on source diversity and analysis depth)
        if self.literature_sources and self.data_insights:
            source_diversity = min(1.0, len(self.literature_sources) / 10)  # Bonus for more sources
            insight_depth = min(1.0, len(self.data_insights) / 6)  # Bonus for more insights
            depth_bonus = (source_diversity + insight_depth) / 2
            
            if depth_bonus > 0:
                scores.append(depth_bonus * 10)
                weights.append(1.0)  # Lower weight for depth bonus
        
        # Calculate weighted average
        if scores and weights:
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            total_weight = sum(weights)
            return weighted_sum / total_weight
        elif scores:
            return sum(scores) / len(scores)
        else:
            return 0.0
    
    def get_quality_analysis(self) -> Dict[str, Any]:
        """Get detailed breakdown of quality score components"""
        analysis = {
            "overall_score": self.metrics.quality_score,
            "components": {},
            "recommendations": []
        }
        
        # Source quality analysis
        if self.literature_sources:
            relevance_scores = [s.relevance_score for s in self.literature_sources]
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            std_dev = (sum((x - avg_relevance) ** 2 for x in relevance_scores) / len(relevance_scores)) ** 0.5 if len(relevance_scores) > 1 else 0
            
            analysis["components"]["source_quality"] = {
                "score": avg_relevance,
                "consistency_bonus": max(0, (2.0 - std_dev) / 2.0),
                "total_sources": len(self.literature_sources),
                "high_quality_sources": len([s for s in relevance_scores if s >= 8.0]),
                "std_deviation": std_dev
            }
            
            if avg_relevance < 7.0:
                analysis["recommendations"].append("Consider adding more relevant sources to improve source quality")
            if std_dev > 1.5:
                analysis["recommendations"].append("Source quality varies significantly - focus on consistent high-quality sources")
        
        # Data insights analysis
        if self.data_insights:
            insight_scores = []
            for insight in self.data_insights:
                significance_score = {"High": 1.0, "Medium": 0.7, "Low": 0.4}.get(insight.significance, 0.5)
                insight_score = (insight.confidence + significance_score) / 2 * 10
                insight_scores.append(insight_score)
            
            avg_insight_score = sum(insight_scores) / len(insight_scores) if insight_scores else 0
            high_quality_insights = len([s for s in insight_scores if s >= 7.0])
            
            analysis["components"]["data_quality"] = {
                "score": avg_insight_score,
                "total_insights": len(self.data_insights),
                "high_quality_insights": high_quality_insights,
                "diversity_bonus": min(1.0, high_quality_insights / len(insight_scores)) if insight_scores else 0
            }
            
            if avg_insight_score < 6.0:
                analysis["recommendations"].append("Data insights could be improved with higher confidence and significance")
            if len(self.data_insights) < 4:
                analysis["recommendations"].append("Consider generating more diverse data insights for comprehensive analysis")
        
        # Paper quality analysis
        if self.paper_draft:
            length_score = min(8.0, len(self.paper_draft) / 2000)
            content_indicators = 0
            if "introduction" in self.paper_draft.lower():
                content_indicators += 0.5
            if "method" in self.paper_draft.lower() or "methodology" in self.paper_draft.lower():
                content_indicators += 0.5
            if "conclusion" in self.paper_draft.lower():
                content_indicators += 0.5
            if "reference" in self.paper_draft.lower() or "bibliography" in self.paper_draft.lower():
                content_indicators += 0.5
            
            analysis["components"]["paper_quality"] = {
                "score": min(10.0, length_score + content_indicators),
                "length_score": length_score,
                "content_indicators": content_indicators,
                "paper_length": len(self.paper_draft),
                "target_length": 16000  # 16k chars for full score
            }
            
            if length_score < 6.0:
                analysis["recommendations"].append("Paper length could be increased for more comprehensive coverage")
            if content_indicators < 1.5:
                analysis["recommendations"].append("Consider adding missing paper sections (introduction, methods, conclusion, references)")
        
        # Completion analysis
        completion_base = self.completion_percentage / 10
        phase_bonus = min(1.0, self.completion_percentage / 100 * 2)
        
        analysis["components"]["completion_quality"] = {
            "score": min(10.0, completion_base + phase_bonus),
            "completion_percentage": self.completion_percentage,
            "phase_bonus": phase_bonus
        }
        
        if self.completion_percentage < 100:
            analysis["recommendations"].append(f"Complete remaining research phases to improve quality score")
        
        # Research depth analysis
        if self.literature_sources and self.data_insights:
            source_diversity = min(1.0, len(self.literature_sources) / 10)
            insight_depth = min(1.0, len(self.data_insights) / 6)
            depth_bonus = (source_diversity + insight_depth) / 2
            
            analysis["components"]["research_depth"] = {
                "score": depth_bonus * 10,
                "source_diversity": source_diversity,
                "insight_depth": insight_depth
            }
            
            if source_diversity < 0.7:
                analysis["recommendations"].append("Consider adding more diverse literature sources")
            if insight_depth < 0.7:
                analysis["recommendations"].append("Generate more comprehensive data insights for deeper analysis")
        
        return analysis
    
    def run_research(self, topic: str) -> Dict[str, Any]:
        """Run the complete agentic research workflow"""
        try:
            logger.info(f"🚀 Starting agentic research on: {topic}")
            print(f"🚀 Starting Agentic Research on: {topic}")
            print("=" * 80)
            
            # Phase 1: Planning
            print("📋 Phase 1: Strategic Planning...")
            self.research_director(topic)
            print(f"✅ {self.agent_messages[-1]}")
            
            # Phase 2: Literature Review
            print("🔍 Phase 2: Advanced Literature Review...")
            self.literature_agent(topic)
            print(f"✅ {self.agent_messages[-1]}")
            
            # Phase 3: Data Analysis
            print("📊 Phase 3: Comprehensive Data Analysis...")
            self.data_agent(topic)
            print(f"✅ {self.agent_messages[-1]}")
            
            # Phase 4: Writing
            print("✍️ Phase 4: Advanced Paper Writing...")
            self.writer_agent(topic)
            print(f"✅ {self.agent_messages[-1]}")
            
            # Phase 5: Editing
            print("✏️ Phase 5: Quality Assurance & Editing...")
            self.editor_agent(topic)
            print(f"✅ {self.agent_messages[-1]}")
            
            print("\n🎉 Agentic Research Complete!")
            print(f"📊 Final Completion: {self.completion_percentage}%")
            print(f"📝 Paper Length: {self.metrics.paper_length:,} characters")
            print(f"⭐ Quality Score: {self.metrics.quality_score:.1f}/10")
            print(f"⏱️ Completion Time: {self.metrics.completion_time:.1f} seconds")
            
            # Final comprehensive metrics calculation
            self._calculate_comprehensive_metrics()
            self.metrics.quality_score = self._calculate_quality_score()
            
            # Return comprehensive results
            return {
                "topic": topic,
                "completion_percentage": self.completion_percentage,
                "current_phase": self.current_phase.value,
                "literature_sources": [asdict(source) for source in self.literature_sources],
                "data_insights": [asdict(insight) for insight in self.data_insights],
                "paper_draft": self.paper_draft,
                "final_paper": self.final_paper,
                "agent_messages": self.agent_messages,
                "research_notes": self.research_notes,
                "metrics": asdict(self.metrics),
                "ui_metrics": self.get_ui_metrics(),
                "metric_summary": self.get_metric_summary(),
                "quality_analysis": self.get_quality_analysis(),
                "timestamp": datetime.now().isoformat(),
                "config": self.config
            }
            
        except Exception as e:
            error_msg = f"Agentic research failed: {e}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            return {
                "error": str(e),
                "topic": topic,
                "completion_percentage": self.completion_percentage,
                "current_phase": self.current_phase.value,
                "timestamp": datetime.now().isoformat()
            }
    
    def export_results(self, format_type: str = "json") -> str:
        """Export research results in various formats"""
        if not self.final_paper:
            raise ValueError("No research results to export")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format_type == "json":
            filename = f"agentic_research_results_{timestamp}.json"
            results = {
                "topic": "Research Topic",
                "completion_percentage": self.completion_percentage,
                "current_phase": self.current_phase.value,
                "literature_sources": [asdict(source) for source in self.literature_sources],
                "data_insights": [asdict(insight) for insight in self.data_insights],
                "paper_draft": self.paper_draft,
                "final_paper": self.final_paper,
                "metrics": asdict(self.metrics),
                "ui_metrics": self.get_ui_metrics(),
                "metric_summary": self.get_metric_summary(),
                "timestamp": timestamp
            }
            
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)
            
            return filename
        
        elif format_type == "txt":
            filename = f"research_paper_{timestamp}.txt"
            with open(filename, "w") as f:
                f.write(self.final_paper)
            return filename
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _calculate_comprehensive_metrics(self):
        """Calculate all comprehensive metrics for UI display"""
        try:
            logger.info("Calculating comprehensive metrics...")
            logger.info(f"Literature sources: {len(self.literature_sources)}")
            logger.info(f"Data insights: {len(self.data_insights)}")
            logger.info(f"Paper draft length: {len(self.paper_draft)}")
            # Source quality metrics
            if self.literature_sources:
                relevance_scores = [s.relevance_score for s in self.literature_sources]
                
                # Relevance distribution
                self.metrics.high_relevance_sources = len([s for s in relevance_scores if s >= 8.0])
                self.metrics.medium_relevance_sources = len([s for s in relevance_scores if 5.0 <= s < 8.0])
                self.metrics.low_relevance_sources = len([s for s in relevance_scores if s < 5.0])
                
                # Relevance consistency (standard deviation)
                if len(relevance_scores) > 1:
                    mean_relevance = sum(relevance_scores) / len(relevance_scores)
                    variance = sum((x - mean_relevance) ** 2 for x in relevance_scores) / len(relevance_scores)
                    self.metrics.relevance_consistency = variance ** 0.5
                else:
                    self.metrics.relevance_consistency = 0.0
                
                # Source diversity score (0-10)
                unique_journals = len(set(s.journal for s in self.literature_sources))
                unique_years = len(set(s.year for s in self.literature_sources))
                self.metrics.source_diversity_score = min(10.0, (unique_journals + unique_years) / 2)
                
                # Citation metrics
                citation_counts = [s.citation_count for s in self.literature_sources if s.citation_count > 0]
                if citation_counts:
                    self.metrics.total_citations = sum(citation_counts)
                    self.metrics.avg_citations_per_source = sum(citation_counts) / len(citation_counts)
                    self.metrics.highly_cited_sources = len([c for c in citation_counts if c >= 100])
                    
                    # Citation impact score (0-10)
                    if self.metrics.avg_citations_per_source >= 100:
                        self.metrics.citation_impact_score = 10.0
                    elif self.metrics.avg_citations_per_source >= 50:
                        self.metrics.citation_impact_score = 8.0
                    elif self.metrics.avg_citations_per_source >= 20:
                        self.metrics.citation_impact_score = 6.0
                    elif self.metrics.avg_citations_per_source >= 10:
                        self.metrics.citation_impact_score = 4.0
                    else:
                        self.metrics.citation_impact_score = 2.0
                else:
                    self.metrics.citation_impact_score = 0.0
                
                # Temporal metrics
                current_year = datetime.now().year
                years = [s.year for s in self.literature_sources if s.year]
                if years:
                    self.metrics.recent_sources = len([y for y in years if y >= current_year - 3])
                    self.metrics.current_sources = len([y for y in years if y >= current_year - 5])
                    self.metrics.outdated_sources = len([y for y in years if y < current_year - 10])
                    
                    # Temporal freshness score (0-10)
                    recent_ratio = self.metrics.recent_sources / len(years) if years else 0
                    if recent_ratio >= 0.7:
                        self.metrics.temporal_freshness_score = 10.0
                    elif recent_ratio >= 0.5:
                        self.metrics.temporal_freshness_score = 8.0
                    elif recent_ratio >= 0.3:
                        self.metrics.temporal_freshness_score = 6.0
                    elif recent_ratio >= 0.1:
                        self.metrics.temporal_freshness_score = 4.0
                    else:
                        self.metrics.temporal_freshness_score = 2.0
                else:
                    self.metrics.temporal_freshness_score = 0.0
            
            # Data insight metrics
            if self.data_insights:
                # Confidence distribution
                self.metrics.high_confidence_insights = len([i for i in self.data_insights if i.confidence >= 0.8])
                self.metrics.medium_confidence_insights = len([i for i in self.data_insights if 0.6 <= i.confidence < 0.8])
                self.metrics.low_confidence_insights = len([i for i in self.data_insights if i.confidence < 0.6])
                
                # Insight significance score (0-10)
                significance_scores = []
                for insight in self.data_insights:
                    if insight.significance == "High":
                        significance_scores.append(1.0)
                    elif insight.significance == "Medium":
                        significance_scores.append(0.7)
                    else:
                        significance_scores.append(0.4)
                
                if significance_scores:
                    self.metrics.insight_significance_score = sum(significance_scores) / len(significance_scores) * 10
                else:
                    self.metrics.insight_significance_score = 0.0
                
                # Insight diversity score (0-10)
                unique_categories = len(set(i.category for i in self.data_insights))
                self.metrics.insight_diversity_score = min(10.0, unique_categories * 1.5)
            
            # Paper quality metrics
            if self.paper_draft:
                # Content structure score (0-10)
                content_indicators = 0
                if "introduction" in self.paper_draft.lower():
                    content_indicators += 2.0
                if "method" in self.paper_draft.lower() or "methodology" in self.paper_draft.lower():
                    content_indicators += 2.0
                if "conclusion" in self.paper_draft.lower():
                    content_indicators += 2.0
                if "reference" in self.paper_draft.lower() or "bibliography" in self.paper_draft.lower():
                    content_indicators += 2.0
                if "abstract" in self.paper_draft.lower():
                    content_indicators += 1.0
                if "discussion" in self.paper_draft.lower():
                    content_indicators += 1.0
                
                self.metrics.content_structure_score = min(10.0, content_indicators)
                
                # Length adequacy score (0-10)
                target_length = self.config.get('paper_target_length', 15000)
                length_ratio = len(self.paper_draft) / target_length
                if length_ratio >= 1.0:
                    self.metrics.length_adequacy_score = 10.0
                elif length_ratio >= 0.8:
                    self.metrics.length_adequacy_score = 8.0
                elif length_ratio >= 0.6:
                    self.metrics.length_adequacy_score = 6.0
                elif length_ratio >= 0.4:
                    self.metrics.length_adequacy_score = 4.0
                else:
                    self.metrics.length_adequacy_score = 2.0
                
                # Writing quality score (combined structure and length)
                self.metrics.writing_quality_score = (self.metrics.content_structure_score + self.metrics.length_adequacy_score) / 2
            
            # Research depth metrics
            if self.literature_sources:
                # Methodology coverage score (0-10)
                methodology_keywords = ['method', 'methodology', 'approach', 'technique', 'procedure', 'protocol']
                methodology_mentions = sum(1 for source in self.literature_sources 
                                        for keyword in methodology_keywords 
                                        if keyword in source.summary.lower())
                self.metrics.methodology_coverage_score = min(10.0, methodology_mentions * 2)
                
                # Theoretical framework score (0-10)
                framework_keywords = ['theory', 'theoretical', 'framework', 'model', 'concept', 'hypothesis']
                framework_mentions = sum(1 for source in self.literature_sources 
                                       for keyword in framework_keywords 
                                       if keyword in source.summary.lower())
                self.metrics.theoretical_framework_score = min(10.0, framework_mentions * 2)
                
                # Empirical evidence score (0-10)
                evidence_keywords = ['data', 'evidence', 'results', 'findings', 'analysis', 'statistics']
                evidence_mentions = sum(1 for source in self.literature_sources 
                                      for keyword in evidence_keywords 
                                      if keyword in source.summary.lower())
                self.metrics.empirical_evidence_score = min(10.0, evidence_mentions * 2)
            
            # Efficiency metrics
            if self.start_time and self.end_time:
                # Research efficiency score (0-10) - quality vs. time
                time_factor = min(1.0, 300 / self.metrics.completion_time) if self.metrics.completion_time > 0 else 0
                quality_factor = self.metrics.quality_score / 10.0
                self.metrics.research_efficiency_score = (time_factor + quality_factor) / 2 * 10
            
            # Phase completion scores
            self.metrics.phase_completion_scores = {
                "planning": 10.0 if self.completion_percentage >= 10 else self.completion_percentage,
                "research": 30.0 if self.completion_percentage >= 30 else max(0, self.completion_percentage - 10),
                "analysis": 50.0 if self.completion_percentage >= 50 else max(0, self.completion_percentage - 30),
                "writing": 75.0 if self.completion_percentage >= 75 else max(0, self.completion_percentage - 50),
                "editing": 100.0 if self.completion_percentage >= 100 else max(0, self.completion_percentage - 75)
            }
            
            # Overall assessment scores
            # Comprehensiveness score (0-10)
            coverage_factors = [
                min(10.0, len(self.literature_sources) / 2),  # Source quantity
                self.metrics.source_diversity_score,  # Source diversity
                min(10.0, len(self.data_insights) * 1.5),  # Insight quantity
                self.metrics.insight_diversity_score  # Insight diversity
            ]
            self.metrics.comprehensiveness_score = sum(coverage_factors) / len(coverage_factors)
            
            # Rigor score (0-10)
            rigor_factors = [
                self.metrics.avg_relevance,  # Source relevance
                self.metrics.citation_impact_score,  # Citation impact
                self.metrics.methodology_coverage_score,  # Methodology coverage
                self.metrics.theoretical_framework_score  # Theoretical framework
            ]
            self.metrics.rigor_score = sum(rigor_factors) / len(rigor_factors)
            
            # Innovation score (0-10) - based on recent sources and unique insights
            innovation_factors = [
                self.metrics.temporal_freshness_score,  # Recent sources
                min(10.0, len(set(i.category for i in self.data_insights)) * 1.5),  # Unique insight categories
                min(10.0, len(set(s.journal for s in self.literature_sources)) / 2)  # Journal diversity
            ]
            self.metrics.innovation_score = sum(innovation_factors) / len(innovation_factors)
            
            # Impact potential score (0-10)
            impact_factors = [
                self.metrics.citation_impact_score,  # Citation patterns
                self.metrics.insight_significance_score / 10,  # Insight significance
                self.metrics.rigor_score / 10,  # Research rigor
                self.metrics.comprehensiveness_score / 10  # Research coverage
            ]
            self.metrics.impact_potential_score = sum(impact_factors) / len(impact_factors) * 10
            
            logger.info("Comprehensive metrics calculation completed successfully")
            logger.info(f"Final scores - Quality: {self.metrics.quality_score:.1f}, Comprehensiveness: {self.metrics.comprehensiveness_score:.1f}, Temporal: {self.metrics.temporal_freshness_score:.1f}")
            
        except Exception as e:
            logger.warning(f"Comprehensive metrics calculation failed: {e}")
            # Keep existing metrics if calculation fails

    def get_ui_metrics(self) -> Dict[str, Any]:
        """Get all metrics organized for UI display"""
        return {
            "overview": {
                "quality_score": round(self.metrics.quality_score, 1),
                "completion_percentage": self.completion_percentage,
                "total_sources": self.metrics.total_sources,
                "data_points": self.metrics.data_points,
                "paper_length": self.metrics.paper_length,
                "completion_time": round(self.metrics.completion_time, 1) if self.metrics.completion_time > 0 else 0
            },
            "source_quality": {
                "avg_relevance": round(self.metrics.avg_relevance, 1),
                "high_relevance_sources": self.metrics.high_relevance_sources,
                "medium_relevance_sources": self.metrics.medium_relevance_sources,
                "low_relevance_sources": self.metrics.low_relevance_sources,
                "relevance_consistency": round(self.metrics.relevance_consistency, 2),
                "source_diversity_score": round(self.metrics.source_diversity_score, 1)
            },
            "citations": {
                "total_citations": self.metrics.total_citations,
                "avg_citations_per_source": round(self.metrics.avg_citations_per_source, 1),
                "highly_cited_sources": self.metrics.highly_cited_sources,
                "citation_impact_score": round(self.metrics.citation_impact_score, 1)
            },
            "temporal": {
                "recent_sources": self.metrics.recent_sources,
                "current_sources": self.metrics.current_sources,
                "outdated_sources": self.metrics.outdated_sources,
                "temporal_freshness_score": round(self.metrics.temporal_freshness_score, 1)
            },
            "insights": {
                "high_confidence_insights": self.metrics.high_confidence_insights,
                "medium_confidence_insights": self.metrics.medium_confidence_insights,
                "low_confidence_insights": self.metrics.low_confidence_insights,
                "insight_significance_score": round(self.metrics.insight_significance_score, 1),
                "insight_diversity_score": round(self.metrics.insight_diversity_score, 1)
            },
            "paper_quality": {
                "content_structure_score": round(self.metrics.content_structure_score, 1),
                "length_adequacy_score": round(self.metrics.length_adequacy_score, 1),
                "writing_quality_score": round(self.metrics.writing_quality_score, 1)
            },
            "research_depth": {
                "methodology_coverage_score": round(self.metrics.methodology_coverage_score, 1),
                "theoretical_framework_score": round(self.metrics.theoretical_framework_score, 1),
                "empirical_evidence_score": round(self.metrics.empirical_evidence_score, 1)
            },
            "efficiency": {
                "research_efficiency_score": round(self.metrics.research_efficiency_score, 1),
                "phase_completion_scores": self.metrics.phase_completion_scores
            },
            "assessment": {
                "comprehensiveness_score": round(self.metrics.comprehensiveness_score, 1),
                "rigor_score": round(self.metrics.rigor_score, 1),
                "innovation_score": round(self.metrics.innovation_score, 1),
                "impact_potential_score": round(self.metrics.impact_potential_score, 1)
            }
        }

    def get_metric_summary(self) -> Dict[str, Any]:
        """Get quick metric summary for dashboard display"""
        # Ensure comprehensive metrics are calculated first
        self._calculate_comprehensive_metrics()
        
        # Log the current state for debugging
        logger.info(f"Getting metric summary - avg_relevance: {self.metrics.avg_relevance}, temporal_freshness: {self.metrics.temporal_freshness_score}, citation_impact: {self.metrics.citation_impact_score}, comprehensiveness: {self.metrics.comprehensiveness_score}")
        
        return {
            "key_metrics": {
                "overall_quality": round(self.metrics.quality_score, 1),
                "completion": f"{self.completion_percentage}%",
                "sources": self.metrics.total_sources,
                "insights": self.metrics.data_points,
                "paper_length": f"{self.metrics.paper_length:,} chars"
            },
            "performance_indicators": {
                "source_quality": "🟢" if self.metrics.avg_relevance >= 7.5 else "🟡" if self.metrics.avg_relevance >= 6.0 else "🔴",
                "temporal_freshness": "🟢" if self.metrics.temporal_freshness_score >= 7.5 else "🟡" if self.metrics.temporal_freshness_score >= 5.0 else "🔴",
                "citation_impact": "🟢" if self.metrics.citation_impact_score >= 7.5 else "🟡" if self.metrics.citation_impact_score >= 5.0 else "🔴",
                "research_depth": "🟢" if self.metrics.comprehensiveness_score >= 7.5 else "🟡" if self.metrics.comprehensiveness_score >= 5.0 else "🔴"
            },
            "top_scores": {
                "highest_score": max([
                    self.metrics.quality_score,
                    self.metrics.citation_impact_score,
                    self.metrics.temporal_freshness_score,
                    self.metrics.comprehensiveness_score
                ]),
                "lowest_score": min([
                    self.metrics.quality_score,
                    self.metrics.citation_impact_score,
                    self.metrics.temporal_freshness_score,
                    self.metrics.comprehensiveness_score
                ]),
                "most_improved": "source_quality" if self.metrics.avg_relevance >= 8.0 else "temporal_freshness" if self.metrics.temporal_freshness_score >= 8.0 else "citation_impact"
            },
            "recommendations": self._get_quick_recommendations()
        }
    
    def _get_quick_recommendations(self) -> List[str]:
        """Get quick recommendations based on current metrics"""
        recommendations = []
        
        if self.metrics.avg_relevance < 7.0:
            recommendations.append("Add more relevant sources")
        
        if self.metrics.temporal_freshness_score < 6.0:
            recommendations.append("Include more recent sources")
        
        if self.metrics.citation_impact_score < 5.0:
            recommendations.append("Focus on highly-cited sources")
        
        if self.metrics.comprehensiveness_score < 6.0:
            recommendations.append("Expand research coverage")
        
        if self.completion_percentage < 100:
            recommendations.append("Complete remaining research phases")
        
        if not recommendations:
            recommendations.append("Research quality is excellent!")
        
        return recommendations[:3]  # Return top 3 recommendations

# ===================================
# Agentic Tool Wrappers
# ===================================

@tool
def research_director_tool(topic: str) -> str:
    """Strategic planning and coordination for research topics. 
    Creates comprehensive research plans with objectives, methodology, and timeline."""
    try:
        assistant = AgenticResearchAssistant()
        return assistant.research_director(topic)
    except Exception as e:
        return f"Research Director Tool failed: {str(e)}"

@tool  
def literature_agent_tool(topic: str) -> str:
    """Advanced literature search and analysis. 
    Finds and analyzes relevant academic sources with relevance scoring."""
    try:
        assistant = AgenticResearchAssistant()
        return assistant.literature_agent(topic)
    except Exception as e:
        return f"Literature Agent Tool failed: {str(e)}"

@tool
def data_agent_tool(topic: str) -> str:
    """Data collection and insights generation. 
    Analyzes research data and generates dynamic insights."""
    try:
        assistant = AgenticResearchAssistant()
        return assistant.data_agent(topic)
    except Exception as e:
        return f"Data Agent Tool failed: {str(e)}"

@tool
def writer_agent_tool(topic: str) -> str:
    """Research paper generation. 
    Creates comprehensive academic papers with proper structure and content."""
    try:
        assistant = AgenticResearchAssistant()
        return assistant.writer_agent(topic)
    except Exception as e:
        return f"Writer Agent Tool failed: {str(e)}"

@tool
def editor_agent_tool(topic: str) -> str:
    """Quality assurance and editing. 
    Reviews and improves research papers for academic standards."""
    try:
        assistant = AgenticResearchAssistant()
        return assistant.editor_agent(topic)
    except Exception as e:
        return f"Editor Agent Tool failed: {str(e)}"

def get_research_tools():
    """Get all research tools for agent frameworks"""
    return [
        research_director_tool,
        literature_agent_tool,
        data_agent_tool,
        writer_agent_tool,
        editor_agent_tool
    ]

# ===================================
# Main execution for testing
# ===================================

if __name__ == "__main__":
    # Create agentic research assistant
    assistant = AgenticResearchAssistant()
    
    # Example usage
    topic = "The impact of artificial intelligence on modern education and learning outcomes"
    
    print("🚀 Agentic Research Assistant Ready!")
    print("=" * 80)
    
    # Run research
    result = assistant.run_research(topic)
    
    # Save results
    if "error" not in result:
        filename = assistant.export_results("json")
        print(f"💾 Results saved to {filename}")
    else:
        print("❌ Research failed, no results to save")
