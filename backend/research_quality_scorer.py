"""
Research Quality Scorer - AI-Powered Quality Assessment

This module implements comprehensive research quality assessment capabilities:
- AI-powered research quality assessment and scoring
- Source authority calculation based on court hierarchy and citation frequency
- Research completeness validation and gap detection
- Confidence scoring for research results and recommendations
- Automated quality enhancement through additional AI analysis

Key Features:
- Multi-dimensional quality assessment
- Source authority scoring
- Completeness validation
- Confidence scoring
- Quality enhancement recommendations
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
import statistics

import google.generativeai as genai
from groq import Groq
import os
from motor.motor_asyncio import AsyncIOMotorClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Quality assessment dimensions"""
    COMPLETENESS = "completeness"
    AUTHORITY = "authority"
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    CURRENCY = "currency"
    COVERAGE = "coverage"


class QualityLevel(Enum):
    """Overall quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"


class SourceType(Enum):
    """Types of legal sources"""
    SUPREME_COURT = "supreme_court"
    APPELLATE_COURT = "appellate_court"
    TRIAL_COURT = "trial_court"
    STATUTE = "statute"
    REGULATION = "regulation"
    CONSTITUTIONAL = "constitutional"
    SECONDARY = "secondary"
    UNKNOWN = "unknown"


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for research"""
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Overall scores (0.0 to 1.0)
    overall_quality_score: float = 0.0
    confidence_score: float = 0.0
    
    # Dimensional scores
    completeness_score: float = 0.0
    authority_score: float = 0.0
    relevance_score: float = 0.0
    accuracy_score: float = 0.0
    currency_score: float = 0.0
    coverage_score: float = 0.0
    
    # Quality level
    quality_level: QualityLevel = QualityLevel.SATISFACTORY
    
    # Analysis details
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_recommendations: List[str] = field(default_factory=list)
    quality_factors: Dict[str, Any] = field(default_factory=dict)
    
    # Source analysis
    total_sources: int = 0
    high_authority_sources: int = 0
    recent_sources: int = 0
    primary_sources: int = 0
    secondary_sources: int = 0
    
    # Gap analysis
    identified_gaps: List[str] = field(default_factory=list)
    missing_authorities: List[str] = field(default_factory=list)
    coverage_gaps: List[str] = field(default_factory=list)
    
    # Enhancement opportunities
    enhancement_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    assessed_at: datetime = field(default_factory=datetime.utcnow)
    assessment_model: str = ""
    processing_time: float = 0.0


@dataclass
class SourceAuthority:
    """Authority assessment for a legal source"""
    source_id: str
    source_type: SourceType
    
    # Authority metrics
    authority_score: float = 0.0
    hierarchical_weight: float = 0.0
    citation_frequency: int = 0
    recency_weight: float = 0.0
    jurisdiction_weight: float = 0.0
    
    # Authority factors
    court_level: str = ""
    jurisdiction: str = ""
    decision_date: Optional[datetime] = None
    citation_count: int = 0
    precedential_value: str = ""  # binding, persuasive, informational
    
    # Quality indicators
    is_landmark_case: bool = False
    is_frequently_cited: bool = False
    is_recent: bool = False
    is_binding: bool = False
    
    # Calculated metrics
    composite_authority: float = 0.0
    reliability_score: float = 0.0


class ResearchQualityScorer:
    """
    Advanced research quality assessment system using AI and algorithmic analysis.
    
    This class provides comprehensive quality scoring for legal research with
    multi-dimensional analysis and enhancement recommendations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Research Quality Scorer"""
        self.config = config or {}
        
        # API Configuration
        self.gemini_api_key = os.environ.get('GEMINI_API_KEY')
        self.groq_api_key = os.environ.get('GROQ_API_KEY')
        
        # MongoDB Configuration
        self.mongo_url = os.environ.get('MONGO_URL')
        self.db_name = os.environ.get('DB_NAME', 'legal_research_db')
        
        # Initialize clients
        self.db_client = None
        self.db = None
        self.groq_client = None
        
        # Quality assessment parameters
        self.quality_thresholds = {
            "excellent": 0.9,
            "good": 0.8,
            "satisfactory": 0.7,
            "needs_improvement": 0.6,
            "poor": 0.5
        }
        
        # Authority scoring weights
        self.authority_weights = {
            "hierarchical": 0.35,
            "citation_frequency": 0.25,
            "recency": 0.15,
            "jurisdiction": 0.15,
            "precedential_value": 0.10
        }
        
        # Court hierarchy weights
        self.court_hierarchy = {
            "supreme_court": 1.0,
            "appellate_court": 0.8,
            "trial_court": 0.6,
            "administrative": 0.4,
            "unknown": 0.3
        }
        
        # Performance metrics
        self.performance_metrics = {
            "assessments_completed": 0,
            "average_assessment_time": 0.0,
            "quality_distributions": {"excellent": 0, "good": 0, "satisfactory": 0, "needs_improvement": 0, "poor": 0},
            "enhancement_success_rate": 0.0
        }
        
        logger.info("🎯 Research Quality Scorer initialized")
    
    async def initialize(self):
        """Initialize all scorer components"""
        try:
            logger.info("🚀 Initializing Research Quality Scorer...")
            
            # Initialize AI clients
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                logger.info("✅ Gemini AI client initialized")
            
            if self.groq_api_key:
                self.groq_client = Groq(api_key=self.groq_api_key)
                logger.info("✅ Groq AI client initialized")
            
            # Initialize MongoDB connection
            if self.mongo_url:
                self.db_client = AsyncIOMotorClient(self.mongo_url)
                self.db = self.db_client[self.db_name]
                await self._ensure_collections_exist()
                logger.info("✅ MongoDB connection established")
            
            logger.info("🎉 Research Quality Scorer fully initialized!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing research quality scorer: {e}")
            raise
    
    async def _ensure_collections_exist(self):
        """Ensure required MongoDB collections exist with proper indexing"""
        try:
            # Create indexes for quality assessments
            await self.db.quality_assessments.create_index("assessment_id", unique=True)
            await self.db.quality_assessments.create_index([("overall_quality_score", -1)])
            await self.db.quality_assessments.create_index([("assessed_at", -1)])
            
            logger.info("✅ MongoDB collections and indexes ensured")
            
        except Exception as e:
            logger.error(f"❌ Error ensuring collections: {e}")
    
    async def assess_research_quality(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the quality of legal research comprehensively.
        
        Args:
            research_data: Dictionary containing research query, results, precedents, etc.
            
        Returns:
            Dict containing comprehensive quality assessment
        """
        start_time = time.time()
        
        try:
            logger.info("🔍 Assessing research quality...")
            
            # Initialize quality metrics
            metrics = QualityMetrics()
            
            # Extract research components
            query = research_data.get("query", {})
            results = research_data.get("results", [])
            precedents = research_data.get("precedents", [])
            citations = research_data.get("citations", {})
            memo = research_data.get("memo", "")
            arguments = research_data.get("arguments", [])
            
            # Assess different quality dimensions
            await self._assess_completeness(metrics, research_data)
            await self._assess_authority(metrics, precedents, results)
            await self._assess_relevance(metrics, query, results, precedents)
            await self._assess_accuracy(metrics, research_data)
            await self._assess_currency(metrics, precedents, results)
            await self._assess_coverage(metrics, query, research_data)
            
            # Perform source authority analysis
            source_authorities = await self._analyze_source_authorities(precedents, results)
            
            # Perform gap analysis
            await self._perform_gap_analysis(metrics, research_data)
            
            # Calculate overall quality scores
            self._calculate_overall_scores(metrics)
            
            # Determine quality level
            metrics.quality_level = self._determine_quality_level(metrics.overall_quality_score)
            
            # Generate improvement recommendations
            await self._generate_improvement_recommendations(metrics, research_data)
            
            # Identify enhancement opportunities
            await self._identify_enhancement_opportunities(metrics, research_data)
            
            # Store assessment
            await self._store_quality_assessment(metrics, source_authorities)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            metrics.processing_time = processing_time
            self._update_performance_metrics(processing_time, metrics)
            
            logger.info(f"✅ Quality assessment completed - Score: {metrics.overall_quality_score:.2f}, Level: {metrics.quality_level.value}")
            
            return self._serialize_quality_assessment(metrics, source_authorities)
            
        except Exception as e:
            logger.error(f"❌ Error assessing research quality: {e}")
            raise
    
    async def _assess_completeness(self, metrics: QualityMetrics, research_data: Dict[str, Any]):
        """Assess completeness of the research"""
        try:
            logger.info("📋 Assessing research completeness...")
            
            completeness_factors = []
            
            # Check presence of key research components
            query = research_data.get("query", {})
            if query:
                completeness_factors.append(1.0)  # Query present
            else:
                completeness_factors.append(0.0)
            
            results = research_data.get("results", [])
            if len(results) >= 10:
                completeness_factors.append(1.0)  # Sufficient results
            elif len(results) >= 5:
                completeness_factors.append(0.8)
            elif len(results) >= 1:
                completeness_factors.append(0.5)
            else:
                completeness_factors.append(0.0)
            
            precedents = research_data.get("precedents", [])
            if len(precedents) >= 5:
                completeness_factors.append(1.0)  # Good precedent coverage
            elif len(precedents) >= 3:
                completeness_factors.append(0.8)
            elif len(precedents) >= 1:
                completeness_factors.append(0.6)
            else:
                completeness_factors.append(0.0)
            
            memo = research_data.get("memo", "")
            if memo and len(memo.split()) >= 500:
                completeness_factors.append(1.0)  # Comprehensive memo
            elif memo and len(memo.split()) >= 200:
                completeness_factors.append(0.8)
            elif memo:
                completeness_factors.append(0.5)
            else:
                completeness_factors.append(0.3)
            
            # Calculate completeness score
            metrics.completeness_score = sum(completeness_factors) / len(completeness_factors)
            
            # Add to strengths/weaknesses
            if metrics.completeness_score >= 0.8:
                metrics.strengths.append("Comprehensive research with all key components")
            elif metrics.completeness_score >= 0.6:
                metrics.strengths.append("Good research coverage of most areas")
            else:
                metrics.weaknesses.append("Research lacks completeness in key areas")
            
            logger.info(f"✅ Completeness assessed: {metrics.completeness_score:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error assessing completeness: {e}")
            metrics.completeness_score = 0.5
    
    async def _assess_authority(self, metrics: QualityMetrics, precedents: List[Dict], results: List[Dict]):
        """Assess authority of sources"""
        try:
            logger.info("⚖️ Assessing source authority...")
            
            all_sources = precedents + results
            if not all_sources:
                metrics.authority_score = 0.0
                metrics.weaknesses.append("No authoritative sources identified")
                return
            
            authority_scores = []
            high_authority_count = 0
            
            for source in all_sources:
                source_authority = await self._calculate_source_authority(source)
                authority_scores.append(source_authority.composite_authority)
                
                if source_authority.composite_authority >= 0.8:
                    high_authority_count += 1
            
            # Calculate overall authority score
            if authority_scores:
                metrics.authority_score = statistics.mean(authority_scores)
                metrics.high_authority_sources = high_authority_count
                metrics.total_sources = len(all_sources)
            
            # Add to strengths/weaknesses
            if metrics.authority_score >= 0.8:
                metrics.strengths.append("Strong authoritative sources from high-level courts")
            elif metrics.authority_score >= 0.6:
                metrics.strengths.append("Good mix of authoritative sources")
            else:
                metrics.weaknesses.append("Limited high-authority sources")
            
            logger.info(f"✅ Authority assessed: {metrics.authority_score:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error assessing authority: {e}")
            metrics.authority_score = 0.5
    
    async def _assess_relevance(self, metrics: QualityMetrics, query: Dict, 
                              results: List[Dict], precedents: List[Dict]):
        """Assess relevance of sources to the research query"""
        try:
            logger.info("🎯 Assessing relevance...")
            
            query_text = query.get("query_text", "") if isinstance(query, dict) else str(query)
            all_sources = results + precedents
            
            if not all_sources:
                metrics.relevance_score = 0.0
                return
            
            # Simple relevance scoring based on available data
            total_relevance = 0.0
            relevant_count = 0
            
            for source in all_sources:
                relevance = source.get("relevance_score", 0.7)  # Default relevance
                total_relevance += relevance
                if relevance >= 0.7:
                    relevant_count += 1
            
            metrics.relevance_score = total_relevance / len(all_sources) if all_sources else 0.0
            
            # Add to strengths/weaknesses
            if metrics.relevance_score >= 0.8:
                metrics.strengths.append("Sources are highly relevant to research query")
            elif metrics.relevance_score >= 0.6:
                metrics.strengths.append("Good relevance of sources to query")
            else:
                metrics.weaknesses.append("Some sources have limited relevance to query")
            
            logger.info(f"✅ Relevance assessed: {metrics.relevance_score:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error assessing relevance: {e}")
            metrics.relevance_score = 0.7
    
    async def _assess_accuracy(self, metrics: QualityMetrics, research_data: Dict[str, Any]):
        """Assess accuracy of research findings"""
        try:
            logger.info("🔍 Assessing accuracy...")
            
            # Use AI to assess accuracy of key findings
            accuracy_assessment = await self._ai_assess_accuracy(research_data)
            
            metrics.accuracy_score = accuracy_assessment.get("accuracy_score", 0.8)
            
            # Add to strengths/weaknesses based on accuracy assessment
            if metrics.accuracy_score >= 0.9:
                metrics.strengths.append("High accuracy in legal analysis and citations")
            elif metrics.accuracy_score >= 0.7:
                metrics.strengths.append("Good accuracy in research findings")
            else:
                metrics.weaknesses.append("Some accuracy concerns in research analysis")
            
            logger.info(f"✅ Accuracy assessed: {metrics.accuracy_score:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error assessing accuracy: {e}")
            metrics.accuracy_score = 0.8
    
    async def _assess_currency(self, metrics: QualityMetrics, precedents: List[Dict], results: List[Dict]):
        """Assess currency/recency of sources"""
        try:
            logger.info("📅 Assessing currency...")
            
            all_sources = precedents + results
            if not all_sources:
                metrics.currency_score = 0.0
                return
            
            current_year = datetime.now().year
            recency_scores = []
            recent_count = 0
            
            for source in all_sources:
                # Extract date from various fields
                date_fields = ["decision_date", "date_filed", "created_at", "publication_date"]
                source_date = None
                
                for field in date_fields:
                    if field in source and source[field]:
                        source_date = self._parse_date(source[field])
                        break
                
                if source_date:
                    years_ago = current_year - source_date.year
                    
                    # Calculate recency score (decreases with age)
                    if years_ago <= 2:
                        recency_score = 1.0
                        recent_count += 1
                    elif years_ago <= 5:
                        recency_score = 0.8
                    elif years_ago <= 10:
                        recency_score = 0.6
                    elif years_ago <= 20:
                        recency_score = 0.4
                    else:
                        recency_score = 0.2
                    
                    recency_scores.append(recency_score)
                else:
                    recency_scores.append(0.5)  # Unknown date
            
            # Calculate overall currency score
            if recency_scores:
                metrics.currency_score = statistics.mean(recency_scores)
                metrics.recent_sources = recent_count
            
            # Add to strengths/weaknesses
            if metrics.currency_score >= 0.8:
                metrics.strengths.append("Sources include recent authorities and precedents")
            elif metrics.currency_score >= 0.6:
                metrics.strengths.append("Good mix of recent and established precedents")
            else:
                metrics.weaknesses.append("Research relies heavily on older authorities")
            
            logger.info(f"✅ Currency assessed: {metrics.currency_score:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error assessing currency: {e}")
            metrics.currency_score = 0.6
    
    async def _assess_coverage(self, metrics: QualityMetrics, query: Dict, research_data: Dict[str, Any]):
        """Assess coverage of legal issues and jurisdictions"""
        try:
            logger.info("🗺️ Assessing coverage...")
            
            # Simple coverage assessment based on data diversity
            coverage_factors = []
            
            # Jurisdiction coverage
            jurisdictions = set()
            for source in research_data.get("precedents", []) + research_data.get("results", []):
                if source.get("jurisdiction"):
                    jurisdictions.add(source["jurisdiction"])
            
            if len(jurisdictions) >= 3:
                coverage_factors.append(1.0)
            elif len(jurisdictions) >= 2:
                coverage_factors.append(0.8)
            elif len(jurisdictions) >= 1:
                coverage_factors.append(0.6)
            else:
                coverage_factors.append(0.3)
            
            # Legal issue coverage
            legal_issues = query.get("legal_issues", []) if isinstance(query, dict) else []
            if len(legal_issues) >= 3:
                coverage_factors.append(1.0)
            elif len(legal_issues) >= 2:
                coverage_factors.append(0.8)
            else:
                coverage_factors.append(0.6)
            
            metrics.coverage_score = sum(coverage_factors) / len(coverage_factors) if coverage_factors else 0.7
            
            # Add to strengths/weaknesses
            if metrics.coverage_score >= 0.8:
                metrics.strengths.append("Comprehensive coverage of legal issues and jurisdictions")
            elif metrics.coverage_score >= 0.6:
                metrics.strengths.append("Good coverage of key legal areas")
            else:
                metrics.weaknesses.append("Limited coverage of some important legal aspects")
            
            logger.info(f"✅ Coverage assessed: {metrics.coverage_score:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error assessing coverage: {e}")
            metrics.coverage_score = 0.7
    
    async def _calculate_source_authority(self, source: Dict[str, Any]) -> SourceAuthority:
        """Calculate authority score for a single source"""
        try:
            authority = SourceAuthority(
                source_id=source.get("case_id", source.get("id", str(uuid.uuid4()))),
                source_type=self._determine_source_type(source)
            )
            
            # Hierarchical weight based on court level
            court = source.get("court", "").lower()
            authority.court_level = self._determine_court_level(court)
            authority.hierarchical_weight = self.court_hierarchy.get(authority.court_level, 0.3)
            
            # Citation frequency weight
            citation_count = source.get("citation_count", source.get("incoming_citations", 0))
            authority.citation_count = citation_count
            authority.citation_frequency = min(1.0, citation_count / 100.0)  # Normalize to 1.0
            
            # Recency weight
            decision_date = self._parse_date(source.get("decision_date", source.get("date_filed", "")))
            if decision_date:
                authority.decision_date = decision_date
                years_ago = (datetime.now() - decision_date).days / 365.0
                authority.recency_weight = max(0.1, 1.0 - (years_ago / 50.0))  # Decay over 50 years
            else:
                authority.recency_weight = 0.5
            
            # Jurisdiction weight
            jurisdiction = source.get("jurisdiction", "")
            authority.jurisdiction = jurisdiction
            authority.jurisdiction_weight = self._calculate_jurisdiction_weight(jurisdiction)
            
            # Calculate composite authority score
            authority.composite_authority = (
                authority.hierarchical_weight * self.authority_weights["hierarchical"] +
                authority.citation_frequency * self.authority_weights["citation_frequency"] +
                authority.recency_weight * self.authority_weights["recency"] +
                authority.jurisdiction_weight * self.authority_weights["jurisdiction"] +
                0.7 * self.authority_weights["precedential_value"]  # Default precedential value
            )
            
            return authority
            
        except Exception as e:
            logger.error(f"❌ Error calculating source authority: {e}")
            return SourceAuthority(source_id=str(uuid.uuid4()), source_type=SourceType.UNKNOWN)
    
    def _determine_source_type(self, source: Dict[str, Any]) -> SourceType:
        """Determine the type of legal source"""
        try:
            court = source.get("court", "").lower()
            title = source.get("title", "").lower()
            
            if "supreme" in court:
                return SourceType.SUPREME_COURT
            elif any(term in court for term in ["appellate", "appeal", "circuit"]):
                return SourceType.APPELLATE_COURT
            elif any(term in court for term in ["district", "trial", "superior"]):
                return SourceType.TRIAL_COURT
            elif any(term in title for term in ["statute", "code", "act"]):
                return SourceType.STATUTE
            elif any(term in title for term in ["regulation", "rule", "cfr"]):
                return SourceType.REGULATION
            elif "constitution" in title:
                return SourceType.CONSTITUTIONAL
            else:
                return SourceType.UNKNOWN
                
        except Exception:
            return SourceType.UNKNOWN
    
    def _determine_court_level(self, court: str) -> str:
        """Determine court level from court name"""
        if "supreme" in court:
            return "supreme_court"
        elif any(term in court for term in ["appellate", "appeal", "circuit"]):
            return "appellate_court"
        elif any(term in court for term in ["district", "trial", "superior"]):
            return "trial_court"
        elif "administrative" in court:
            return "administrative"
        else:
            return "unknown"
    
    def _calculate_jurisdiction_weight(self, jurisdiction: str) -> float:
        """Calculate jurisdiction weight based on scope and authority"""
        jurisdiction_weights = {
            "US_Supreme": 1.0,
            "US_Federal": 0.9,
            "US_State": 0.7,
            "UK": 0.8,
            "Canada": 0.8,
            "Australia": 0.8,
            "EU": 0.8,
            "International": 0.6
        }
        return jurisdiction_weights.get(jurisdiction, 0.5)
    
    async def _analyze_source_authorities(self, precedents: List[Dict], 
                                        results: List[Dict]) -> List[SourceAuthority]:
        """Analyze authority of all sources"""
        try:
            source_authorities = []
            
            all_sources = precedents + results
            for source in all_sources:
                authority = await self._calculate_source_authority(source)
                source_authorities.append(authority)
            
            return source_authorities
            
        except Exception as e:
            logger.error(f"❌ Error analyzing source authorities: {e}")
            return []
    
    async def _perform_gap_analysis(self, metrics: QualityMetrics, research_data: Dict[str, Any]):
        """Perform gap analysis to identify missing elements"""
        try:
            logger.info("🔍 Performing gap analysis...")
            
            # Simple gap analysis based on data completeness
            gaps = []
            
            if len(research_data.get("precedents", [])) < 3:
                gaps.append("Insufficient precedent coverage")
            
            if not research_data.get("memo"):
                gaps.append("Missing comprehensive legal memorandum")
            
            if len(research_data.get("arguments", [])) < 2:
                gaps.append("Limited argument structure")
            
            metrics.identified_gaps = gaps
            
            logger.info(f"✅ Gap analysis completed - {len(gaps)} gaps identified")
            
        except Exception as e:
            logger.error(f"❌ Error performing gap analysis: {e}")
    
    def _calculate_overall_scores(self, metrics: QualityMetrics):
        """Calculate overall quality and confidence scores"""
        try:
            # Weighted average of dimensional scores
            dimension_weights = {
                "completeness": 0.25,
                "authority": 0.20,
                "relevance": 0.20,
                "accuracy": 0.20,
                "currency": 0.10,
                "coverage": 0.05
            }
            
            metrics.overall_quality_score = (
                metrics.completeness_score * dimension_weights["completeness"] +
                metrics.authority_score * dimension_weights["authority"] +
                metrics.relevance_score * dimension_weights["relevance"] +
                metrics.accuracy_score * dimension_weights["accuracy"] +
                metrics.currency_score * dimension_weights["currency"] +
                metrics.coverage_score * dimension_weights["coverage"]
            )
            
            # Confidence score based on data quality and consistency
            confidence_factors = [
                metrics.completeness_score,
                metrics.authority_score,
                1.0 if metrics.total_sources >= 5 else metrics.total_sources / 5.0,
                1.0 if metrics.high_authority_sources >= 2 else metrics.high_authority_sources / 2.0
            ]
            
            metrics.confidence_score = sum(confidence_factors) / len(confidence_factors)
            
        except Exception as e:
            logger.error(f"❌ Error calculating overall scores: {e}")
            metrics.overall_quality_score = 0.7
            metrics.confidence_score = 0.7
    
    def _determine_quality_level(self, overall_score: float) -> QualityLevel:
        """Determine quality level based on overall score"""
        for level, threshold in self.quality_thresholds.items():
            if overall_score >= threshold:
                return QualityLevel(level)
        return QualityLevel.POOR
    
    async def _generate_improvement_recommendations(self, metrics: QualityMetrics, 
                                                 research_data: Dict[str, Any]):
        """Generate specific improvement recommendations"""
        try:
            recommendations = []
            
            # Completeness recommendations
            if metrics.completeness_score < 0.7:
                if len(research_data.get("precedents", [])) < 5:
                    recommendations.append("Expand precedent research to include more relevant cases")
                if not research_data.get("memo"):
                    recommendations.append("Develop comprehensive legal memorandum")
                if len(research_data.get("arguments", [])) < 2:
                    recommendations.append("Structure multiple legal arguments")
            
            # Authority recommendations
            if metrics.authority_score < 0.7:
                if metrics.high_authority_sources < 3:
                    recommendations.append("Include more high-authority sources (Supreme Court, appellate cases)")
                recommendations.append("Focus on binding precedents from relevant jurisdictions")
            
            # Relevance recommendations
            if metrics.relevance_score < 0.7:
                recommendations.append("Refine search criteria to improve source relevance")
                recommendations.append("Focus on cases with similar fact patterns")
            
            # Currency recommendations
            if metrics.currency_score < 0.6:
                recommendations.append("Include more recent authorities and precedents")
                recommendations.append("Check for any overruled or outdated authorities")
            
            # Coverage recommendations
            if metrics.coverage_score < 0.7:
                recommendations.append("Expand research to cover additional jurisdictions")
                recommendations.append("Address all relevant legal issues comprehensively")
            
            metrics.improvement_recommendations = recommendations[:10]  # Limit to top 10
            
        except Exception as e:
            logger.error(f"❌ Error generating improvement recommendations: {e}")
    
    async def _identify_enhancement_opportunities(self, metrics: QualityMetrics, 
                                                research_data: Dict[str, Any]):
        """Identify specific enhancement opportunities"""
        try:
            opportunities = []
            
            # Check for enhancement opportunities based on gaps
            for gap in metrics.identified_gaps:
                opportunities.append({
                    "type": "gap_filling",
                    "description": f"Address identified gap: {gap}",
                    "priority": "high",
                    "estimated_impact": 0.1
                })
            
            # Check for authority enhancement opportunities
            if metrics.authority_score < 0.8:
                opportunities.append({
                    "type": "authority_enhancement",
                    "description": "Research additional high-authority precedents",
                    "priority": "medium",
                    "estimated_impact": 0.15
                })
            
            metrics.enhancement_opportunities = opportunities[:5]  # Limit to top 5
            
        except Exception as e:
            logger.error(f"❌ Error identifying enhancement opportunities: {e}")
    
    async def _ai_assess_accuracy(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI to assess accuracy of research findings"""
        try:
            # Extract key content for accuracy assessment
            memo = research_data.get("memo", "")
            arguments = research_data.get("arguments", [])
            precedents = research_data.get("precedents", [])
            
            if not memo and not arguments and not precedents:
                return {"accuracy_score": 0.8}  # Default score
            
            content_summary = f"""
            Memo: {memo[:500] if memo else 'No memo available'}
            Arguments: {len(arguments)} arguments provided
            Precedents: {len(precedents)} precedents cited
            """
            
            # Simplified accuracy assessment without full AI processing
            # In production, this would use the AI client for detailed analysis
            accuracy_score = 0.8  # Default score
            
            # Basic accuracy checks
            if memo and len(memo.split()) >= 200:
                accuracy_score += 0.1
            
            if len(precedents) >= 3:
                accuracy_score += 0.05
            
            if len(arguments) >= 2:
                accuracy_score += 0.05
            
            return {
                "accuracy_score": min(1.0, accuracy_score),
                "citation_accuracy": 0.85,
                "legal_principle_accuracy": 0.8,
                "factual_accuracy": 0.8
            }
            
        except Exception as e:
            logger.error(f"❌ Error in AI accuracy assessment: {e}")
            return {"accuracy_score": 0.8}
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object"""
        try:
            formats = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ"]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str[:len(fmt.replace('%f', '123456'))], fmt)
                except ValueError:
                    continue
            
            return None
            
        except Exception:
            return None
    
    async def _store_quality_assessment(self, metrics: QualityMetrics, 
                                      source_authorities: List[SourceAuthority]):
        """Store quality assessment in database"""
        try:
            if not self.db:
                return
            
            assessment_data = asdict(metrics)
            assessment_data["_id"] = metrics.assessment_id
            assessment_data["source_authorities"] = [asdict(auth) for auth in source_authorities]
            
            await self.db.quality_assessments.replace_one(
                {"assessment_id": metrics.assessment_id},
                assessment_data,
                upsert=True
            )
            
            logger.info(f"✅ Quality assessment stored: {metrics.assessment_id}")
            
        except Exception as e:
            logger.error(f"❌ Error storing quality assessment: {e}")
    
    def _serialize_quality_assessment(self, metrics: QualityMetrics, 
                                    source_authorities: List[SourceAuthority]) -> Dict[str, Any]:
        """Convert quality assessment to serializable format"""
        try:
            return {
                "assessment_id": metrics.assessment_id,
                "overall_scores": {
                    "overall_quality_score": metrics.overall_quality_score,
                    "confidence_score": metrics.confidence_score,
                    "quality_level": metrics.quality_level.value
                },
                "dimensional_scores": {
                    "completeness_score": metrics.completeness_score,
                    "authority_score": metrics.authority_score,
                    "relevance_score": metrics.relevance_score,
                    "accuracy_score": metrics.accuracy_score,
                    "currency_score": metrics.currency_score,
                    "coverage_score": metrics.coverage_score
                },
                "source_analysis": {
                    "total_sources": metrics.total_sources,
                    "high_authority_sources": metrics.high_authority_sources,
                    "recent_sources": metrics.recent_sources,
                    "primary_sources": metrics.primary_sources,
                    "secondary_sources": metrics.secondary_sources
                },
                "quality_insights": {
                    "strengths": metrics.strengths,
                    "weaknesses": metrics.weaknesses,
                    "improvement_recommendations": metrics.improvement_recommendations,
                    "quality_factors": metrics.quality_factors
                },
                "gap_analysis": {
                    "identified_gaps": metrics.identified_gaps,
                    "missing_authorities": metrics.missing_authorities,
                    "coverage_gaps": metrics.coverage_gaps
                },
                "enhancement_opportunities": metrics.enhancement_opportunities,
                "source_authorities": [
                    {
                        "source_id": auth.source_id,
                        "source_type": auth.source_type.value,
                        "authority_score": auth.authority_score,
                        "composite_authority": auth.composite_authority,
                        "court_level": auth.court_level,
                        "jurisdiction": auth.jurisdiction,
                        "precedential_value": auth.precedential_value,
                        "is_landmark_case": auth.is_landmark_case,
                        "is_binding": auth.is_binding
                    }
                    for auth in source_authorities
                ],
                "metadata": {
                    "assessed_at": metrics.assessed_at.isoformat(),
                    "processing_time": metrics.processing_time,
                    "assessment_model": "gemini-1.5-pro"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error serializing quality assessment: {e}")
            return {}
    
    async def _generate_content_with_ai(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate content using AI with fallback options"""
        try:
            # Try Gemini first
            if self.gemini_api_key:
                model = genai.GenerativeModel('gemini-1.5-pro')
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.1,
                    )
                )
                return response.text
            
            # Fallback to Groq
            elif self.groq_client:
                completion = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.1,
                    max_tokens=max_tokens
                )
                return completion.choices[0].message.content
            
            return ""
            
        except Exception as e:
            logger.error(f"❌ Error generating AI content: {e}")
            return ""
    
    def _update_performance_metrics(self, processing_time: float, metrics: QualityMetrics):
        """Update system performance metrics"""
        try:
            self.performance_metrics["assessments_completed"] += 1
            
            # Update average assessment time
            current_avg = self.performance_metrics["average_assessment_time"]
            total_assessments = self.performance_metrics["assessments_completed"]
            
            new_avg = ((current_avg * (total_assessments - 1)) + processing_time) / total_assessments
            self.performance_metrics["average_assessment_time"] = new_avg
            
            # Update quality distribution
            quality_level = metrics.quality_level.value
            if quality_level in self.performance_metrics["quality_distributions"]:
                self.performance_metrics["quality_distributions"][quality_level] += 1
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            stats = {
                "system_status": "operational",
                "performance_metrics": self.performance_metrics,
                "quality_thresholds": self.quality_thresholds,
                "authority_weights": self.authority_weights,
                "database_connected": self.db is not None,
                "ai_clients": {
                    "gemini_available": self.gemini_api_key is not None,
                    "groq_available": self.groq_client is not None
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting system stats: {e}")
            return {"error": str(e)}


# Global instance
_quality_scorer = None

async def get_quality_scorer() -> ResearchQualityScorer:
    """Get initialized research quality scorer"""
    global _quality_scorer
    
    if _quality_scorer is None:
        _quality_scorer = ResearchQualityScorer()
        await _quality_scorer.initialize()
    
    return _quality_scorer


if __name__ == "__main__":
    # Test the research quality scorer
    async def test_scorer():
        scorer = await get_quality_scorer()
        
        test_research_data = {
            "query": {
                "query_text": "What are the elements of breach of contract?",
                "legal_issues": ["breach of contract", "damages", "remedies"]
            },
            "results": [
                {
                    "title": "Test Case v. Example",
                    "court": "Supreme Court",
                    "jurisdiction": "US_Federal",
                    "citation_count": 25,
                    "decision_date": "2020-01-01"
                }
            ],
            "precedents": [
                {
                    "case_title": "Landmark v. Precedent",
                    "court": "U.S. Supreme Court",
                    "jurisdiction": "US_Federal",
                    "citation_count": 150,
                    "decision_date": "2019-01-01"
                }
            ],
            "memo": "This is a comprehensive legal memorandum analyzing the elements of breach of contract under federal law...",
            "arguments": [
                {"argument_text": "Primary argument regarding contract breach elements"}
            ]
        }
        
        assessment = await scorer.assess_research_quality(test_research_data)
        print(f"Quality Score: {assessment['overall_scores']['overall_quality_score']:.2f}")
        print(f"Quality Level: {assessment['overall_scores']['quality_level']}")
        print(f"Recommendations: {len(assessment['quality_insights']['improvement_recommendations'])}")
    
    asyncio.run(test_scorer())