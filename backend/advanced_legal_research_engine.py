"""
Advanced Legal Research Engine - Main Orchestration Engine

This module implements the core orchestration engine for advanced legal research
that coordinates between different research components and provides enterprise-grade
legal research capabilities rivaling Harvey AI and CoCounsel.

Key Features:
- Research workflow management and result aggregation
- Integration with existing legal_rag_system and legal_knowledge_builder
- Research session management and caching for performance optimization
- Multi-dimensional research coordination (precedent, citation, memo, argument)
- AI-powered research quality assessment and enhancement
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

import google.generativeai as genai
from groq import Groq
import httpx
import os
from motor.motor_asyncio import AsyncIOMotorClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchType(Enum):
    """Types of legal research operations"""
    PRECEDENT_SEARCH = "precedent_search"
    CITATION_ANALYSIS = "citation_analysis"
    MEMO_GENERATION = "memo_generation"
    ARGUMENT_BUILDING = "argument_building"
    JURISDICTION_COMPARISON = "jurisdiction_comparison"
    COMPREHENSIVE = "comprehensive"


class ResearchPriority(Enum):
    """Research operation priorities"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class ResearchStatus(Enum):
    """Research operation status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CACHED = "cached"


@dataclass
class ResearchQuery:
    """Structured research query with all parameters"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_text: str = ""
    research_type: ResearchType = ResearchType.COMPREHENSIVE
    jurisdiction: str = "US"
    legal_domain: str = "general"
    priority: ResearchPriority = ResearchPriority.MEDIUM
    
    # Query parameters
    court_level: Optional[str] = None
    date_range: Optional[Dict[str, str]] = None
    case_type: Optional[str] = None
    legal_issues: List[str] = field(default_factory=list)
    
    # Processing configuration  
    max_results: int = 50
    min_confidence: float = 0.7
    include_analysis: bool = True
    cache_results: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    user_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchResult:
    """Comprehensive research result structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str = ""
    research_type: ResearchType = ResearchType.COMPREHENSIVE
    
    # Results data
    results: List[Dict[str, Any]] = field(default_factory=list)
    precedent_matches: List[Dict[str, Any]] = field(default_factory=list)
    citation_network: Dict[str, Any] = field(default_factory=dict)
    generated_memo: Optional[str] = None
    legal_arguments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality metrics
    confidence_score: float = 0.0
    completeness_score: float = 0.0
    authority_score: float = 0.0
    
    # Processing metadata
    processing_time: float = 0.0
    models_used: List[str] = field(default_factory=list)
    sources_count: int = 0
    status: ResearchStatus = ResearchStatus.PENDING
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class AdvancedLegalResearchEngine:
    """
    Main orchestration engine for advanced legal research operations.
    
    This class coordinates all research components and provides a unified interface
    for enterprise-grade legal research functionality.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Advanced Legal Research Engine"""
        self.config = config or {}
        self.session_id = str(uuid.uuid4())
        
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
        
        # Component references (will be injected)
        self.precedent_matcher = None
        self.citation_analyzer = None
        self.memo_generator = None
        self.argument_structurer = None
        self.jurisdiction_searcher = None
        self.quality_scorer = None
        
        # Caching and performance
        self.result_cache = {}
        self.performance_metrics = {}
        
        # Research sessions
        self.active_sessions = {}
        
        logger.info("🚀 Advanced Legal Research Engine initialized")
    
    async def initialize(self):
        """Initialize all engine components and connections"""
        try:
            logger.info("🔧 Initializing Advanced Legal Research Engine...")
            
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
                logger.info("✅ MongoDB connection established")
            
            # Initialize component systems (will be imported dynamically)
            await self._initialize_components()
            
            # Load existing cache and metrics
            await self._load_performance_data()
            
            logger.info("🎉 Advanced Legal Research Engine fully initialized!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing research engine: {e}")
            raise
    
    async def _initialize_components(self):
        """Initialize all research component systems"""
        try:
            # Import and initialize component systems
            from precedent_matching_system import PrecedentMatchingSystem
            from citation_network_analyzer import CitationNetworkAnalyzer  
            from research_memo_generator import ResearchMemoGenerator
            from legal_argument_structurer import LegalArgumentStructurer
            from multi_jurisdiction_search import MultiJurisdictionSearch
            from research_quality_scorer import ResearchQualityScorer
            
            # Initialize components
            self.precedent_matcher = PrecedentMatchingSystem()
            self.citation_analyzer = CitationNetworkAnalyzer()
            self.memo_generator = ResearchMemoGenerator()
            self.argument_structurer = LegalArgumentStructurer()
            self.jurisdiction_searcher = MultiJurisdictionSearch()
            self.quality_scorer = ResearchQualityScorer()
            
            # Initialize all components
            await asyncio.gather(
                self.precedent_matcher.initialize(),
                self.citation_analyzer.initialize(),
                self.memo_generator.initialize(),
                self.argument_structurer.initialize(),
                self.jurisdiction_searcher.initialize(),
                self.quality_scorer.initialize()
            )
            
            logger.info("✅ All research components initialized")
            
        except ImportError as e:
            logger.warning(f"⚠️ Some research components not available: {e}")
            # Continue with available components
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            raise
    
    async def coordinate_research(self, query: ResearchQuery) -> ResearchResult:
        """
        Main research coordination method that orchestrates all research components.
        
        Args:
            query: Structured research query with parameters
            
        Returns:
            ResearchResult: Comprehensive research results
        """
        start_time = time.time()
        result = ResearchResult(query_id=query.id, research_type=query.research_type)
        
        try:
            logger.info(f"🔍 Starting research coordination for query: {query.id}")
            
            # Check cache first
            cached_result = await self._check_cache(query)
            if cached_result and query.cache_results:
                logger.info(f"📋 Returning cached result for query: {query.id}")
                cached_result.status = ResearchStatus.CACHED
                return cached_result
            
            result.status = ResearchStatus.PROCESSING
            
            # Determine research components needed based on query type
            components_needed = self._determine_research_components(query)
            
            # Execute research operations in parallel where possible
            research_tasks = []
            
            if 'precedent' in components_needed:
                research_tasks.append(self._execute_precedent_research(query, result))
            
            if 'citation' in components_needed:
                research_tasks.append(self._execute_citation_analysis(query, result))
            
            if 'jurisdiction' in components_needed:
                research_tasks.append(self._execute_jurisdiction_research(query, result))
            
            # Execute primary research tasks
            if research_tasks:
                await asyncio.gather(*research_tasks, return_exceptions=True)
            
            # Sequential tasks that depend on primary research results
            if 'memo' in components_needed and result.results:
                await self._execute_memo_generation(query, result)
            
            if 'argument' in components_needed and result.results:
                await self._execute_argument_structuring(query, result)
            
            # Final quality assessment
            await self._execute_quality_assessment(query, result)
            
            # Aggregate and finalize results
            await self._finalize_research_results(query, result)
            
            result.status = ResearchStatus.COMPLETED
            result.processing_time = time.time() - start_time
            
            # Cache results if requested
            if query.cache_results:
                await self._cache_result(query, result)
            
            # Store research session
            await self._store_research_session(query, result)
            
            logger.info(f"✅ Research coordination completed for query: {query.id} in {result.processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in research coordination: {e}")
            result.status = ResearchStatus.ERROR
            result.processing_time = time.time() - start_time
            raise
    
    def _determine_research_components(self, query: ResearchQuery) -> List[str]:
        """Determine which research components are needed based on query type"""
        components = []
        
        if query.research_type == ResearchType.PRECEDENT_SEARCH:
            components = ['precedent', 'citation']
        elif query.research_type == ResearchType.CITATION_ANALYSIS:
            components = ['citation']
        elif query.research_type == ResearchType.MEMO_GENERATION:
            components = ['precedent', 'citation', 'memo']
        elif query.research_type == ResearchType.ARGUMENT_BUILDING:
            components = ['precedent', 'citation', 'argument']
        elif query.research_type == ResearchType.JURISDICTION_COMPARISON:
            components = ['precedent', 'jurisdiction']
        elif query.research_type == ResearchType.COMPREHENSIVE:
            components = ['precedent', 'citation', 'jurisdiction', 'memo', 'argument']
        
        return components
    
    async def _execute_precedent_research(self, query: ResearchQuery, result: ResearchResult):
        """Execute precedent matching research"""
        try:
            if not self.precedent_matcher:
                logger.warning("Precedent matcher not available")
                return
                
            logger.info("🔍 Executing precedent research...")
            
            precedent_results = await self.precedent_matcher.find_similar_cases(
                query_case={"facts": query.query_text, "legal_issues": query.legal_issues},
                filters={
                    "jurisdiction": query.jurisdiction,
                    "legal_domain": query.legal_domain,
                    "court_level": query.court_level,
                    "date_range": query.date_range,
                    "max_results": query.max_results
                }
            )
            
            result.precedent_matches = precedent_results
            result.models_used.append("precedent_matcher")
            
            logger.info(f"✅ Found {len(precedent_results)} precedent matches")
            
        except Exception as e:
            logger.error(f"❌ Error in precedent research: {e}")
            raise
    
    async def _execute_citation_analysis(self, query: ResearchQuery, result: ResearchResult):
        """Execute citation network analysis"""
        try:
            if not self.citation_analyzer:
                logger.warning("Citation analyzer not available")
                return
                
            logger.info("📊 Executing citation analysis...")
            
            # Use precedent results if available for citation analysis
            cases_for_analysis = result.precedent_matches or [{"query": query.query_text}]
            
            citation_network = await self.citation_analyzer.build_citation_network(
                cases=cases_for_analysis,
                depth=2,
                jurisdiction_filter=query.jurisdiction
            )
            
            result.citation_network = citation_network
            result.models_used.append("citation_analyzer")
            
            logger.info("✅ Citation network analysis completed")
            
        except Exception as e:
            logger.error(f"❌ Error in citation analysis: {e}")
            raise
    
    async def _execute_jurisdiction_research(self, query: ResearchQuery, result: ResearchResult):
        """Execute multi-jurisdiction research"""
        try:
            if not self.jurisdiction_searcher:
                logger.warning("Jurisdiction searcher not available")
                return
                
            logger.info("🌐 Executing multi-jurisdiction research...")
            
            jurisdiction_results = await self.jurisdiction_searcher.search_across_jurisdictions(
                query=query.query_text,
                jurisdictions=[query.jurisdiction, "US", "UK", "CA", "AU"],
                legal_domain=query.legal_domain,
                comparison_mode=True
            )
            
            # Add jurisdiction results to main results
            result.results.extend(jurisdiction_results)
            result.models_used.append("jurisdiction_searcher")
            
            logger.info(f"✅ Multi-jurisdiction research completed with {len(jurisdiction_results)} results")
            
        except Exception as e:
            logger.error(f"❌ Error in jurisdiction research: {e}")
            raise
    
    async def _execute_memo_generation(self, query: ResearchQuery, result: ResearchResult):
        """Execute legal memo generation"""
        try:
            if not self.memo_generator:
                logger.warning("Memo generator not available")
                return
                
            logger.info("📝 Generating legal research memo...")
            
            memo_data = {
                "query": query.query_text,
                "precedents": result.precedent_matches,
                "citations": result.citation_network,
                "jurisdiction": query.jurisdiction,
                "legal_issues": query.legal_issues
            }
            
            generated_memo = await self.memo_generator.generate_research_memo(
                memo_data=memo_data,
                memo_type="comprehensive",
                format_style="traditional"
            )
            
            result.generated_memo = generated_memo.get("content", "")
            result.models_used.append("memo_generator")
            
            logger.info("✅ Legal research memo generated")
            
        except Exception as e:
            logger.error(f"❌ Error in memo generation: {e}")
            raise
    
    async def _execute_argument_structuring(self, query: ResearchQuery, result: ResearchResult):
        """Execute legal argument structuring"""
        try:
            if not self.argument_structurer:
                logger.warning("Argument structurer not available")
                return
                
            logger.info("⚖️ Structuring legal arguments...")
            
            argument_data = {
                "legal_question": query.query_text,
                "precedents": result.precedent_matches,
                "jurisdiction": query.jurisdiction,
                "case_type": query.case_type
            }
            
            structured_arguments = await self.argument_structurer.structure_legal_arguments(
                argument_data=argument_data,
                argument_strength="strong",
                include_counterarguments=True
            )
            
            result.legal_arguments = structured_arguments
            result.models_used.append("argument_structurer")
            
            logger.info("✅ Legal arguments structured")
            
        except Exception as e:
            logger.error(f"❌ Error in argument structuring: {e}")
            raise
    
    async def _execute_quality_assessment(self, query: ResearchQuery, result: ResearchResult):
        """Execute comprehensive quality assessment"""
        try:
            if not self.quality_scorer:
                logger.warning("Quality scorer not available")
                return
                
            logger.info("🎯 Executing quality assessment...")
            
            quality_assessment = await self.quality_scorer.assess_research_quality(
                research_data={
                    "query": query,
                    "results": result.results,
                    "precedents": result.precedent_matches,
                    "citations": result.citation_network,
                    "memo": result.generated_memo,
                    "arguments": result.legal_arguments
                }
            )
            
            result.confidence_score = quality_assessment.get("confidence_score", 0.0)
            result.completeness_score = quality_assessment.get("completeness_score", 0.0)
            result.authority_score = quality_assessment.get("authority_score", 0.0)
            result.models_used.append("quality_scorer")
            
            logger.info(f"✅ Quality assessment completed - Confidence: {result.confidence_score:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error in quality assessment: {e}")
            raise
    
    async def _finalize_research_results(self, query: ResearchQuery, result: ResearchResult):
        """Finalize and optimize research results"""
        try:
            logger.info("🎯 Finalizing research results...")
            
            # Aggregate all results
            all_results = result.results.copy()
            
            # Add precedent matches to main results if not already included
            for precedent in result.precedent_matches:
                if precedent not in all_results:
                    all_results.append(precedent)
            
            # Sort results by relevance/authority score
            all_results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
            
            # Limit to max results
            result.results = all_results[:query.max_results]
            result.sources_count = len(result.results)
            
            # Set expiration time for caching (24 hours)
            result.expires_at = datetime.utcnow() + timedelta(hours=24)
            result.updated_at = datetime.utcnow()
            
            logger.info(f"✅ Research results finalized with {result.sources_count} sources")
            
        except Exception as e:
            logger.error(f"❌ Error finalizing results: {e}")
            raise
    
    async def get_research_status(self, research_id: str) -> Dict[str, Any]:
        """Get status of a research operation"""
        try:
            # Check active sessions
            if research_id in self.active_sessions:
                session = self.active_sessions[research_id]
                return {
                    "research_id": research_id,
                    "status": session.get("status", "processing"),
                    "progress": session.get("progress", 0.0),
                    "estimated_completion": session.get("estimated_completion"),
                    "current_operation": session.get("current_operation", ""),
                    "results_count": session.get("results_count", 0)
                }
            
            # Check database for completed research
            if self.db:
                stored_result = await self.db.research_sessions.find_one({"research_id": research_id})
                if stored_result:
                    return {
                        "research_id": research_id,
                        "status": stored_result.get("status", "completed"),
                        "completed_at": stored_result.get("completed_at"),
                        "results_count": stored_result.get("results_count", 0),
                        "confidence_score": stored_result.get("confidence_score", 0.0)
                    }
            
            # Check cache
            if research_id in self.result_cache:
                cached = self.result_cache[research_id]
                return {
                    "research_id": research_id,
                    "status": "cached",
                    "cached_at": cached.get("cached_at"),
                    "results_count": len(cached.get("results", [])),
                    "confidence_score": cached.get("confidence_score", 0.0)
                }
            
            return {
                "research_id": research_id,
                "status": "not_found",
                "message": "Research session not found"
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting research status: {e}")
            raise
    
    async def _check_cache(self, query: ResearchQuery) -> Optional[ResearchResult]:
        """Check if research results are cached"""
        try:
            cache_key = self._generate_cache_key(query)
            
            if cache_key in self.result_cache:
                cached_data = self.result_cache[cache_key]
                
                # Check if cache is still valid
                if cached_data.get("expires_at") and cached_data["expires_at"] > datetime.utcnow():
                    # Convert cached data back to ResearchResult
                    result = ResearchResult(**cached_data["result"])
                    logger.info(f"📋 Cache hit for query: {query.id}")
                    return result
                else:
                    # Remove expired cache
                    del self.result_cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error checking cache: {e}")
            return None
    
    async def _cache_result(self, query: ResearchQuery, result: ResearchResult):
        """Cache research results for future use"""
        try:
            cache_key = self._generate_cache_key(query)
            
            cache_data = {
                "result": asdict(result),
                "cached_at": datetime.utcnow(),
                "expires_at": result.expires_at,
                "cache_key": cache_key
            }
            
            self.result_cache[cache_key] = cache_data
            logger.info(f"📋 Cached results for query: {query.id}")
            
        except Exception as e:
            logger.error(f"❌ Error caching results: {e}")
    
    def _generate_cache_key(self, query: ResearchQuery) -> str:
        """Generate cache key from query parameters"""
        key_components = [
            query.query_text,
            query.research_type.value,
            query.jurisdiction,
            query.legal_domain,
            str(query.max_results),
            str(query.min_confidence)
        ]
        return hash(tuple(key_components))
    
    async def _store_research_session(self, query: ResearchQuery, result: ResearchResult):
        """Store research session in database"""
        try:
            if not self.db:
                return
            
            session_data = {
                "research_id": result.id,
                "query_id": query.id,
                "query_text": query.query_text,
                "research_type": query.research_type.value,
                "jurisdiction": query.jurisdiction,
                "legal_domain": query.legal_domain,
                "status": result.status.value,
                "results_count": result.sources_count,
                "confidence_score": result.confidence_score,
                "completeness_score": result.completeness_score,
                "authority_score": result.authority_score,
                "processing_time": result.processing_time,
                "models_used": result.models_used,
                "created_at": query.created_at,
                "completed_at": result.updated_at
            }
            
            await self.db.research_sessions.insert_one(session_data)
            logger.info(f"💾 Research session stored for query: {query.id}")
            
        except Exception as e:
            logger.error(f"❌ Error storing research session: {e}")
    
    async def _load_performance_data(self):
        """Load performance metrics and cache data"""
        try:
            # Initialize performance tracking
            self.performance_metrics = {
                "total_queries": 0,
                "average_processing_time": 0.0,
                "cache_hit_rate": 0.0,
                "quality_scores": []
            }
            
            logger.info("📊 Performance tracking initialized")
            
        except Exception as e:
            logger.error(f"❌ Error loading performance data: {e}")
    
    async def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics"""
        try:
            stats = {
                "engine_id": self.session_id,
                "initialized_components": {
                    "precedent_matcher": self.precedent_matcher is not None,
                    "citation_analyzer": self.citation_analyzer is not None,
                    "memo_generator": self.memo_generator is not None,
                    "argument_structurer": self.argument_structurer is not None,
                    "jurisdiction_searcher": self.jurisdiction_searcher is not None,
                    "quality_scorer": self.quality_scorer is not None
                },
                "performance_metrics": self.performance_metrics,
                "cache_stats": {
                    "cached_results": len(self.result_cache),
                    "cache_hit_rate": self.performance_metrics.get("cache_hit_rate", 0.0)
                },
                "active_sessions": len(self.active_sessions),
                "database_connected": self.db is not None,
                "ai_clients": {
                    "gemini": self.gemini_api_key is not None,
                    "groq": self.groq_client is not None
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting engine stats: {e}")
            return {"error": str(e)}


# Global engine instance
_research_engine = None

async def get_research_engine() -> AdvancedLegalResearchEngine:
    """Get initialized research engine instance"""
    global _research_engine
    
    if _research_engine is None:
        _research_engine = AdvancedLegalResearchEngine()
        await _research_engine.initialize()
    
    return _research_engine


if __name__ == "__main__":
    # Test the research engine
    async def test_engine():
        engine = await get_research_engine()
        
        # Test query
        query = ResearchQuery(
            query_text="What are the elements of breach of contract under US law?",
            research_type=ResearchType.COMPREHENSIVE,
            jurisdiction="US",
            legal_domain="contract_law",
            legal_issues=["breach of contract", "damages", "remedies"]
        )
        
        result = await engine.coordinate_research(query)
        print(f"Research completed with {result.sources_count} sources")
        print(f"Confidence score: {result.confidence_score:.2f}")
        print(f"Processing time: {result.processing_time:.2f}s")
    
    asyncio.run(test_engine())