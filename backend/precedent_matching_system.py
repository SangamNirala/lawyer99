"""
Precedent Matching System - AI-Powered Precedent Analysis

This module implements advanced precedent matching capabilities using semantic
similarity, multi-dimensional case analysis, and AI-powered legal principle extraction.

Key Features:
- Multi-dimensional case similarity scoring (factual, legal, procedural, jurisdictional)  
- Legal principle extraction using Gemini AI
- Precedent relevance ranking with confidence scoring
- Integration with CourtListener data and legal knowledge base
- Semantic similarity using advanced embeddings
"""

import asyncio
import json
import logging
import numpy as np
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

import google.generativeai as genai
from groq import Groq
import httpx
import os
from motor.motor_asyncio import AsyncIOMotorClient

# Import embeddings with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

import faiss
import re
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimilarityDimension(Enum):
    """Dimensions for case similarity analysis"""
    FACTUAL = "factual"
    LEGAL = "legal" 
    PROCEDURAL = "procedural"
    JURISDICTIONAL = "jurisdictional"
    TEMPORAL = "temporal"


class MatchType(Enum):
    """Types of precedent matches"""
    IDENTICAL = "identical"
    HIGHLY_SIMILAR = "highly_similar"
    MODERATELY_SIMILAR = "moderately_similar"
    DISTINGUISHABLE = "distinguishable"
    ANALOGOUS = "analogous"


@dataclass
class LegalPrinciple:
    """Extracted legal principle from case analysis"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    principle_text: str = ""
    legal_domain: str = ""
    authority_level: str = ""
    jurisdiction: str = ""
    source_case: str = ""
    confidence_score: float = 0.0
    supporting_citations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SimilarityScore:
    """Multi-dimensional similarity score"""
    factual_similarity: float = 0.0
    legal_similarity: float = 0.0
    procedural_similarity: float = 0.0
    jurisdictional_similarity: float = 0.0
    temporal_similarity: float = 0.0
    overall_similarity: float = 0.0
    confidence_score: float = 0.0
    
    def calculate_overall(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted overall similarity score"""
        if weights is None:
            weights = {
                'factual': 0.35,
                'legal': 0.30,
                'procedural': 0.15,
                'jurisdictional': 0.15,
                'temporal': 0.05
            }
        
        overall = (
            self.factual_similarity * weights.get('factual', 0.0) +
            self.legal_similarity * weights.get('legal', 0.0) +
            self.procedural_similarity * weights.get('procedural', 0.0) +
            self.jurisdictional_similarity * weights.get('jurisdictional', 0.0) +
            self.temporal_similarity * weights.get('temporal', 0.0)
        )
        
        self.overall_similarity = min(max(overall, 0.0), 1.0)
        return self.overall_similarity


@dataclass
class PrecedentMatch:
    """Comprehensive precedent match result"""
    case_id: str = ""
    case_title: str = ""
    citation: str = ""
    court: str = ""
    jurisdiction: str = ""
    decision_date: Optional[datetime] = None
    
    # Case content
    case_summary: str = ""
    legal_issues: List[str] = field(default_factory=list)
    holdings: List[str] = field(default_factory=list)
    key_facts: List[str] = field(default_factory=list)
    
    # Similarity analysis
    similarity_scores: SimilarityScore = field(default_factory=SimilarityScore)
    match_type: MatchType = MatchType.MODERATELY_SIMILAR
    relevance_score: float = 0.0
    
    # Legal principles
    extracted_principles: List[LegalPrinciple] = field(default_factory=list)
    
    # Authority and citation metrics
    citation_count: int = 0
    authority_score: float = 0.0
    
    # Match metadata
    match_reasoning: str = ""
    distinguishing_factors: List[str] = field(default_factory=list)
    supporting_quotes: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)


class PrecedentMatchingSystem:
    """
    Advanced precedent matching system using AI and semantic similarity.
    
    This system provides enterprise-grade precedent analysis with multi-dimensional
    similarity scoring and legal principle extraction.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Precedent Matching System"""
        self.config = config or {}
        
        # API Configuration
        self.gemini_api_key = os.environ.get('GEMINI_API_KEY')
        self.groq_api_key = os.environ.get('GROQ_API_KEY')
        self.courtlistener_api_key = os.environ.get('COURTLISTENER_API_KEY')
        
        # MongoDB Configuration  
        self.mongo_url = os.environ.get('MONGO_URL')
        self.db_name = os.environ.get('DB_NAME', 'legal_research_db')
        
        # Initialize clients
        self.db_client = None
        self.db = None
        self.groq_client = None
        self.courtlistener_client = None
        
        # Embeddings model
        self.embeddings_model = None
        self.embedding_dimension = 384
        
        # FAISS index for fast similarity search
        self.faiss_index = None
        self.case_metadata = {}
        
        # Caching
        self.similarity_cache = {}
        self.principle_cache = {}
        
        # Performance metrics
        self.performance_metrics = {
            "total_matches": 0,
            "average_match_time": 0.0,
            "cache_hits": 0
        }
        
        logger.info("🔍 Precedent Matching System initialized")
    
    async def initialize(self):
        """Initialize all system components"""
        try:
            logger.info("🚀 Initializing Precedent Matching System...")
            
            # Initialize AI clients
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                logger.info("✅ Gemini AI client initialized")
            
            if self.groq_api_key:
                self.groq_client = Groq(api_key=self.groq_api_key)
                logger.info("✅ Groq AI client initialized")
            
            # Initialize CourtListener client
            if self.courtlistener_api_key:
                self.courtlistener_client = httpx.AsyncClient(
                    base_url="https://www.courtlistener.com/api/rest/v3",
                    headers={"Authorization": f"Token {self.courtlistener_api_key}"}
                )
                logger.info("✅ CourtListener client initialized")
            
            # Initialize MongoDB connection
            if self.mongo_url:
                self.db_client = AsyncIOMotorClient(self.mongo_url)
                self.db = self.db_client[self.db_name]
                logger.info("✅ MongoDB connection established")
            
            # Initialize embeddings model
            await self._initialize_embeddings()
            
            # Initialize FAISS index
            await self._initialize_faiss_index()
            
            # Load existing case data
            await self._load_case_database()
            
            logger.info("🎉 Precedent Matching System fully initialized!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing precedent matching system: {e}")
            raise
    
    async def _initialize_embeddings(self):
        """Initialize the embeddings model for semantic similarity"""
        try:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("⚠️ Sentence transformers not available, using fallback")
                return
            
            logger.info("🔤 Loading legal embeddings model...")
            # Use a model optimized for legal text
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dimension = 384
            
            logger.info("✅ Embeddings model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing embeddings: {e}")
            self.embeddings_model = None
    
    async def _initialize_faiss_index(self):
        """Initialize FAISS index for fast similarity search"""
        try:
            logger.info("📊 Initializing FAISS index for case similarity...")
            
            # Create FAISS index for cosine similarity
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
            
            logger.info("✅ FAISS index initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing FAISS index: {e}")
            raise
    
    async def _load_case_database(self):
        """Load existing case database for precedent matching"""
        try:
            logger.info("📚 Loading case database...")
            
            # Load from legal knowledge base if available
            knowledge_base_path = "/app/legal_knowledge_base.json"
            if os.path.exists(knowledge_base_path):
                with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                    cases = json.load(f)
                
                # Process cases for FAISS indexing
                await self._process_cases_for_indexing(cases)
                
                logger.info(f"✅ Loaded {len(cases)} cases into precedent database")
            else:
                logger.warning("⚠️ Legal knowledge base not found, using CourtListener API")
                await self._load_courtlistener_cases()
            
        except Exception as e:
            logger.error(f"❌ Error loading case database: {e}")
    
    async def _process_cases_for_indexing(self, cases: List[Dict]):
        """Process cases and create embeddings for FAISS indexing"""
        try:
            if not self.embeddings_model:
                logger.warning("⚠️ Embeddings model not available for indexing")
                return
            
            logger.info("🔍 Processing cases for embeddings indexing...")
            
            case_texts = []
            case_ids = []
            
            for i, case in enumerate(cases):
                # Combine relevant text fields for embedding
                case_text = f"{case.get('title', '')} {case.get('content', '')}".strip()
                if case_text:
                    case_texts.append(case_text[:2000])  # Limit text length
                    case_id = case.get('id', f'case_{i}')
                    case_ids.append(case_id)
                    
                    # Store case metadata
                    self.case_metadata[case_id] = case
            
            if case_texts:
                # Generate embeddings
                embeddings = self.embeddings_model.encode(
                    case_texts, 
                    convert_to_numpy=True,
                    show_progress_bar=True
                )
                
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings)
                
                # Add to FAISS index
                self.faiss_index.add(embeddings)
                
                logger.info(f"✅ Indexed {len(case_texts)} cases with embeddings")
            
        except Exception as e:
            logger.error(f"❌ Error processing cases for indexing: {e}")
    
    async def _load_courtlistener_cases(self):
        """Load cases from CourtListener API"""
        try:
            if not self.courtlistener_client:
                logger.warning("⚠️ CourtListener client not available")
                return
            
            logger.info("🔄 Loading recent cases from CourtListener...")
            
            # Fetch recent landmark cases
            params = {
                "type": "o",  # Opinions
                "filed_after": "2020-01-01",
                "precedential_status": "Published",
                "ordering": "-date_filed",
                "page_size": 100
            }
            
            response = await self.courtlistener_client.get("/search/", params=params)
            if response.status_code == 200:
                data = response.json()
                cases = data.get("results", [])
                
                # Convert CourtListener format to our format
                converted_cases = []
                for case in cases:
                    converted_case = {
                        "id": case.get("id", ""),
                        "title": case.get("caseName", ""),
                        "content": case.get("snippet", ""),
                        "citation": case.get("citation", ""),
                        "court": case.get("court", ""),
                        "date_filed": case.get("dateFiled", ""),
                        "jurisdiction": self._extract_jurisdiction(case.get("court", "")),
                        "legal_domain": "general",
                        "source": "CourtListener"
                    }
                    converted_cases.append(converted_case)
                
                # Process for indexing
                await self._process_cases_for_indexing(converted_cases)
                
                logger.info(f"✅ Loaded {len(cases)} cases from CourtListener")
            
        except Exception as e:
            logger.error(f"❌ Error loading CourtListener cases: {e}")
    
    def _extract_jurisdiction(self, court_name: str) -> str:
        """Extract jurisdiction from court name"""
        court_lower = court_name.lower()
        
        if "supreme court" in court_lower and "united states" in court_lower:
            return "US_Supreme"
        elif "circuit" in court_lower:
            return "US_Federal"
        elif any(state in court_lower for state in ["california", "new york", "texas", "florida"]):
            return "US_State"
        else:
            return "US"
    
    async def find_similar_cases(self, query_case: Dict[str, Any], 
                               filters: Dict[str, Any] = None) -> List[PrecedentMatch]:
        """
        Find similar cases using multi-dimensional similarity analysis.
        
        Args:
            query_case: Case to find precedents for
            filters: Optional filters for jurisdiction, date range, etc.
            
        Returns:
            List[PrecedentMatch]: Ranked list of similar cases
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Finding similar cases for query...")
            
            # Extract query information
            query_facts = query_case.get("facts", "")
            query_issues = query_case.get("legal_issues", [])
            
            # Apply filters
            filters = filters or {}
            max_results = filters.get("max_results", 20)
            min_similarity = filters.get("min_similarity", 0.6)
            
            # Perform semantic similarity search
            similar_cases = await self._semantic_similarity_search(
                query_text=query_facts,
                max_results=max_results * 2  # Get more candidates for filtering
            )
            
            # Perform detailed similarity analysis
            precedent_matches = []
            
            for case_data in similar_cases:
                match = await self._analyze_case_similarity(
                    query_case=query_case,
                    candidate_case=case_data,
                    filters=filters
                )
                
                if match and match.similarity_scores.overall_similarity >= min_similarity:
                    precedent_matches.append(match)
            
            # Sort by relevance score
            precedent_matches.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Limit to requested number of results
            precedent_matches = precedent_matches[:max_results]
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, len(precedent_matches))
            
            logger.info(f"✅ Found {len(precedent_matches)} similar cases in {processing_time:.2f}s")
            
            return precedent_matches
            
        except Exception as e:
            logger.error(f"❌ Error finding similar cases: {e}")
            raise
    
    async def _semantic_similarity_search(self, query_text: str, max_results: int = 50) -> List[Dict]:
        """Perform semantic similarity search using FAISS"""
        try:
            if not self.embeddings_model or not self.faiss_index:
                logger.warning("⚠️ Embeddings or FAISS index not available")
                return []
            
            # Generate query embedding
            query_embedding = self.embeddings_model.encode([query_text], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            similarities, indices = self.faiss_index.search(query_embedding, max_results)
            
            # Retrieve case metadata
            similar_cases = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(self.case_metadata) and similarity > 0.0:
                    case_id = list(self.case_metadata.keys())[idx]
                    case_data = self.case_metadata[case_id].copy()
                    case_data['semantic_similarity'] = float(similarity)
                    similar_cases.append(case_data)
            
            return similar_cases
            
        except Exception as e:
            logger.error(f"❌ Error in semantic similarity search: {e}")
            return []
    
    async def analyze_case_similarity(self, case1: Dict, case2: Dict) -> SimilarityScore:
        """
        Analyze multi-dimensional similarity between two cases.
        
        Args:
            case1: First case for comparison
            case2: Second case for comparison
            
        Returns:
            SimilarityScore: Detailed similarity analysis
        """
        try:
            logger.info("🔍 Analyzing case similarity...")
            
            similarity = SimilarityScore()
            
            # Factual similarity
            similarity.factual_similarity = await self._calculate_factual_similarity(case1, case2)
            
            # Legal similarity  
            similarity.legal_similarity = await self._calculate_legal_similarity(case1, case2)
            
            # Procedural similarity
            similarity.procedural_similarity = await self._calculate_procedural_similarity(case1, case2)
            
            # Jurisdictional similarity
            similarity.jurisdictional_similarity = await self._calculate_jurisdictional_similarity(case1, case2)
            
            # Temporal similarity
            similarity.temporal_similarity = await self._calculate_temporal_similarity(case1, case2)
            
            # Calculate overall similarity
            similarity.calculate_overall()
            
            # Calculate confidence score based on data completeness
            similarity.confidence_score = self._calculate_confidence_score(case1, case2, similarity)
            
            logger.info(f"✅ Similarity analysis completed - Overall: {similarity.overall_similarity:.2f}")
            
            return similarity
            
        except Exception as e:
            logger.error(f"❌ Error analyzing case similarity: {e}")
            return SimilarityScore()
    
    async def _analyze_case_similarity(self, query_case: Dict, candidate_case: Dict, 
                                     filters: Dict = None) -> Optional[PrecedentMatch]:
        """Analyze similarity and create precedent match"""
        try:
            # Perform similarity analysis
            similarity_scores = await self.analyze_case_similarity(query_case, candidate_case)
            
            # Apply filters
            if filters:
                jurisdiction = filters.get("jurisdiction")
                if jurisdiction and candidate_case.get("jurisdiction") != jurisdiction:
                    return None
                
                date_range = filters.get("date_range")
                if date_range:
                    case_date = candidate_case.get("date_filed")
                    if not self._date_in_range(case_date, date_range):
                        return None
            
            # Create precedent match
            match = PrecedentMatch(
                case_id=candidate_case.get("id", ""),
                case_title=candidate_case.get("title", ""),
                citation=candidate_case.get("citation", ""),
                court=candidate_case.get("court", ""),
                jurisdiction=candidate_case.get("jurisdiction", ""),
                case_summary=candidate_case.get("content", "")[:500],
                similarity_scores=similarity_scores
            )
            
            # Determine match type
            match.match_type = self._determine_match_type(similarity_scores.overall_similarity)
            
            # Calculate relevance score (combines similarity and authority)
            match.relevance_score = self._calculate_relevance_score(
                similarity_scores, 
                candidate_case.get("citation_count", 0)
            )
            
            # Extract legal principles
            match.extracted_principles = await self.extract_legal_principles(
                candidate_case.get("content", "")
            )
            
            # Generate match reasoning
            match.match_reasoning = await self._generate_match_reasoning(
                query_case, candidate_case, similarity_scores
            )
            
            return match
            
        except Exception as e:
            logger.error(f"❌ Error analyzing case similarity: {e}")
            return None
    
    async def _calculate_factual_similarity(self, case1: Dict, case2: Dict) -> float:
        """Calculate factual similarity between cases"""
        try:
            facts1 = case1.get("facts", case1.get("content", ""))
            facts2 = case2.get("facts", case2.get("content", ""))
            
            if not facts1 or not facts2:
                return 0.0
            
            # Use embeddings for factual similarity if available
            if self.embeddings_model:
                embeddings = self.embeddings_model.encode([facts1[:1000], facts2[:1000]])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return max(0.0, similarity)
            
            # Fallback to text overlap
            return self._text_similarity(facts1, facts2)
            
        except Exception as e:
            logger.error(f"❌ Error calculating factual similarity: {e}")
            return 0.0
    
    async def _calculate_legal_similarity(self, case1: Dict, case2: Dict) -> float:
        """Calculate legal similarity based on legal issues and holdings"""
        try:
            issues1 = case1.get("legal_issues", [])
            issues2 = case2.get("legal_issues", [])
            
            if not issues1 or not issues2:
                # Fallback to content analysis
                content1 = case1.get("content", "")
                content2 = case2.get("content", "")
                return self._text_similarity(content1, content2)
            
            # Calculate overlap in legal issues
            issues1_set = set([issue.lower().strip() for issue in issues1])
            issues2_set = set([issue.lower().strip() for issue in issues2])
            
            if not issues1_set or not issues2_set:
                return 0.0
            
            intersection = len(issues1_set & issues2_set)
            union = len(issues1_set | issues2_set)
            
            jaccard_similarity = intersection / union if union > 0 else 0.0
            
            return jaccard_similarity
            
        except Exception as e:
            logger.error(f"❌ Error calculating legal similarity: {e}")
            return 0.0
    
    async def _calculate_procedural_similarity(self, case1: Dict, case2: Dict) -> float:
        """Calculate procedural similarity (court level, case type, etc.)"""
        try:
            court1 = case1.get("court", "").lower()
            court2 = case2.get("court", "").lower()
            
            case_type1 = case1.get("case_type", "").lower()
            case_type2 = case2.get("case_type", "").lower()
            
            similarity_score = 0.0
            
            # Court level similarity
            if court1 and court2:
                if court1 == court2:
                    similarity_score += 0.5
                elif self._same_court_level(court1, court2):
                    similarity_score += 0.3
            
            # Case type similarity
            if case_type1 and case_type2:
                if case_type1 == case_type2:
                    similarity_score += 0.5
                elif self._similar_case_type(case_type1, case_type2):
                    similarity_score += 0.3
            
            return min(1.0, similarity_score)
            
        except Exception as e:
            logger.error(f"❌ Error calculating procedural similarity: {e}")
            return 0.0
    
    async def _calculate_jurisdictional_similarity(self, case1: Dict, case2: Dict) -> float:
        """Calculate jurisdictional similarity"""
        try:
            jurisdiction1 = case1.get("jurisdiction", "").lower()
            jurisdiction2 = case2.get("jurisdiction", "").lower()
            
            if not jurisdiction1 or not jurisdiction2:
                return 0.5  # Unknown jurisdiction
            
            if jurisdiction1 == jurisdiction2:
                return 1.0
            
            # Check for hierarchical relationships (e.g., federal vs state)
            if "us" in jurisdiction1 and "us" in jurisdiction2:
                return 0.7
            
            return 0.3  # Different jurisdictions
            
        except Exception as e:
            logger.error(f"❌ Error calculating jurisdictional similarity: {e}")
            return 0.0
    
    async def _calculate_temporal_similarity(self, case1: Dict, case2: Dict) -> float:
        """Calculate temporal similarity based on decision dates"""
        try:
            date1_str = case1.get("date_filed") or case1.get("created_at", "")
            date2_str = case2.get("date_filed") or case2.get("created_at", "")
            
            if not date1_str or not date2_str:
                return 0.5  # Unknown dates
            
            # Parse dates
            date1 = self._parse_date(date1_str)
            date2 = self._parse_date(date2_str)
            
            if not date1 or not date2:
                return 0.5
            
            # Calculate time difference in years
            time_diff = abs((date1 - date2).days) / 365.0
            
            # Similarity decreases with time difference
            if time_diff <= 1:
                return 1.0
            elif time_diff <= 5:
                return 0.8
            elif time_diff <= 10:
                return 0.6
            elif time_diff <= 20:
                return 0.4
            else:
                return 0.2
            
        except Exception as e:
            logger.error(f"❌ Error calculating temporal similarity: {e}")
            return 0.5
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate basic text similarity using word overlap"""
        try:
            words1 = set(re.findall(r'\b\w+\b', text1.lower()))
            words2 = set(re.findall(r'\b\w+\b', text2.lower()))
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _same_court_level(self, court1: str, court2: str) -> bool:
        """Check if courts are at the same level"""
        supreme_keywords = ["supreme", "highest"]
        appellate_keywords = ["appellate", "appeal", "circuit"]
        trial_keywords = ["district", "trial", "superior"]
        
        court1_level = self._get_court_level(court1)
        court2_level = self._get_court_level(court2)
        
        return court1_level == court2_level
    
    def _get_court_level(self, court: str) -> str:
        """Determine court level from court name"""
        court_lower = court.lower()
        
        if any(keyword in court_lower for keyword in ["supreme", "highest"]):
            return "supreme"
        elif any(keyword in court_lower for keyword in ["appellate", "appeal", "circuit"]):
            return "appellate"
        elif any(keyword in court_lower for keyword in ["district", "trial", "superior"]):
            return "trial"
        else:
            return "unknown"
    
    def _similar_case_type(self, type1: str, type2: str) -> bool:
        """Check if case types are similar"""
        civil_types = ["civil", "contract", "tort", "property"]
        criminal_types = ["criminal", "felony", "misdemeanor"]
        
        if any(t in type1 for t in civil_types) and any(t in type2 for t in civil_types):
            return True
        if any(t in type1 for t in criminal_types) and any(t in type2 for t in criminal_types):
            return True
        
        return False
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object"""
        try:
            # Try common date formats
            formats = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ"]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str[:len(fmt.replace('%f', '123456'))], fmt)
                except ValueError:
                    continue
            
            return None
            
        except Exception:
            return None
    
    def _date_in_range(self, date_str: str, date_range: Dict) -> bool:
        """Check if date falls within specified range"""
        try:
            date = self._parse_date(date_str)
            if not date:
                return True  # If can't parse, don't filter out
            
            start_str = date_range.get("start")
            end_str = date_range.get("end")
            
            if start_str:
                start_date = self._parse_date(start_str)
                if start_date and date < start_date:
                    return False
            
            if end_str:
                end_date = self._parse_date(end_str)
                if end_date and date > end_date:
                    return False
            
            return True
            
        except Exception:
            return True
    
    def _determine_match_type(self, similarity_score: float) -> MatchType:
        """Determine match type based on similarity score"""
        if similarity_score >= 0.95:
            return MatchType.IDENTICAL
        elif similarity_score >= 0.85:
            return MatchType.HIGHLY_SIMILAR
        elif similarity_score >= 0.70:
            return MatchType.MODERATELY_SIMILAR
        elif similarity_score >= 0.55:
            return MatchType.ANALOGOUS
        else:
            return MatchType.DISTINGUISHABLE
    
    def _calculate_relevance_score(self, similarity: SimilarityScore, citation_count: int = 0) -> float:
        """Calculate overall relevance score combining similarity and authority"""
        try:
            # Base score from similarity
            relevance = similarity.overall_similarity * 0.8
            
            # Authority bonus based on citation count
            authority_bonus = min(0.2, (citation_count / 100.0) * 0.2)
            relevance += authority_bonus
            
            # Confidence penalty
            confidence_factor = similarity.confidence_score
            relevance *= confidence_factor
            
            return min(1.0, max(0.0, relevance))
            
        except Exception:
            return similarity.overall_similarity
    
    def _calculate_confidence_score(self, case1: Dict, case2: Dict, similarity: SimilarityScore) -> float:
        """Calculate confidence score based on data completeness and quality"""
        try:
            completeness_score = 0.0
            
            # Check data completeness
            fields_to_check = ["title", "content", "jurisdiction", "court"]
            for field in fields_to_check:
                if case1.get(field) and case2.get(field):
                    completeness_score += 0.25
            
            # Penalize low similarity variance
            similarity_variance = np.var([
                similarity.factual_similarity,
                similarity.legal_similarity,
                similarity.procedural_similarity,
                similarity.jurisdictional_similarity
            ])
            
            variance_factor = min(1.0, similarity_variance + 0.5)
            
            return completeness_score * variance_factor
            
        except Exception:
            return 0.7  # Default confidence
    
    async def extract_legal_principles(self, case_content: str) -> List[LegalPrinciple]:
        """
        Extract legal principles from case content using AI analysis.
        
        Args:
            case_content: Full text of the legal case
            
        Returns:
            List[LegalPrinciple]: Extracted legal principles
        """
        try:
            logger.info("🧠 Extracting legal principles using AI...")
            
            if not case_content or len(case_content.strip()) < 100:
                return []
            
            # Check cache first
            content_hash = hash(case_content[:1000])
            if content_hash in self.principle_cache:
                return self.principle_cache[content_hash]
            
            # Use Gemini for principle extraction
            principles = await self._extract_principles_with_gemini(case_content)
            
            # Cache results
            self.principle_cache[content_hash] = principles
            
            logger.info(f"✅ Extracted {len(principles)} legal principles")
            
            return principles
            
        except Exception as e:
            logger.error(f"❌ Error extracting legal principles: {e}")
            return []
    
    async def _extract_principles_with_gemini(self, case_content: str) -> List[LegalPrinciple]:
        """Extract legal principles using Gemini AI"""
        try:
            prompt = f"""
            Analyze this legal case content and extract the key legal principles:

            {case_content[:2000]}

            Please identify and extract:
            1. Core legal principles established or applied
            2. Rules of law stated by the court
            3. Legal tests or standards articulated
            4. Important precedential value

            Format your response as JSON with this structure:
            {{
                "principles": [
                    {{
                        "principle_text": "The specific legal principle or rule",
                        "legal_domain": "contract_law|tort_law|constitutional_law|criminal_law|etc",
                        "authority_level": "binding|persuasive|informative", 
                        "supporting_reasoning": "Brief explanation of the principle"
                    }}
                ]
            }}

            Focus on principles that would be useful as legal precedent.
            """
            
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1500,
                    temperature=0.1,
                )
            )
            
            # Parse JSON response
            response_text = response.text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group())
                principles = []
                
                for p in data.get("principles", []):
                    principle = LegalPrinciple(
                        principle_text=p.get("principle_text", ""),
                        legal_domain=p.get("legal_domain", "general"),
                        authority_level=p.get("authority_level", "informative"),
                        confidence_score=0.8  # High confidence from AI
                    )
                    principles.append(principle)
                
                return principles
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Error with Gemini principle extraction: {e}")
            return []
    
    async def _generate_match_reasoning(self, query_case: Dict, candidate_case: Dict, 
                                      similarity: SimilarityScore) -> str:
        """Generate reasoning for why cases match"""
        try:
            reasoning_parts = []
            
            # Factual similarity reasoning
            if similarity.factual_similarity > 0.7:
                reasoning_parts.append(f"Strong factual similarity ({similarity.factual_similarity:.2f})")
            
            # Legal similarity reasoning  
            if similarity.legal_similarity > 0.7:
                reasoning_parts.append(f"Similar legal issues ({similarity.legal_similarity:.2f})")
            
            # Jurisdictional reasoning
            if similarity.jurisdictional_similarity > 0.8:
                reasoning_parts.append("Same jurisdiction")
            
            # Overall assessment
            if similarity.overall_similarity > 0.8:
                reasoning_parts.append("Highly relevant precedent")
            elif similarity.overall_similarity > 0.6:
                reasoning_parts.append("Moderately relevant precedent")
            
            return " | ".join(reasoning_parts) if reasoning_parts else "Case shows some similarity"
            
        except Exception as e:
            logger.error(f"❌ Error generating match reasoning: {e}")
            return "Automated similarity analysis"
    
    def _update_performance_metrics(self, processing_time: float, results_count: int):
        """Update system performance metrics"""
        try:
            self.performance_metrics["total_matches"] += 1
            
            # Update average processing time
            current_avg = self.performance_metrics["average_match_time"]
            total_matches = self.performance_metrics["total_matches"]
            
            new_avg = ((current_avg * (total_matches - 1)) + processing_time) / total_matches
            self.performance_metrics["average_match_time"] = new_avg
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance and status statistics"""
        try:
            stats = {
                "system_status": "operational",
                "indexed_cases": len(self.case_metadata),
                "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0,
                "performance_metrics": self.performance_metrics,
                "cache_stats": {
                    "similarity_cache_size": len(self.similarity_cache),
                    "principle_cache_size": len(self.principle_cache)
                },
                "ai_clients": {
                    "gemini_available": self.gemini_api_key is not None,
                    "groq_available": self.groq_client is not None,
                    "courtlistener_available": self.courtlistener_client is not None
                },
                "embeddings_model": "all-MiniLM-L6-v2" if self.embeddings_model else "unavailable"
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting system stats: {e}")
            return {"error": str(e)}


# Global instance
_precedent_matcher = None

async def get_precedent_matcher() -> PrecedentMatchingSystem:
    """Get initialized precedent matching system"""
    global _precedent_matcher
    
    if _precedent_matcher is None:
        _precedent_matcher = PrecedentMatchingSystem()
        await _precedent_matcher.initialize()
    
    return _precedent_matcher


if __name__ == "__main__":
    # Test the precedent matching system
    async def test_system():
        matcher = await get_precedent_matcher()
        
        query_case = {
            "facts": "Plaintiff entered into a contract for web development services. Defendant failed to deliver the website on time, causing business losses.",
            "legal_issues": ["breach of contract", "damages", "specific performance"]
        }
        
        matches = await matcher.find_similar_cases(query_case)
        print(f"Found {len(matches)} similar cases")
        
        for match in matches[:3]:
            print(f"Case: {match.case_title}")
            print(f"Similarity: {match.similarity_scores.overall_similarity:.2f}")
            print(f"Relevance: {match.relevance_score:.2f}")
            print("---")
    
    asyncio.run(test_system())