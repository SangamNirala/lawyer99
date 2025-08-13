"""
Citation Network Analyzer - Citation Relationship Mapping and Network Graph Construction

This module implements advanced citation network analysis capabilities including:
- Citation relationship mapping and network graph construction
- Authority scoring using PageRank-style algorithms for legal citations
- Citation pattern analysis and landmark case identification
- Legal evolution tracing through citation chains
- AI-powered overruling detection and precedent validation

Key Features:
- Network graph construction from citation data
- PageRank authority scoring for cases
- Citation pattern analysis
- Overruling detection and precedent validation
- Legal evolution tracing
"""

import asyncio
import json
import logging
import time
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import networkx as nx

import google.generativeai as genai
from groq import Groq
import httpx
import os
from motor.motor_asyncio import AsyncIOMotorClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CitationRelationshipType(Enum):
    """Types of citation relationships"""
    CITES = "cites"
    OVERRULES = "overrules"
    DISTINGUISHES = "distinguishes"
    FOLLOWS = "follows"
    QUESTIONS = "questions"
    SUPPORTS = "supports"
    CRITICIZED = "criticized"
    EXPLAINED = "explained"


class AuthorityLevel(Enum):
    """Legal authority levels"""
    SUPREME_COURT = "supreme_court"
    APPELLATE_COURT = "appellate_court"
    TRIAL_COURT = "trial_court"
    ADMINISTRATIVE = "administrative"


@dataclass
class CitationNode:
    """Represents a case node in the citation network"""
    case_id: str
    case_title: str = ""
    citation: str = ""
    court: str = ""
    jurisdiction: str = ""
    decision_date: Optional[datetime] = None
    authority_level: AuthorityLevel = AuthorityLevel.TRIAL_COURT
    
    # Citation metrics
    citation_count: int = 0
    incoming_citations: int = 0
    outgoing_citations: int = 0
    authority_score: float = 0.0
    pagerank_score: float = 0.0
    influence_score: float = 0.0
    
    # Network position
    centrality_score: float = 0.0
    betweenness_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    
    # Legal characteristics
    legal_principles: List[str] = field(default_factory=list)
    landmark_status: str = "standard"  # landmark, influential, standard, limited
    temporal_influence: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CitationEdge:
    """Represents a citation relationship between cases"""
    source_case_id: str
    target_case_id: str
    relationship_type: CitationRelationshipType
    strength_score: float = 0.0  # 0.0 to 1.0
    legal_principle: str = ""
    extracted_reasoning: str = ""
    context_snippet: str = ""
    ai_confidence: float = 0.0
    validation_status: str = "ai_validated"  # ai_validated, needs_review, human_verified
    citation_frequency: int = 1
    jurisdictional_impact: Dict[str, float] = field(default_factory=dict)
    temporal_relevance: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CitationNetwork:
    """Complete citation network structure"""
    network_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    nodes: Dict[str, CitationNode] = field(default_factory=dict)
    edges: List[CitationEdge] = field(default_factory=list)
    
    # Network metrics
    total_nodes: int = 0
    total_edges: int = 0
    network_density: float = 0.0
    average_path_length: float = 0.0
    clustering_coefficient: float = 0.0
    
    # Analysis results
    landmark_cases: List[str] = field(default_factory=list)
    authority_ranking: List[Dict[str, Any]] = field(default_factory=list)
    legal_evolution_chains: List[List[str]] = field(default_factory=list)
    overruling_relationships: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    jurisdiction_scope: List[str] = field(default_factory=list)
    time_span: Dict[str, datetime] = field(default_factory=dict)


class CitationNetworkAnalyzer:
    """
    Advanced citation network analyzer for legal research.
    
    This class provides comprehensive citation network analysis including
    relationship mapping, authority scoring, and legal evolution tracking.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Citation Network Analyzer"""
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
        
        # NetworkX graph for analysis
        self.citation_graph = nx.DiGraph()
        
        # Caching
        self.authority_cache = {}
        self.network_cache = {}
        
        # Performance metrics
        self.performance_metrics = {
            "networks_analyzed": 0,
            "authority_calculations": 0,
            "overruling_detections": 0,
            "average_analysis_time": 0.0
        }
        
        logger.info("📊 Citation Network Analyzer initialized")
    
    async def initialize(self):
        """Initialize all analyzer components"""
        try:
            logger.info("🚀 Initializing Citation Network Analyzer...")
            
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
                await self._ensure_collections_exist()
                logger.info("✅ MongoDB connection established")
            
            logger.info("🎉 Citation Network Analyzer fully initialized!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing citation network analyzer: {e}")
            raise
    
    async def _ensure_collections_exist(self):
        """Ensure required MongoDB collections exist with proper indexing"""
        try:
            # Create indexes for citation_network collection
            await self.db.citation_network.create_index("case_id", unique=True)
            await self.db.citation_network.create_index([("authority_score", -1)])
            await self.db.citation_network.create_index([("pagerank_score", -1)])
            await self.db.citation_network.create_index("jurisdiction")
            
            # Create indexes for precedent_relationships collection
            await self.db.precedent_relationships.create_index([("source_case_id", 1), ("target_case_id", 1)])
            await self.db.precedent_relationships.create_index("relationship_type")
            await self.db.precedent_relationships.create_index([("strength_score", -1)])
            
            logger.info("✅ MongoDB collections and indexes ensured")
            
        except Exception as e:
            logger.error(f"❌ Error ensuring collections: {e}")
    
    async def build_citation_network(self, cases: List[Dict[str, Any]], 
                                   depth: int = 2, jurisdiction_filter: str = None) -> Dict[str, Any]:
        """
        Build citation network from a set of cases.
        
        Args:
            cases: List of cases to build network from
            depth: Depth of citation analysis (1-3)
            jurisdiction_filter: Optional jurisdiction filter
            
        Returns:
            Dict containing the citation network analysis
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Building citation network for {len(cases)} cases...")
            
            # Initialize network
            network = CitationNetwork()
            network.jurisdiction_scope = [jurisdiction_filter] if jurisdiction_filter else []
            
            # Process initial cases
            for case in cases:
                await self._add_case_to_network(case, network)
            
            # Expand network through citation analysis
            await self._expand_network_by_citations(network, depth, jurisdiction_filter)
            
            # Analyze network structure
            await self._analyze_network_structure(network)
            
            # Calculate authority scores
            await self._calculate_authority_scores(network)
            
            # Detect landmark cases
            await self._identify_landmark_cases(network)
            
            # Trace legal evolution
            await self._trace_legal_evolution(network)
            
            # Detect overruling relationships
            await self._detect_overruling_relationships(network)
            
            # Store network in database
            await self._store_citation_network(network)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, network)
            
            logger.info(f"✅ Citation network built with {network.total_nodes} nodes and {network.total_edges} edges in {processing_time:.2f}s")
            
            return self._serialize_network(network)
            
        except Exception as e:
            logger.error(f"❌ Error building citation network: {e}")
            raise
    
    async def _add_case_to_network(self, case: Dict[str, Any], network: CitationNetwork):
        """Add a case as a node to the citation network"""
        try:
            case_id = case.get("case_id", case.get("id", str(uuid.uuid4())))
            
            if case_id in network.nodes:
                return  # Already added
            
            # Extract case information
            decision_date = self._parse_date(case.get("decision_date", case.get("date_filed", "")))
            authority_level = self._determine_authority_level(case.get("court", ""))
            
            # Create citation node
            node = CitationNode(
                case_id=case_id,
                case_title=case.get("case_title", case.get("title", "")),
                citation=case.get("citation", ""),
                court=case.get("court", ""),
                jurisdiction=case.get("jurisdiction", ""),
                decision_date=decision_date,
                authority_level=authority_level,
                legal_principles=case.get("legal_issues", [])
            )
            
            network.nodes[case_id] = node
            network.total_nodes += 1
            
        except Exception as e:
            logger.error(f"❌ Error adding case to network: {e}")
    
    async def _expand_network_by_citations(self, network: CitationNetwork, depth: int, jurisdiction_filter: str):
        """Expand network by following citation relationships"""
        try:
            logger.info(f"🔗 Expanding network through citations (depth: {depth})...")
            
            current_cases = set(network.nodes.keys())
            
            for current_depth in range(depth):
                new_cases = set()
                
                for case_id in current_cases:
                    # Get citations for this case
                    citations = await self._extract_case_citations(case_id, jurisdiction_filter)
                    
                    for citation in citations:
                        cited_case_id = citation.get("cited_case_id")
                        if cited_case_id and cited_case_id not in network.nodes:
                            # Add new case to network
                            cited_case_data = await self._fetch_case_data(cited_case_id)
                            if cited_case_data:
                                await self._add_case_to_network(cited_case_data, network)
                                new_cases.add(cited_case_id)
                        
                        # Add citation edge
                        if cited_case_id:
                            await self._add_citation_edge(case_id, cited_case_id, citation, network)
                
                current_cases = new_cases
                if not current_cases:
                    break
            
            logger.info(f"✅ Network expanded to {network.total_nodes} nodes")
            
        except Exception as e:
            logger.error(f"❌ Error expanding network: {e}")
    
    async def _extract_case_citations(self, case_id: str, jurisdiction_filter: str = None) -> List[Dict]:
        """Extract citations from a case"""
        try:
            # Check if we have citation data in database
            stored_citations = await self.db.precedent_relationships.find(
                {"source_case_id": case_id}
            ).to_list(length=100)
            
            if stored_citations:
                return [self._convert_stored_citation(c) for c in stored_citations]
            
            # Fetch from CourtListener if available
            if self.courtlistener_client:
                return await self._fetch_courtlistener_citations(case_id, jurisdiction_filter)
            
            # Use AI to extract citations from case content
            case_data = await self._fetch_case_data(case_id)
            if case_data and case_data.get("content"):
                return await self._extract_citations_with_ai(case_data["content"])
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Error extracting citations for case {case_id}: {e}")
            return []
    
    async def _fetch_courtlistener_citations(self, case_id: str, jurisdiction_filter: str = None) -> List[Dict]:
        """Fetch citations from CourtListener API"""
        try:
            params = {"ordering": "-date_created", "page_size": 50}
            if jurisdiction_filter:
                params["court__jurisdiction"] = jurisdiction_filter
            
            response = await self.courtlistener_client.get(f"/search/?q=cites:({case_id})", params=params)
            
            if response.status_code == 200:
                data = response.json()
                citations = []
                
                for result in data.get("results", []):
                    citations.append({
                        "cited_case_id": result.get("id"),
                        "citation_context": result.get("snippet", ""),
                        "relationship_type": "cites",
                        "court": result.get("court", ""),
                        "date": result.get("dateFiled", "")
                    })
                
                return citations
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Error fetching CourtListener citations: {e}")
            return []
    
    async def _extract_citations_with_ai(self, case_content: str) -> List[Dict]:
        """Extract citations from case content using AI"""
        try:
            prompt = f"""
            Analyze this legal case content and extract all citations to other cases:
            
            {case_content[:3000]}
            
            For each citation found, identify:
            1. The cited case name or identifier
            2. The type of citation relationship (cites, follows, distinguishes, overrules, etc.)
            3. The legal principle or reasoning context
            4. The strength of the relationship (0.0 to 1.0)
            
            Format as JSON array:
            [
                {{
                    "cited_case_id": "case identifier or name",
                    "relationship_type": "cites|follows|distinguishes|overrules|questions|supports",
                    "legal_principle": "legal principle discussed",
                    "context_snippet": "relevant text excerpt",
                    "strength_score": 0.8,
                    "confidence": 0.9
                }}
            ]
            """
            
            content = await self._generate_content_with_ai(prompt, max_tokens=1500)
            
            if content:
                try:
                    import re
                    json_match = re.search(r'\[.*\]', content, re.DOTALL)
                    if json_match:
                        citations = json.loads(json_match.group())
                        return citations
                except json.JSONDecodeError:
                    logger.warning("⚠️ Could not parse AI citation extraction response")
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Error extracting citations with AI: {e}")
            return []
    
    async def _add_citation_edge(self, source_id: str, target_id: str, 
                               citation_data: Dict, network: CitationNetwork):
        """Add a citation edge to the network"""
        try:
            relationship_type = CitationRelationshipType(
                citation_data.get("relationship_type", "cites")
            )
            
            edge = CitationEdge(
                source_case_id=source_id,
                target_case_id=target_id,
                relationship_type=relationship_type,
                strength_score=citation_data.get("strength_score", 0.7),
                legal_principle=citation_data.get("legal_principle", ""),
                extracted_reasoning=citation_data.get("context_snippet", ""),
                ai_confidence=citation_data.get("confidence", 0.8)
            )
            
            network.edges.append(edge)
            network.total_edges += 1
            
            # Update node citation counts
            if source_id in network.nodes:
                network.nodes[source_id].outgoing_citations += 1
            if target_id in network.nodes:
                network.nodes[target_id].incoming_citations += 1
            
        except Exception as e:
            logger.error(f"❌ Error adding citation edge: {e}")
    
    async def _analyze_network_structure(self, network: CitationNetwork):
        """Analyze the structure of the citation network"""
        try:
            logger.info("📊 Analyzing network structure...")
            
            # Build NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes
            for node_id, node in network.nodes.items():
                G.add_node(node_id, **asdict(node))
            
            # Add edges
            for edge in network.edges:
                G.add_edge(edge.source_case_id, edge.target_case_id, 
                          weight=edge.strength_score, **asdict(edge))
            
            self.citation_graph = G
            
            # Calculate network metrics
            if G.number_of_nodes() > 1:
                network.network_density = nx.density(G)
                
                if nx.is_weakly_connected(G):
                    network.average_path_length = nx.average_shortest_path_length(G.to_undirected())
                
                network.clustering_coefficient = nx.average_clustering(G.to_undirected())
            
            # Calculate centrality measures for nodes
            if G.number_of_nodes() > 1:
                centrality = nx.degree_centrality(G)
                betweenness = nx.betweenness_centrality(G)
                
                for node_id in network.nodes:
                    if node_id in centrality:
                        network.nodes[node_id].centrality_score = centrality[node_id]
                        network.nodes[node_id].betweenness_centrality = betweenness.get(node_id, 0.0)
            
            logger.info(f"✅ Network structure analyzed - Density: {network.network_density:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing network structure: {e}")
    
    async def _calculate_authority_scores(self, network: CitationNetwork):
        """Calculate authority scores using PageRank algorithm"""
        try:
            logger.info("📈 Calculating authority scores...")
            
            if self.citation_graph.number_of_nodes() == 0:
                return
            
            # Calculate PageRank scores
            pagerank_scores = nx.pagerank(self.citation_graph, weight='weight')
            
            # Calculate authority scores combining multiple factors
            for node_id, node in network.nodes.items():
                if node_id in pagerank_scores:
                    node.pagerank_score = pagerank_scores[node_id]
                
                # Authority score combines PageRank, citation count, and court level
                court_weight = self._get_court_authority_weight(node.authority_level)
                citation_weight = min(1.0, node.incoming_citations / 100.0)  # Normalize citation count
                temporal_weight = self._calculate_temporal_relevance(node.decision_date)
                
                node.authority_score = (
                    node.pagerank_score * 0.4 +
                    citation_weight * 0.3 +
                    court_weight * 0.2 +
                    temporal_weight * 0.1
                )
                
                # Influence score (how much this case influences others)
                node.influence_score = node.pagerank_score * (node.outgoing_citations / max(1, network.total_nodes))
            
            logger.info("✅ Authority scores calculated")
            
        except Exception as e:
            logger.error(f"❌ Error calculating authority scores: {e}")
    
    def _get_court_authority_weight(self, authority_level: AuthorityLevel) -> float:
        """Get authority weight based on court level"""
        weights = {
            AuthorityLevel.SUPREME_COURT: 1.0,
            AuthorityLevel.APPELLATE_COURT: 0.8,
            AuthorityLevel.TRIAL_COURT: 0.6,
            AuthorityLevel.ADMINISTRATIVE: 0.4
        }
        return weights.get(authority_level, 0.5)
    
    def _calculate_temporal_relevance(self, decision_date: Optional[datetime]) -> float:
        """Calculate temporal relevance score"""
        if not decision_date:
            return 0.5
        
        years_ago = (datetime.utcnow() - decision_date).days / 365.0
        
        # More recent cases have higher temporal relevance
        if years_ago <= 5:
            return 1.0
        elif years_ago <= 10:
            return 0.8
        elif years_ago <= 20:
            return 0.6
        else:
            return 0.4
    
    async def _identify_landmark_cases(self, network: CitationNetwork):
        """Identify landmark cases in the network"""
        try:
            logger.info("🏛️ Identifying landmark cases...")
            
            # Sort nodes by authority score
            sorted_nodes = sorted(
                network.nodes.values(),
                key=lambda x: x.authority_score,
                reverse=True
            )
            
            # Identify landmark cases based on multiple criteria
            landmark_threshold = 0.8
            influential_threshold = 0.6
            
            for node in sorted_nodes:
                if (node.authority_score >= landmark_threshold and 
                    node.incoming_citations >= 10):
                    node.landmark_status = "landmark"
                    network.landmark_cases.append(node.case_id)
                elif (node.authority_score >= influential_threshold and 
                      node.incoming_citations >= 5):
                    node.landmark_status = "influential"
                elif node.incoming_citations <= 1:
                    node.landmark_status = "limited"
            
            logger.info(f"✅ Identified {len(network.landmark_cases)} landmark cases")
            
        except Exception as e:
            logger.error(f"❌ Error identifying landmark cases: {e}")
    
    async def _trace_legal_evolution(self, network: CitationNetwork):
        """Trace legal evolution through citation chains"""
        try:
            logger.info("📈 Tracing legal evolution...")
            
            # Find citation chains that show legal evolution
            evolution_chains = []
            
            # Look for chains of cases that build upon each other
            for node_id, node in network.nodes.items():
                if node.landmark_status in ["landmark", "influential"]:
                    # Trace forward and backward from this case
                    chain = await self._build_evolution_chain(node_id, network)
                    if len(chain) >= 3:  # Minimum chain length
                        evolution_chains.append(chain)
            
            network.legal_evolution_chains = evolution_chains
            
            logger.info(f"✅ Found {len(evolution_chains)} legal evolution chains")
            
        except Exception as e:
            logger.error(f"❌ Error tracing legal evolution: {e}")
    
    async def _build_evolution_chain(self, start_node_id: str, network: CitationNetwork) -> List[str]:
        """Build an evolution chain starting from a landmark case"""
        try:
            chain = [start_node_id]
            
            # Find cases that cite this one (forward evolution)
            citing_cases = [
                edge.source_case_id for edge in network.edges
                if edge.target_case_id == start_node_id and 
                edge.relationship_type in [CitationRelationshipType.FOLLOWS, CitationRelationshipType.SUPPORTS]
            ]
            
            # Sort by authority score and add to chain
            citing_cases_with_scores = [
                (case_id, network.nodes[case_id].authority_score)
                for case_id in citing_cases if case_id in network.nodes
            ]
            citing_cases_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Add top citing cases to chain
            for case_id, _ in citing_cases_with_scores[:3]:
                chain.append(case_id)
            
            return chain
            
        except Exception as e:
            logger.error(f"❌ Error building evolution chain: {e}")
            return [start_node_id]
    
    async def _detect_overruling_relationships(self, network: CitationNetwork):
        """Detect overruling relationships using AI analysis"""
        try:
            logger.info("⚖️ Detecting overruling relationships...")
            
            overruling_relationships = []
            
            # Look for overruling citations
            overruling_edges = [
                edge for edge in network.edges
                if edge.relationship_type == CitationRelationshipType.OVERRULES
            ]
            
            for edge in overruling_edges:
                # Verify overruling relationship with AI
                verification = await self._verify_overruling_relationship(edge, network)
                
                if verification.get("confirmed", False):
                    overruling_relationships.append({
                        "overruling_case": edge.source_case_id,
                        "overruled_case": edge.target_case_id,
                        "legal_principle": edge.legal_principle,
                        "confidence": verification.get("confidence", 0.0),
                        "reasoning": verification.get("reasoning", "")
                    })
            
            network.overruling_relationships = overruling_relationships
            
            logger.info(f"✅ Detected {len(overruling_relationships)} overruling relationships")
            
        except Exception as e:
            logger.error(f"❌ Error detecting overruling relationships: {e}")
    
    async def _verify_overruling_relationship(self, edge: CitationEdge, network: CitationNetwork) -> Dict:
        """Verify overruling relationship using AI analysis"""
        try:
            source_case = network.nodes.get(edge.source_case_id)
            target_case = network.nodes.get(edge.target_case_id)
            
            if not source_case or not target_case:
                return {"confirmed": False, "confidence": 0.0}
            
            prompt = f"""
            Analyze whether this citation represents a true overruling relationship:
            
            Later Case: {source_case.case_title} ({source_case.citation})
            Earlier Case: {target_case.case_title} ({target_case.citation})
            
            Context: {edge.extracted_reasoning}
            Legal Principle: {edge.legal_principle}
            
            Determine if the later case actually overrules the earlier case by:
            1. Explicitly stating it overrules the earlier case
            2. Rejecting the legal principle from the earlier case
            3. Creating a new precedent that conflicts with the earlier case
            
            Respond with JSON:
            {{
                "confirmed": true/false,
                "confidence": 0.0-1.0,
                "reasoning": "explanation of the relationship",
                "overruling_type": "explicit|implicit|partial|none"
            }}
            """
            
            response = await self._generate_content_with_ai(prompt, max_tokens=500)
            
            if response:
                try:
                    import re
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            return {"confirmed": False, "confidence": 0.0}
            
        except Exception as e:
            logger.error(f"❌ Error verifying overruling relationship: {e}")
            return {"confirmed": False, "confidence": 0.0}
    
    async def _store_citation_network(self, network: CitationNetwork):
        """Store citation network in MongoDB"""
        try:
            if not self.db:
                return
            
            # Store nodes in citation_network collection
            for node in network.nodes.values():
                node_data = asdict(node)
                node_data["_id"] = node.case_id
                
                await self.db.citation_network.replace_one(
                    {"case_id": node.case_id},
                    node_data,
                    upsert=True
                )
            
            # Store edges in precedent_relationships collection
            for edge in network.edges:
                edge_data = asdict(edge)
                edge_data["id"] = str(uuid.uuid4())
                
                await self.db.precedent_relationships.replace_one(
                    {
                        "source_case_id": edge.source_case_id,
                        "target_case_id": edge.target_case_id,
                        "relationship_type": edge.relationship_type.value
                    },
                    edge_data,
                    upsert=True
                )
            
            logger.info("✅ Citation network stored in database")
            
        except Exception as e:
            logger.error(f"❌ Error storing citation network: {e}")
    
    def _serialize_network(self, network: CitationNetwork) -> Dict[str, Any]:
        """Convert network to serializable format"""
        try:
            return {
                "network_id": network.network_id,
                "total_nodes": network.total_nodes,
                "total_edges": network.total_edges,
                "network_density": network.network_density,
                "average_path_length": network.average_path_length,
                "clustering_coefficient": network.clustering_coefficient,
                "landmark_cases": network.landmark_cases,
                "authority_ranking": [
                    {
                        "case_id": node.case_id,
                        "case_title": node.case_title,
                        "authority_score": node.authority_score,
                        "pagerank_score": node.pagerank_score,
                        "citation_count": node.incoming_citations,
                        "landmark_status": node.landmark_status
                    }
                    for node in sorted(network.nodes.values(), 
                                     key=lambda x: x.authority_score, reverse=True)[:20]
                ],
                "legal_evolution_chains": network.legal_evolution_chains,
                "overruling_relationships": network.overruling_relationships,
                "jurisdiction_scope": network.jurisdiction_scope,
                "analysis_timestamp": network.analysis_timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error serializing network: {e}")
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
    
    def _determine_authority_level(self, court_name: str) -> AuthorityLevel:
        """Determine authority level from court name"""
        court_lower = court_name.lower()
        
        if "supreme" in court_lower:
            return AuthorityLevel.SUPREME_COURT
        elif any(term in court_lower for term in ["appellate", "appeal", "circuit"]):
            return AuthorityLevel.APPELLATE_COURT
        elif any(term in court_lower for term in ["district", "trial", "superior"]):
            return AuthorityLevel.TRIAL_COURT
        else:
            return AuthorityLevel.ADMINISTRATIVE
    
    async def _fetch_case_data(self, case_id: str) -> Optional[Dict]:
        """Fetch case data from available sources"""
        try:
            # Check local database first
            if self.db:
                case_data = await self.db.legal_cases.find_one({"id": case_id})
                if case_data:
                    return case_data
            
            # Try CourtListener
            if self.courtlistener_client:
                response = await self.courtlistener_client.get(f"/search/?id={case_id}")
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    if results:
                        return self._convert_courtlistener_case(results[0])
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error fetching case data for {case_id}: {e}")
            return None
    
    def _convert_courtlistener_case(self, courtlistener_case: Dict) -> Dict:
        """Convert CourtListener case format to our format"""
        return {
            "id": courtlistener_case.get("id"),
            "case_title": courtlistener_case.get("caseName", ""),
            "citation": courtlistener_case.get("citation", ""),
            "court": courtlistener_case.get("court", ""),
            "jurisdiction": self._extract_jurisdiction(courtlistener_case.get("court", "")),
            "date_filed": courtlistener_case.get("dateFiled", ""),
            "content": courtlistener_case.get("snippet", ""),
            "source": "CourtListener"
        }
    
    def _convert_stored_citation(self, stored_citation: Dict) -> Dict:
        """Convert stored citation to expected format"""
        return {
            "cited_case_id": stored_citation.get("target_case_id"),
            "relationship_type": stored_citation.get("relationship_type"),
            "legal_principle": stored_citation.get("legal_principle", ""),
            "context_snippet": stored_citation.get("extracted_reasoning", ""),
            "strength_score": stored_citation.get("strength_score", 0.7),
            "confidence": stored_citation.get("ai_confidence", 0.8)
        }
    
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
    
    def _update_performance_metrics(self, processing_time: float, network: CitationNetwork):
        """Update performance metrics"""
        try:
            self.performance_metrics["networks_analyzed"] += 1
            
            # Update average analysis time
            current_avg = self.performance_metrics["average_analysis_time"]
            total_analyses = self.performance_metrics["networks_analyzed"]
            
            new_avg = ((current_avg * (total_analyses - 1)) + processing_time) / total_analyses
            self.performance_metrics["average_analysis_time"] = new_avg
            
            # Update other metrics
            self.performance_metrics["authority_calculations"] += network.total_nodes
            self.performance_metrics["overruling_detections"] += len(network.overruling_relationships)
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            stats = {
                "system_status": "operational",
                "performance_metrics": self.performance_metrics,
                "cache_stats": {
                    "authority_cache_size": len(self.authority_cache),
                    "network_cache_size": len(self.network_cache)
                },
                "database_connected": self.db is not None,
                "ai_clients": {
                    "gemini_available": self.gemini_api_key is not None,
                    "groq_available": self.groq_client is not None,
                    "courtlistener_available": self.courtlistener_client is not None
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting system stats: {e}")
            return {"error": str(e)}


# Global instance
_citation_analyzer = None

async def get_citation_analyzer() -> CitationNetworkAnalyzer:
    """Get initialized citation network analyzer"""
    global _citation_analyzer
    
    if _citation_analyzer is None:
        _citation_analyzer = CitationNetworkAnalyzer()
        await _citation_analyzer.initialize()
    
    return _citation_analyzer


if __name__ == "__main__":
    # Test the citation network analyzer
    async def test_analyzer():
        analyzer = await get_citation_analyzer()
        
        # Test cases
        test_cases = [
            {
                "case_id": "test_case_1",
                "case_title": "Test v. Example",
                "citation": "123 F.3d 456 (9th Cir. 2020)",
                "court": "United States Court of Appeals for the Ninth Circuit",
                "jurisdiction": "US_Federal",
                "content": "This case establishes important precedent regarding contract interpretation..."
            }
        ]
        
        network = await analyzer.build_citation_network(test_cases, depth=2)
        print(f"Network built with {network['total_nodes']} nodes and {network['total_edges']} edges")
    
    asyncio.run(test_analyzer())