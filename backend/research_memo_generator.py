"""
Research Memo Generator - Automated Legal Research Memo Generation

This module implements comprehensive legal research memo generation capabilities using AI:
- Automated legal research memo generation using AI
- IRAC methodology implementation for legal analysis structure
- Citation integration and validation within memos
- Multiple memo formats (brief, comprehensive, summary)
- AI quality assessment and confidence scoring for generated memos

Key Features:
- IRAC (Issue, Rule, Application, Conclusion) structure implementation
- Multiple memo formats and templates
- Citation integration and validation
- AI-powered quality assessment
- Professional legal writing style
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

import google.generativeai as genai
from groq import Groq
import os
from motor.motor_asyncio import AsyncIOMotorClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoType(Enum):
    """Types of legal research memos"""
    BRIEF = "brief"
    COMPREHENSIVE = "comprehensive"
    SUMMARY = "summary"
    ARGUMENT_OUTLINE = "argument_outline"


class MemoFormat(Enum):
    """Legal memo formats"""
    IRAC = "irac"
    CREAC = "creac"
    TRADITIONAL = "traditional"
    EXECUTIVE_BRIEF = "executive_brief"


class QualityLevel(Enum):
    """Quality levels for memo assessment"""
    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    NEEDS_IMPROVEMENT = "needs_improvement"


@dataclass
class MemoSection:
    """Individual section of a legal memo"""
    section_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    section_type: str = ""  # "issue", "rule", "application", "conclusion", etc.
    title: str = ""
    content: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    word_count: int = 0


@dataclass
class GeneratedMemo:
    """Complete generated legal research memo"""
    memo_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    research_query: str = ""
    memo_type: MemoType = MemoType.COMPREHENSIVE
    memo_format: MemoFormat = MemoFormat.IRAC
    
    # Memo content
    title: str = ""
    executive_summary: str = ""
    memo_sections: List[MemoSection] = field(default_factory=list)
    conclusion: str = ""
    recommendations: str = ""
    
    # Supporting materials
    supporting_cases: List[Dict[str, Any]] = field(default_factory=list)
    legal_authorities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality metrics
    confidence_rating: float = 0.0
    ai_quality_score: float = 0.0
    completeness_score: float = 0.0
    quality_level: QualityLevel = QualityLevel.SATISFACTORY
    
    # Validation status
    auto_validation_status: str = "validated"  # validated, needs_review, high_confidence, low_confidence
    validation_notes: List[str] = field(default_factory=list)
    
    # Metadata
    word_count: int = 0
    reading_time_estimate: int = 0  # in minutes
    export_formats: List[str] = field(default_factory=lambda: ["pdf", "docx", "html"])
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)


class ResearchMemoGenerator:
    """
    Advanced research memo generator for automated legal memo creation.
    
    This class provides comprehensive legal memo generation using AI with
    proper legal structure and citation integration.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Research Memo Generator"""
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
        
        # Memo templates and formats
        self.memo_templates = self._initialize_memo_templates()
        self.citation_formatter = self._initialize_citation_formatter()
        
        # Quality assessment parameters
        self.quality_thresholds = {
            "excellent": 0.9,
            "good": 0.8,
            "satisfactory": 0.7,
            "needs_improvement": 0.6
        }
        
        # Performance metrics
        self.performance_metrics = {
            "memos_generated": 0,
            "average_generation_time": 0.0,
            "quality_scores": [],
            "validation_success_rate": 0.0
        }
        
        logger.info("📝 Research Memo Generator initialized")
    
    async def initialize(self):
        """Initialize all generator components"""
        try:
            logger.info("🚀 Initializing Research Memo Generator...")
            
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
            
            logger.info("🎉 Research Memo Generator fully initialized!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing research memo generator: {e}")
            raise
    
    async def _ensure_collections_exist(self):
        """Ensure required MongoDB collections exist with proper indexing"""
        try:
            # Create indexes for research_memos collection
            await self.db.research_memos.create_index("memo_id", unique=True)
            await self.db.research_memos.create_index([("confidence_rating", -1)])
            await self.db.research_memos.create_index([("created_at", -1)])
            await self.db.research_memos.create_index("research_query", name="text_search")
            
            logger.info("✅ MongoDB collections and indexes ensured")
            
        except Exception as e:
            logger.error(f"❌ Error ensuring collections: {e}")
    
    def _initialize_memo_templates(self) -> Dict[str, Dict]:
        """Initialize memo templates for different formats"""
        return {
            "irac": {
                "sections": ["issue", "rule", "application", "conclusion"],
                "description": "Issue, Rule, Application, Conclusion format",
                "structure": {
                    "issue": "Identify and frame the legal issue(s)",
                    "rule": "State the applicable legal rule(s) and precedents",
                    "application": "Apply the rule to the specific facts",
                    "conclusion": "Draw conclusions based on the analysis"
                }
            },
            "creac": {
                "sections": ["conclusion", "rule", "explanation", "application", "conclusion"],
                "description": "Conclusion, Rule, Explanation, Application, Conclusion format",
                "structure": {
                    "conclusion": "State the conclusion upfront",
                    "rule": "Present the governing legal rule",
                    "explanation": "Explain the rule through case law",
                    "application": "Apply the rule to client facts",
                    "conclusion": "Restate and expand the conclusion"
                }
            },
            "traditional": {
                "sections": ["summary", "facts", "analysis", "recommendation"],
                "description": "Traditional legal memo format",
                "structure": {
                    "summary": "Executive summary of findings",
                    "facts": "Relevant factual background",
                    "analysis": "Legal analysis and discussion",
                    "recommendation": "Recommendations and next steps"
                }
            },
            "executive_brief": {
                "sections": ["executive_summary", "key_findings", "recommendations"],
                "description": "Executive brief format for senior stakeholders",
                "structure": {
                    "executive_summary": "High-level overview and conclusions",
                    "key_findings": "Most important legal findings",
                    "recommendations": "Strategic recommendations"
                }
            }
        }
    
    def _initialize_citation_formatter(self) -> Dict[str, str]:
        """Initialize citation formatting patterns"""
        return {
            "case": "{case_name}, {citation} ({court} {year})",
            "statute": "{title} § {section} ({year})",
            "regulation": "{title} C.F.R. § {section} ({year})",
            "article": "{author}, {title}, {journal} {volume}, {page} ({year})"
        }
    
    async def generate_research_memo(self, memo_data: Dict[str, Any], 
                                   memo_type: str = "comprehensive",
                                   format_style: str = "irac") -> Dict[str, Any]:
        """
        Generate a comprehensive research memo from research data.
        
        Args:
            memo_data: Dictionary containing research query, precedents, citations, etc.
            memo_type: Type of memo to generate (brief, comprehensive, summary, argument_outline)
            format_style: Format style (irac, creac, traditional, executive_brief)
            
        Returns:
            Dict containing the generated memo
        """
        start_time = time.time()
        
        try:
            logger.info(f"📝 Generating {memo_type} memo in {format_style} format...")
            
            # Parse parameters
            memo_type_enum = MemoType(memo_type)
            format_enum = MemoFormat(format_style)
            
            # Create memo structure
            memo = GeneratedMemo(
                research_query=memo_data.get("query", ""),
                memo_type=memo_type_enum,
                memo_format=format_enum
            )
            
            # Extract research data
            precedents = memo_data.get("precedents", [])
            citations = memo_data.get("citations", {})
            jurisdiction = memo_data.get("jurisdiction", "US")
            legal_issues = memo_data.get("legal_issues", [])
            
            # Generate memo title
            memo.title = await self._generate_memo_title(memo_data)
            
            # Generate executive summary
            memo.executive_summary = await self._generate_executive_summary(memo_data, precedents)
            
            # Generate memo sections based on format
            memo.memo_sections = await self._generate_memo_sections(
                memo_data, precedents, format_enum
            )
            
            # Generate conclusion and recommendations
            memo.conclusion = await self._generate_conclusion(memo_data, precedents)
            memo.recommendations = await self._generate_recommendations(memo_data, precedents)
            
            # Process supporting materials
            memo.supporting_cases = await self._process_supporting_cases(precedents)
            memo.legal_authorities = await self._identify_legal_authorities(memo_data, precedents)
            
            # Calculate metrics
            memo.word_count = self._calculate_word_count(memo)
            memo.reading_time_estimate = max(1, memo.word_count // 200)  # Assume 200 words/minute
            
            # Perform quality assessment
            await self._assess_memo_quality(memo, memo_data)
            
            # Validate memo
            await self._validate_memo(memo)
            
            # Store memo in database
            await self._store_memo(memo)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, memo)
            
            logger.info(f"✅ Research memo generated ({memo.word_count} words) in {processing_time:.2f}s")
            
            return self._serialize_memo(memo)
            
        except Exception as e:
            logger.error(f"❌ Error generating research memo: {e}")
            raise
    
    async def _generate_memo_title(self, memo_data: Dict[str, Any]) -> str:
        """Generate an appropriate title for the memo"""
        try:
            query = memo_data.get("query", "")
            legal_issues = memo_data.get("legal_issues", [])
            jurisdiction = memo_data.get("jurisdiction", "US")
            
            # Use AI to generate a professional title
            prompt = f"""
            Generate a professional legal research memo title for this research:
            
            Research Query: {query}
            Legal Issues: {', '.join(legal_issues) if legal_issues else 'General legal analysis'}
            Jurisdiction: {jurisdiction}
            
            Create a concise, professional title that clearly indicates:
            1. The main legal issue or topic
            2. The type of analysis (if applicable)
            3. Professional tone suitable for legal memorandum
            
            Examples of good titles:
            - "Legal Analysis of Contract Breach Claims Under California Law"
            - "Memorandum Regarding Employment Discrimination Liability"
            - "Research Memo: Intellectual Property Infringement Risk Assessment"
            
            Provide only the title, no additional text.
            """
            
            title = await self._generate_content_with_ai(prompt, max_tokens=100)
            
            # Clean up the title
            title = title.strip().strip('"').strip("'")
            
            # Fallback title if AI generation fails
            if not title or len(title) < 10:
                main_topic = legal_issues[0] if legal_issues else "Legal Analysis"
                title = f"Legal Research Memorandum: {main_topic}"
            
            return title
            
        except Exception as e:
            logger.error(f"❌ Error generating memo title: {e}")
            return "Legal Research Memorandum"
    
    async def _generate_executive_summary(self, memo_data: Dict[str, Any], 
                                        precedents: List[Dict]) -> str:
        """Generate executive summary of the research findings"""
        try:
            query = memo_data.get("query", "")
            jurisdiction = memo_data.get("jurisdiction", "US")
            legal_issues = memo_data.get("legal_issues", [])
            
            # Format precedents for summary
            precedent_summary = self._format_precedents_for_prompt(precedents[:3])
            
            prompt = f"""
            Write a concise executive summary for a legal research memo based on this research:
            
            Research Question: {query}
            Jurisdiction: {jurisdiction}
            Key Legal Issues: {', '.join(legal_issues)}
            
            Key Precedents Found:
            {precedent_summary}
            
            The executive summary should:
            1. Clearly state the research question
            2. Provide a high-level answer or conclusion
            3. Mention key legal authorities
            4. Highlight any significant risks or opportunities
            5. Be written for legal professionals
            6. Be 3-4 sentences maximum
            
            Write in a professional, authoritative tone suitable for a legal memorandum.
            """
            
            summary = await self._generate_content_with_ai(prompt, max_tokens=300)
            
            return summary.strip() if summary else "Executive summary pending detailed analysis."
            
        except Exception as e:
            logger.error(f"❌ Error generating executive summary: {e}")
            return "Executive summary could not be generated."
    
    async def _generate_memo_sections(self, memo_data: Dict[str, Any], 
                                    precedents: List[Dict], format_enum: MemoFormat) -> List[MemoSection]:
        """Generate memo sections based on the specified format"""
        try:
            template = self.memo_templates.get(format_enum.value, self.memo_templates["irac"])
            sections = []
            
            for section_type in template["sections"]:
                section_content = await self._generate_section_content(
                    section_type, memo_data, precedents, template
                )
                
                section = MemoSection(
                    section_type=section_type,
                    title=section_type.replace("_", " ").title(),
                    content=section_content,
                    citations=self._extract_citations_from_content(section_content),
                    word_count=len(section_content.split())
                )
                
                # Calculate confidence score for section
                section.confidence_score = await self._calculate_section_confidence(
                    section, memo_data, precedents
                )
                
                sections.append(section)
            
            return sections
            
        except Exception as e:
            logger.error(f"❌ Error generating memo sections: {e}")
            return []
    
    async def _generate_section_content(self, section_type: str, memo_data: Dict[str, Any],
                                      precedents: List[Dict], template: Dict) -> str:
        """Generate content for a specific memo section"""
        try:
            query = memo_data.get("query", "")
            jurisdiction = memo_data.get("jurisdiction", "US")
            legal_issues = memo_data.get("legal_issues", [])
            
            section_description = template["structure"].get(section_type, "")
            precedent_text = self._format_precedents_for_prompt(precedents[:5])
            
            # Section-specific prompts
            section_prompts = {
                "issue": f"""
                Frame the legal issue(s) for this research memo:
                
                Research Query: {query}
                Legal Issues: {', '.join(legal_issues)}
                Jurisdiction: {jurisdiction}
                
                Instructions:
                1. Identify the specific legal question(s) to be answered
                2. Frame issues clearly and concisely
                3. Use precise legal terminology
                4. Present issues in order of importance
                
                Write 2-3 well-structured paragraphs that clearly identify and frame the legal issues.
                """,
                
                "rule": f"""
                State the applicable legal rules and precedents:
                
                Legal Issues: {', '.join(legal_issues)}
                Jurisdiction: {jurisdiction}
                
                Available Precedents:
                {precedent_text}
                
                Instructions:
                1. State the governing legal rules clearly
                2. Cite relevant precedential cases
                3. Explain the legal standards that apply
                4. Address any exceptions or variations
                5. Use proper legal citation format
                
                Write 3-4 paragraphs explaining the applicable legal rules with proper citations.
                """,
                
                "application": f"""
                Apply the legal rules to the specific facts:
                
                Research Query: {query}
                Legal Rules Context: {', '.join(legal_issues)}
                
                Supporting Precedents:
                {precedent_text}
                
                Instructions:
                1. Apply the legal rules to the specific factual scenario
                2. Draw analogies and distinctions with precedent cases
                3. Analyze both sides of the argument
                4. Discuss potential outcomes
                5. Address counterarguments
                
                Write 4-5 paragraphs applying the legal rules to the facts with detailed analysis.
                """,
                
                "conclusion": f"""
                Draw conclusions based on the legal analysis:
                
                Research Query: {query}
                Key Findings Context: Based on analysis of applicable legal rules and precedents
                
                Instructions:
                1. State clear conclusions based on the analysis
                2. Address the probability of different outcomes
                3. Highlight key factors that support the conclusion
                4. Acknowledge any uncertainties or limitations
                5. Be definitive where appropriate, cautious where necessary
                
                Write 2-3 paragraphs with clear, well-reasoned conclusions.
                """
            }
            
            # Default prompt for other section types
            default_prompt = f"""
            Generate content for the {section_type} section of a legal memorandum:
            
            Research Query: {query}
            Section Purpose: {section_description}
            Jurisdiction: {jurisdiction}
            
            Supporting Materials:
            {precedent_text}
            
            Write professional legal content appropriate for this section type.
            Be thorough, accurate, and use proper legal writing style.
            """
            
            prompt = section_prompts.get(section_type, default_prompt)
            content = await self._generate_content_with_ai(prompt, max_tokens=800)
            
            return content.strip() if content else f"Content for {section_type} section pending."
            
        except Exception as e:
            logger.error(f"❌ Error generating section content for {section_type}: {e}")
            return f"Error generating {section_type} section content."
    
    async def _generate_conclusion(self, memo_data: Dict[str, Any], precedents: List[Dict]) -> str:
        """Generate overall memo conclusion"""
        try:
            query = memo_data.get("query", "")
            legal_issues = memo_data.get("legal_issues", [])
            
            prompt = f"""
            Write a comprehensive conclusion for this legal research memorandum:
            
            Research Question: {query}
            Legal Issues Analyzed: {', '.join(legal_issues)}
            
            Based on the research conducted, including analysis of {len(precedents)} relevant precedents:
            
            The conclusion should:
            1. Directly answer the research question
            2. Summarize key legal findings
            3. State the strength of the legal position
            4. Identify any remaining uncertainties
            5. Be authoritative but appropriately cautious
            
            Write 2-3 well-structured paragraphs that provide a clear, professional conclusion.
            """
            
            conclusion = await self._generate_content_with_ai(prompt, max_tokens=400)
            
            return conclusion.strip() if conclusion else "Conclusion pending final analysis."
            
        except Exception as e:
            logger.error(f"❌ Error generating conclusion: {e}")
            return "Conclusion could not be generated."
    
    async def _generate_recommendations(self, memo_data: Dict[str, Any], precedents: List[Dict]) -> str:
        """Generate strategic recommendations"""
        try:
            query = memo_data.get("query", "")
            legal_issues = memo_data.get("legal_issues", [])
            
            prompt = f"""
            Provide strategic recommendations based on this legal research:
            
            Research Context: {query}
            Legal Issues: {', '.join(legal_issues)}
            Precedents Analyzed: {len(precedents)} cases
            
            Generate practical recommendations that:
            1. Address immediate action items
            2. Suggest risk mitigation strategies
            3. Identify opportunities to strengthen position
            4. Recommend additional research if needed
            5. Consider business/practical implications
            
            Format as a numbered list of 3-5 specific, actionable recommendations.
            Use professional legal advisory language.
            """
            
            recommendations = await self._generate_content_with_ai(prompt, max_tokens=500)
            
            return recommendations.strip() if recommendations else "Recommendations pending analysis completion."
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            return "Recommendations could not be generated."
    
    async def _process_supporting_cases(self, precedents: List[Dict]) -> List[Dict[str, Any]]:
        """Process and format supporting cases for the memo"""
        try:
            supporting_cases = []
            
            for precedent in precedents[:10]:  # Limit to top 10
                case_info = {
                    "case_id": precedent.get("case_id", ""),
                    "case_title": precedent.get("case_title", "Unknown Case"),
                    "citation": precedent.get("citation", ""),
                    "relevance_score": precedent.get("relevance_score", 0.0),
                    "key_holding": precedent.get("holdings", [""])[0] if precedent.get("holdings") else "",
                    "citation_format": self._format_citation(precedent)
                }
                
                supporting_cases.append(case_info)
            
            return supporting_cases
            
        except Exception as e:
            logger.error(f"❌ Error processing supporting cases: {e}")
            return []
    
    async def _identify_legal_authorities(self, memo_data: Dict[str, Any], 
                                        precedents: List[Dict]) -> List[Dict[str, Any]]:
        """Identify and format legal authorities (statutes, regulations, etc.)"""
        try:
            legal_authorities = []
            
            # Extract authorities from memo data
            query = memo_data.get("query", "")
            legal_issues = memo_data.get("legal_issues", [])
            
            # Use AI to identify relevant authorities
            prompt = f"""
            Identify relevant legal authorities for this research:
            
            Research Query: {query}
            Legal Issues: {', '.join(legal_issues)}
            
            Identify statutory, regulatory, or constitutional authorities that would be relevant.
            For each authority, provide:
            - Type (statute, regulation, constitutional provision, rule)
            - Citation
            - Relevance to the research
            
            Format as JSON array:
            [
                {
                    "authority_type": "statute",
                    "citation": "15 U.S.C. § 1051",
                    "title": "Trademark Act",
                    "relevance": "Governs trademark registration requirements",
                    "content_snippet": "Brief description of relevant provision"
                }
            ]
            """
            
            authorities_content = await self._generate_content_with_ai(prompt, max_tokens=800)
            
            if authorities_content:
                try:
                    json_match = re.search(r'\[.*\]', authorities_content, re.DOTALL)
                    if json_match:
                        authorities = json.loads(json_match.group())
                        legal_authorities.extend(authorities)
                except json.JSONDecodeError:
                    logger.warning("⚠️ Could not parse legal authorities from AI response")
            
            return legal_authorities
            
        except Exception as e:
            logger.error(f"❌ Error identifying legal authorities: {e}")
            return []
    
    async def _assess_memo_quality(self, memo: GeneratedMemo, memo_data: Dict[str, Any]):
        """Assess the quality of the generated memo"""
        try:
            logger.info("🎯 Assessing memo quality...")
            
            # Calculate AI quality score based on multiple factors
            content_completeness = self._assess_content_completeness(memo)
            citation_quality = self._assess_citation_quality(memo)
            structure_quality = self._assess_structure_quality(memo)
            legal_accuracy = await self._assess_legal_accuracy(memo, memo_data)
            
            # Weighted average
            memo.ai_quality_score = (
                content_completeness * 0.3 +
                citation_quality * 0.2 +
                structure_quality * 0.2 +
                legal_accuracy * 0.3
            )
            
            # Calculate completeness score
            memo.completeness_score = self._calculate_completeness_score(memo)
            
            # Overall confidence rating
            memo.confidence_rating = (memo.ai_quality_score + memo.completeness_score) / 2
            
            # Determine quality level
            memo.quality_level = self._determine_quality_level(memo.confidence_rating)
            
            logger.info(f"✅ Memo quality assessed - Score: {memo.confidence_rating:.2f}, Level: {memo.quality_level.value}")
            
        except Exception as e:
            logger.error(f"❌ Error assessing memo quality: {e}")
            memo.ai_quality_score = 0.7  # Default score
            memo.confidence_rating = 0.7
    
    def _assess_content_completeness(self, memo: GeneratedMemo) -> float:
        """Assess content completeness"""
        try:
            completeness_factors = []
            
            # Check if all required sections have content
            required_sections = ["issue", "rule", "application", "conclusion"]
            sections_with_content = sum(
                1 for section in memo.memo_sections 
                if section.content and len(section.content.split()) > 20
            )
            
            if memo.memo_sections:
                completeness_factors.append(sections_with_content / len(memo.memo_sections))
            
            # Check word count appropriateness
            if memo.word_count >= 800:  # Comprehensive memo
                completeness_factors.append(1.0)
            elif memo.word_count >= 500:  # Adequate length
                completeness_factors.append(0.8)
            elif memo.word_count >= 300:  # Minimum acceptable
                completeness_factors.append(0.6)
            else:  # Too short
                completeness_factors.append(0.4)
            
            # Check if conclusion and recommendations exist
            if memo.conclusion and len(memo.conclusion.split()) > 10:
                completeness_factors.append(1.0)
            else:
                completeness_factors.append(0.5)
            
            if memo.recommendations and len(memo.recommendations.split()) > 10:
                completeness_factors.append(1.0)
            else:
                completeness_factors.append(0.5)
            
            return sum(completeness_factors) / len(completeness_factors) if completeness_factors else 0.5
            
        except Exception:
            return 0.5
    
    def _assess_citation_quality(self, memo: GeneratedMemo) -> float:
        """Assess citation quality and integration"""
        try:
            total_citations = 0
            valid_citations = 0
            
            # Count citations across all sections
            for section in memo.memo_sections:
                section_citations = len(section.citations)
                total_citations += section_citations
                
                # Simple validation of citation format
                for citation in section.citations:
                    if self._validate_citation_format(citation):
                        valid_citations += 1
            
            # Add supporting cases citations
            total_citations += len(memo.supporting_cases)
            valid_citations += len(memo.supporting_cases)  # Assume these are valid
            
            if total_citations == 0:
                return 0.3  # Low score for no citations
            
            citation_ratio = valid_citations / total_citations
            
            # Bonus for having multiple types of authorities
            authority_types = set(auth.get("authority_type", "") for auth in memo.legal_authorities)
            authority_bonus = min(0.2, len(authority_types) * 0.05)
            
            return min(1.0, citation_ratio + authority_bonus)
            
        except Exception:
            return 0.5
    
    def _assess_structure_quality(self, memo: GeneratedMemo) -> float:
        """Assess structural quality of the memo"""
        try:
            structure_score = 0.0
            
            # Check if memo has proper title
            if memo.title and len(memo.title.split()) >= 3:
                structure_score += 0.2
            
            # Check if executive summary exists
            if memo.executive_summary and len(memo.executive_summary.split()) >= 20:
                structure_score += 0.2
            
            # Check section organization
            if len(memo.memo_sections) >= 3:
                structure_score += 0.3
            elif len(memo.memo_sections) >= 2:
                structure_score += 0.2
            
            # Check for conclusion
            if memo.conclusion and len(memo.conclusion.split()) >= 15:
                structure_score += 0.2
            
            # Check for recommendations
            if memo.recommendations and len(memo.recommendations.split()) >= 10:
                structure_score += 0.1
            
            return min(1.0, structure_score)
            
        except Exception:
            return 0.5
    
    async def _assess_legal_accuracy(self, memo: GeneratedMemo, memo_data: Dict[str, Any]) -> float:
        """Assess legal accuracy using AI analysis"""
        try:
            # Extract key content for accuracy assessment
            combined_content = f"""
            Title: {memo.title}
            Executive Summary: {memo.executive_summary}
            Conclusion: {memo.conclusion}
            Key Sections: {' '.join([section.content[:200] + '...' for section in memo.memo_sections[:2]])}
            """
            
            prompt = f"""
            Assess the legal accuracy and soundness of this memo content:
            
            Original Research Query: {memo_data.get("query", "")}
            Legal Issues: {', '.join(memo_data.get("legal_issues", []))}
            
            Memo Content:
            {combined_content}
            
            Evaluate the legal accuracy on a scale of 0.0 to 1.0 based on:
            1. Correct statement of legal principles
            2. Appropriate application of law to facts
            3. Sound legal reasoning
            4. Accurate use of legal terminology
            5. Logical consistency
            
            Consider potential issues like:
            - Overstated conclusions
            - Misapplication of legal rules
            - Factual or legal errors
            - Inappropriate generalizations
            
            Respond with just a decimal score between 0.0 and 1.0, followed by a brief explanation.
            Example: "0.85 - Generally accurate legal analysis with sound reasoning and appropriate conclusions."
            """
            
            response = await self._generate_content_with_ai(prompt, max_tokens=200)
            
            if response:
                # Extract score from response
                score_match = re.search(r'(\d*\.?\d+)', response)
                if score_match:
                    score = float(score_match.group(1))
                    return min(1.0, max(0.0, score))
            
            return 0.75  # Default score if analysis fails
            
        except Exception as e:
            logger.error(f"❌ Error assessing legal accuracy: {e}")
            return 0.75
    
    def _calculate_completeness_score(self, memo: GeneratedMemo) -> float:
        """Calculate overall completeness score"""
        try:
            completeness_factors = []
            
            # Required elements check
            required_elements = [
                (memo.title, 0.1),
                (memo.executive_summary, 0.2),
                (memo.memo_sections, 0.4),
                (memo.conclusion, 0.2),
                (memo.supporting_cases, 0.1)
            ]
            
            for element, weight in required_elements:
                if element:
                    if isinstance(element, list):
                        completeness_factors.append(weight if len(element) > 0 else 0)
                    else:
                        completeness_factors.append(weight if len(str(element).split()) > 5 else weight * 0.5)
                else:
                    completeness_factors.append(0)
            
            return sum(completeness_factors)
            
        except Exception:
            return 0.7
    
    def _determine_quality_level(self, confidence_rating: float) -> QualityLevel:
        """Determine quality level based on confidence rating"""
        if confidence_rating >= self.quality_thresholds["excellent"]:
            return QualityLevel.EXCELLENT
        elif confidence_rating >= self.quality_thresholds["good"]:
            return QualityLevel.GOOD
        elif confidence_rating >= self.quality_thresholds["satisfactory"]:
            return QualityLevel.SATISFACTORY
        else:
            return QualityLevel.NEEDS_IMPROVEMENT
    
    async def _validate_memo(self, memo: GeneratedMemo):
        """Validate the generated memo"""
        try:
            validation_issues = []
            
            # Check for minimum content requirements
            if memo.word_count < 300:
                validation_issues.append("Memo is too short for comprehensive analysis")
            
            if not memo.supporting_cases:
                validation_issues.append("No supporting case authorities identified")
            
            if memo.confidence_rating < 0.6:
                validation_issues.append("Low confidence in analysis quality")
            
            # Set validation status
            if not validation_issues:
                memo.auto_validation_status = "high_confidence"
            elif len(validation_issues) <= 2 and memo.confidence_rating >= 0.7:
                memo.auto_validation_status = "validated"
            elif memo.confidence_rating >= 0.6:
                memo.auto_validation_status = "needs_review"
            else:
                memo.auto_validation_status = "low_confidence"
            
            memo.validation_notes = validation_issues
            
        except Exception as e:
            logger.error(f"❌ Error validating memo: {e}")
            memo.auto_validation_status = "needs_review"
    
    def _calculate_word_count(self, memo: GeneratedMemo) -> int:
        """Calculate total word count for the memo"""
        try:
            total_words = 0
            
            # Count words in all text fields
            total_words += len(memo.title.split()) if memo.title else 0
            total_words += len(memo.executive_summary.split()) if memo.executive_summary else 0
            total_words += len(memo.conclusion.split()) if memo.conclusion else 0
            total_words += len(memo.recommendations.split()) if memo.recommendations else 0
            
            # Count words in all sections
            for section in memo.memo_sections:
                total_words += len(section.content.split()) if section.content else 0
            
            return total_words
            
        except Exception:
            return 0
    
    def _extract_citations_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Extract citations from content text"""
        try:
            citations = []
            
            # Simple regex patterns for common citation formats
            citation_patterns = [
                r'(\w+\s+v\.\s+\w+.*?\(\d{4}\))',  # Case citations
                r'(\d+\s+U\.S\.C\.\s+§\s+\d+)',    # USC citations
                r'(\d+\s+C\.F\.R\.\s+§\s+\d+)',    # CFR citations
            ]
            
            for pattern in citation_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    citations.append({
                        "citation_text": match,
                        "type": "extracted",
                        "confidence": 0.8
                    })
            
            return citations[:10]  # Limit to 10 citations per section
            
        except Exception:
            return []
    
    def _validate_citation_format(self, citation: Dict[str, Any]) -> bool:
        """Validate citation format"""
        try:
            citation_text = citation.get("citation_text", "")
            
            # Basic validation - check for common citation elements
            if len(citation_text) < 10:
                return False
            
            # Check for year in parentheses
            if re.search(r'\(\d{4}\)', citation_text):
                return True
            
            # Check for statutory citation format
            if re.search(r'\d+\s+(U\.S\.C\.|C\.F\.R\.)\s+§\s+\d+', citation_text):
                return True
            
            return False
            
        except Exception:
            return False
    
    def _format_citation(self, precedent: Dict) -> str:
        """Format a precedent as a proper legal citation"""
        try:
            case_title = precedent.get("case_title", "Unknown Case")
            citation = precedent.get("citation", "")
            court = precedent.get("court", "")
            year = ""
            
            # Extract year from date if available
            date_str = precedent.get("decision_date", precedent.get("date_filed", ""))
            if date_str:
                year_match = re.search(r'(\d{4})', date_str)
                if year_match:
                    year = year_match.group(1)
            
            # Format according to legal citation standards
            if citation and year:
                return f"{case_title}, {citation} ({year})"
            elif citation:
                return f"{case_title}, {citation}"
            else:
                return case_title
            
        except Exception:
            return precedent.get("case_title", "Unknown Case")
    
    def _format_precedents_for_prompt(self, precedents: List[Dict]) -> str:
        """Format precedents for AI prompts"""
        try:
            formatted = []
            for i, precedent in enumerate(precedents, 1):
                title = precedent.get("case_title", "Unknown Case")
                citation = precedent.get("citation", "")
                relevance = precedent.get("relevance_score", 0.0)
                summary = precedent.get("case_summary", "")[:200]
                holdings = precedent.get("holdings", [])
                key_holding = holdings[0] if holdings else "Key holding not available"
                
                formatted.append(f"""
                {i}. {title}
                   Citation: {citation}
                   Relevance: {relevance:.2f}
                   Key Holding: {key_holding}
                   Summary: {summary}...
                """)
            
            return "\n".join(formatted) if formatted else "No precedents available."
            
        except Exception:
            return "No precedents available."
    
    async def _store_memo(self, memo: GeneratedMemo):
        """Store generated memo in database"""
        try:
            if not self.db:
                return
            
            memo_data = asdict(memo)
            memo_data["_id"] = memo.memo_id
            
            await self.db.research_memos.replace_one(
                {"memo_id": memo.memo_id},
                memo_data,
                upsert=True
            )
            
            logger.info(f"✅ Memo stored in database: {memo.memo_id}")
            
        except Exception as e:
            logger.error(f"❌ Error storing memo: {e}")
    
    def _serialize_memo(self, memo: GeneratedMemo) -> Dict[str, Any]:
        """Convert memo to serializable format"""
        try:
            return {
                "memo_id": memo.memo_id,
                "title": memo.title,
                "memo_type": memo.memo_type.value,
                "memo_format": memo.memo_format.value,
                "executive_summary": memo.executive_summary,
                "memo_sections": [
                    {
                        "section_id": section.section_id,
                        "section_type": section.section_type,
                        "title": section.title,
                        "content": section.content,
                        "citations": section.citations,
                        "confidence_score": section.confidence_score,
                        "word_count": section.word_count
                    }
                    for section in memo.memo_sections
                ],
                "conclusion": memo.conclusion,
                "recommendations": memo.recommendations,
                "supporting_cases": memo.supporting_cases,
                "legal_authorities": memo.legal_authorities,
                "quality_metrics": {
                    "confidence_rating": memo.confidence_rating,
                    "ai_quality_score": memo.ai_quality_score,
                    "completeness_score": memo.completeness_score,
                    "quality_level": memo.quality_level.value,
                    "validation_status": memo.auto_validation_status,
                    "validation_notes": memo.validation_notes
                },
                "metadata": {
                    "word_count": memo.word_count,
                    "reading_time_estimate": memo.reading_time_estimate,
                    "export_formats": memo.export_formats,
                    "created_at": memo.created_at.isoformat(),
                    "last_modified": memo.last_modified.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error serializing memo: {e}")
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
    
    async def _calculate_section_confidence(self, section: MemoSection, 
                                          memo_data: Dict[str, Any], precedents: List[Dict]) -> float:
        """Calculate confidence score for a memo section"""
        try:
            confidence_factors = []
            
            # Content length factor
            if section.word_count >= 100:
                confidence_factors.append(1.0)
            elif section.word_count >= 50:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.5)
            
            # Citation factor
            if len(section.citations) >= 2:
                confidence_factors.append(1.0)
            elif len(section.citations) >= 1:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.6)
            
            # Precedent support factor
            if len(precedents) >= 3:
                confidence_factors.append(1.0)
            elif len(precedents) >= 1:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.6)
            
            return sum(confidence_factors) / len(confidence_factors)
            
        except Exception:
            return 0.7
    
    def _update_performance_metrics(self, processing_time: float, memo: GeneratedMemo):
        """Update system performance metrics"""
        try:
            self.performance_metrics["memos_generated"] += 1
            self.performance_metrics["quality_scores"].append(memo.confidence_rating)
            
            # Update average generation time
            current_avg = self.performance_metrics["average_generation_time"]
            total_memos = self.performance_metrics["memos_generated"]
            
            new_avg = ((current_avg * (total_memos - 1)) + processing_time) / total_memos
            self.performance_metrics["average_generation_time"] = new_avg
            
            # Update validation success rate
            if memo.auto_validation_status in ["validated", "high_confidence"]:
                successful_validations = sum(1 for score in self.performance_metrics["quality_scores"] if score >= 0.7)
                self.performance_metrics["validation_success_rate"] = successful_validations / total_memos
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            stats = {
                "system_status": "operational",
                "performance_metrics": self.performance_metrics,
                "templates_available": list(self.memo_templates.keys()),
                "database_connected": self.db is not None,
                "ai_clients": {
                    "gemini_available": self.gemini_api_key is not None,
                    "groq_available": self.groq_client is not None
                }
            }
            
            # Add quality statistics
            if self.performance_metrics["quality_scores"]:
                scores = self.performance_metrics["quality_scores"]
                stats["quality_statistics"] = {
                    "average_quality_score": sum(scores) / len(scores),
                    "high_quality_memos": len([s for s in scores if s >= 0.8]),
                    "total_memos_generated": len(scores)
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting system stats: {e}")
            return {"error": str(e)}


# Global instance
_memo_generator = None

async def get_memo_generator() -> ResearchMemoGenerator:
    """Get initialized research memo generator"""
    global _memo_generator
    
    if _memo_generator is None:
        _memo_generator = ResearchMemoGenerator()
        await _memo_generator.initialize()
    
    return _memo_generator


if __name__ == "__main__":
    # Test the research memo generator
    async def test_generator():
        generator = await get_memo_generator()
        
        test_memo_data = {
            "query": "What are the elements of breach of contract under California law?",
            "jurisdiction": "CA",
            "legal_issues": ["breach of contract", "damages", "remedies"],
            "precedents": [
                {
                    "case_title": "Test v. Example",
                    "citation": "123 Cal. App. 4th 456 (2020)",
                    "relevance_score": 0.85,
                    "case_summary": "Case involving contract breach and damages",
                    "holdings": ["Breach requires proof of duty, breach, causation, and damages"]
                }
            ]
        }
        
        memo = await generator.generate_research_memo(test_memo_data)
        print(f"Generated memo: {memo['title']}")
        print(f"Word count: {memo['metadata']['word_count']}")
        print(f"Quality score: {memo['quality_metrics']['confidence_rating']:.2f}")
    
    asyncio.run(test_generator())