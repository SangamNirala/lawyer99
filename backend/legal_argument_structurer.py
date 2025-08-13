"""
Legal Argument Structurer - AI-Powered Legal Argument Construction

This module implements advanced legal argument construction with precedent support,
argument strength optimization, and counterargument identification.

Key Features:
- AI-powered legal argument construction with precedent support
- Argument strength optimization and persuasiveness enhancement
- Counterargument identification and mitigation strategies
- Supporting precedent ranking and selection
- Argument template generation for different legal scenarios
"""

import asyncio
import json
import logging
import re
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArgumentType(Enum):
    """Types of legal arguments"""
    PRIMARY = "primary"
    SUPPORTING = "supporting"
    COUNTERARGUMENT = "counterargument"
    MITIGATION = "mitigation"
    ALTERNATIVE = "alternative"
    PROCEDURAL = "procedural"


class ArgumentStrength(Enum):
    """Argument strength levels"""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    SPECULATIVE = "speculative"


class LegalPosition(Enum):
    """Legal positions for argument construction"""
    PLAINTIFF = "plaintiff"
    DEFENDANT = "defendant"
    PETITIONER = "petitioner"
    RESPONDENT = "respondent"
    APPELLANT = "appellant"
    APPELLEE = "appellee"


@dataclass
class LegalArgument:
    """Individual legal argument structure"""
    argument_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    argument_type: ArgumentType = ArgumentType.PRIMARY
    position: LegalPosition = LegalPosition.PLAINTIFF
    
    # Argument content
    argument_text: str = ""
    legal_principle: str = ""
    supporting_reasoning: str = ""
    
    # Supporting evidence
    supporting_cases: List[Dict[str, Any]] = field(default_factory=list)
    statutory_support: List[Dict[str, Any]] = field(default_factory=list)
    factual_support: List[str] = field(default_factory=list)
    
    # Strength assessment
    strength_score: float = 0.0
    persuasiveness_score: float = 0.0
    authority_score: float = 0.0
    
    # Counter-analysis
    potential_counterarguments: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    distinguishing_factors: List[str] = field(default_factory=list)
    
    # Context
    jurisdiction: str = ""
    legal_context: str = ""
    target_audience: str = "court"  # court, jury, opposing_counsel
    
    # Quality metrics
    confidence_score: float = 0.0
    completeness_score: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ArgumentStructure:
    """Complete legal argument structure"""
    structure_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    case_title: str = ""
    legal_question: str = ""
    jurisdiction: str = ""
    
    # Argument hierarchy
    primary_arguments: List[LegalArgument] = field(default_factory=list)
    supporting_arguments: List[LegalArgument] = field(default_factory=list)
    counterarguments: List[LegalArgument] = field(default_factory=list)
    mitigation_arguments: List[LegalArgument] = field(default_factory=list)
    
    # Overall strategy
    argument_strategy: str = ""
    strength_assessment: Dict[str, float] = field(default_factory=dict)
    risk_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Precedent integration
    key_precedents: List[Dict[str, Any]] = field(default_factory=list)
    precedent_hierarchy: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality assessment
    overall_strength: float = 0.0
    persuasiveness_rating: float = 0.0
    completeness_rating: float = 0.0
    
    # Metadata
    created_for_position: LegalPosition = LegalPosition.PLAINTIFF
    target_court_level: str = "trial"
    estimated_success_probability: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class LegalArgumentStructurer:
    """
    Advanced legal argument construction system using AI and precedent analysis.
    
    This system constructs persuasive legal arguments with comprehensive
    precedent support and counterargument analysis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Legal Argument Structurer"""
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
        
        # Argument templates and patterns
        self.argument_templates = {}
        self.strength_indicators = {}
        self.counterargument_patterns = {}
        
        # Caching system
        self.argument_cache = {}
        self.precedent_cache = {}
        
        # Configuration
        self.max_arguments_per_type = 5
        self.min_strength_threshold = 0.6
        self.max_counterarguments = 3
        
        # Performance tracking
        self.performance_metrics = {
            "arguments_structured": 0,
            "average_structure_time": 0.0,
            "average_strength_score": 0.0,
            "success_predictions": []
        }
        
        logger.info("⚖️ Legal Argument Structurer initialized")
    
    async def initialize(self):
        """Initialize all system components"""
        try:
            logger.info("🚀 Initializing Legal Argument Structurer...")
            
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
            
            # Load argument templates
            await self._load_argument_templates()
            
            # Initialize strength indicators
            await self._initialize_strength_indicators()
            
            # Load counterargument patterns
            await self._load_counterargument_patterns()
            
            logger.info("🎉 Legal Argument Structurer fully initialized!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing argument structurer: {e}")
            raise
    
    async def _load_argument_templates(self):
        """Load argument templates for different legal scenarios"""
        try:
            logger.info("📋 Loading argument templates...")
            
            # Contract law templates
            self.argument_templates["contract_breach"] = {
                "primary_elements": [
                    "Existence of valid contract",
                    "Performance or excuse of performance",
                    "Breach by defendant",
                    "Causation and damages"
                ],
                "supporting_arguments": [
                    "Clear contract terms",
                    "Notice of breach",
                    "Mitigation efforts",
                    "Calculation of damages"
                ],
                "common_defenses": [
                    "Impossibility of performance",
                    "Frustration of purpose",
                    "Mutual mistake",
                    "Duress or undue influence"
                ]
            }
            
            # Tort law templates
            self.argument_templates["negligence"] = {
                "primary_elements": [
                    "Duty of care owed to plaintiff",
                    "Breach of duty (standard of care)",
                    "Causation (factual and legal)",
                    "Damages"
                ],
                "supporting_arguments": [
                    "Foreseeability of harm",
                    "Reasonable person standard",
                    "Expert testimony on standard",
                    "Proximate cause analysis"
                ],
                "common_defenses": [
                    "Comparative negligence",
                    "Assumption of risk",
                    "Intervening cause",
                    "Statute of limitations"
                ]
            }
            
            # Employment law templates
            self.argument_templates["wrongful_termination"] = {
                "primary_elements": [
                    "At-will employment exception",
                    "Illegal or improper termination",
                    "Causation between protected activity and termination",
                    "Damages from termination"
                ],
                "supporting_arguments": [
                    "Public policy violation",
                    "Whistleblower protection",
                    "Discriminatory animus",
                    "Breach of implied contract"
                ],
                "common_defenses": [
                    "Legitimate business reason",
                    "Progressive discipline",
                    "Performance issues",
                    "At-will employment"
                ]
            }
            
            logger.info("✅ Argument templates loaded")
            
        except Exception as e:
            logger.error(f"❌ Error loading argument templates: {e}")
    
    async def _initialize_strength_indicators(self):
        """Initialize indicators for argument strength assessment"""
        try:
            self.strength_indicators = {
                "strong_indicators": [
                    "binding precedent",
                    "unanimous court decision",
                    "recent authority",
                    "factual similarity",
                    "clear legal principle",
                    "statutory support",
                    "constitutional basis"
                ],
                "moderate_indicators": [
                    "persuasive precedent",
                    "majority decision",
                    "analogous facts",
                    "policy arguments",
                    "jurisdictional support",
                    "expert consensus"
                ],
                "weak_indicators": [
                    "minority view",
                    "distinguishable precedent",
                    "outdated authority",
                    "policy speculation",
                    "unsettled law",
                    "conflicting authorities"
                ]
            }
            
            logger.info("✅ Strength indicators initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing strength indicators: {e}")
    
    async def _load_counterargument_patterns(self):
        """Load common counterargument patterns"""
        try:
            self.counterargument_patterns = {
                "precedent_challenges": [
                    "distinguishable facts",
                    "different legal context",
                    "superseded by later decisions",
                    "limited to specific circumstances",
                    "dicta rather than holding"
                ],
                "policy_arguments": [
                    "unintended consequences",
                    "judicial restraint",
                    "legislative intent",
                    "separation of powers",
                    "federalism concerns"
                ],
                "factual_disputes": [
                    "credibility issues",
                    "missing evidence",
                    "alternative interpretation",
                    "burden of proof",
                    "expert disagreement"
                ]
            }
            
            logger.info("✅ Counterargument patterns loaded")
            
        except Exception as e:
            logger.error(f"❌ Error loading counterargument patterns: {e}")
    
    async def structure_legal_arguments(self, argument_data: Dict[str, Any], 
                                      argument_strength: str = "strong",
                                      include_counterarguments: bool = True) -> List[Dict[str, Any]]:
        """
        Structure comprehensive legal arguments with precedent support.
        
        Args:
            argument_data: Data for argument construction
            argument_strength: Target argument strength level
            include_counterarguments: Whether to include counterargument analysis
            
        Returns:
            List of structured legal arguments
        """
        start_time = time.time()
        
        try:
            logger.info("⚖️ Structuring legal arguments...")
            
            # Extract argument parameters
            legal_question = argument_data.get("legal_question", "")
            precedents = argument_data.get("precedents", [])
            jurisdiction = argument_data.get("jurisdiction", "US")
            case_type = argument_data.get("case_type", "")
            position = LegalPosition(argument_data.get("position", "plaintiff"))
            
            # Initialize argument structure
            structure = ArgumentStructure(
                legal_question=legal_question,
                jurisdiction=jurisdiction,
                created_for_position=position
            )
            
            # Identify argument template
            template_key = self._identify_argument_template(legal_question, case_type)
            
            # Generate primary arguments
            primary_args = await self._generate_primary_arguments(
                legal_question, precedents, jurisdiction, template_key, position
            )
            structure.primary_arguments = primary_args
            
            # Generate supporting arguments
            supporting_args = await self._generate_supporting_arguments(
                primary_args, precedents, legal_question
            )
            structure.supporting_arguments = supporting_args
            
            # Generate counterarguments if requested
            if include_counterarguments:
                counter_args = await self._generate_counterarguments(
                    primary_args, legal_question, precedents
                )
                structure.counterarguments = counter_args
                
                # Generate mitigation strategies
                mitigation_args = await self._generate_mitigation_strategies(
                    counter_args, primary_args
                )
                structure.mitigation_arguments = mitigation_args
            
            # Rank and organize precedents
            structure.key_precedents = await self._rank_supporting_precedents(precedents, primary_args)
            structure.precedent_hierarchy = await self._organize_precedent_hierarchy(structure.key_precedents)
            
            # Assess argument strength
            await self._assess_argument_strength(structure)
            
            # Generate argument strategy
            structure.argument_strategy = await self._generate_argument_strategy(structure)
            
            # Perform risk analysis
            structure.risk_analysis = await self._perform_risk_analysis(structure)
            
            # Calculate success probability
            structure.estimated_success_probability = await self._estimate_success_probability(structure)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, structure)
            
            # Convert to serializable format
            structured_arguments = self._serialize_argument_structure(structure)
            
            logger.info(f"✅ Legal arguments structured - Primary: {len(primary_args)}, Supporting: {len(supporting_args)}")
            
            return structured_arguments
            
        except Exception as e:
            logger.error(f"❌ Error structuring legal arguments: {e}")
            raise
    
    def _identify_argument_template(self, legal_question: str, case_type: str) -> str:
        """Identify appropriate argument template based on legal question"""
        try:
            question_lower = legal_question.lower()
            
            # Contract law patterns
            if any(term in question_lower for term in ["contract", "breach", "agreement", "performance"]):
                return "contract_breach"
            
            # Tort law patterns
            elif any(term in question_lower for term in ["negligence", "duty", "care", "tort", "liability"]):
                return "negligence"
            
            # Employment law patterns
            elif any(term in question_lower for term in ["employment", "termination", "wrongful", "discrimination"]):
                return "wrongful_termination"
            
            # Default to contract if uncertain
            else:
                return "contract_breach"
                
        except Exception:
            return "contract_breach"
    
    async def _generate_primary_arguments(self, legal_question: str, precedents: List[Dict],
                                        jurisdiction: str, template_key: str, 
                                        position: LegalPosition) -> List[LegalArgument]:
        """Generate primary legal arguments"""
        try:
            logger.info("🎯 Generating primary arguments...")
            
            primary_arguments = []
            template = self.argument_templates.get(template_key, {})
            primary_elements = template.get("primary_elements", [])
            
            # Generate arguments for each primary element
            for i, element in enumerate(primary_elements[:self.max_arguments_per_type]):
                
                prompt = f"""
                Construct a strong primary legal argument for the {position.value} position:
                
                Legal Question: {legal_question}
                Jurisdiction: {jurisdiction}
                Argument Element: {element}
                
                Supporting Precedents:
                {self._format_precedents_for_prompt(precedents[:3])}
                
                Generate a comprehensive argument that:
                1. States the legal principle clearly
                2. Applies relevant precedents
                3. Connects facts to law
                4. Addresses the specific element: {element}
                5. Uses persuasive legal reasoning
                6. Cites supporting authorities
                
                Provide the argument in this JSON format:
                {{
                    "argument_text": "The complete legal argument",
                    "legal_principle": "Core legal principle",
                    "supporting_reasoning": "Detailed reasoning",
                    "factual_support": ["key fact 1", "key fact 2"],
                    "strength_indicators": ["reason 1", "reason 2"]
                }}
                """
                
                argument_content = await self._generate_content_with_ai(prompt, max_tokens=1000)
                
                if argument_content:
                    try:
                        # Parse AI response
                        json_match = re.search(r'\{.*\}', argument_content, re.DOTALL)
                        if json_match:
                            arg_data = json.loads(json_match.group())
                            
                            argument = LegalArgument(
                                argument_type=ArgumentType.PRIMARY,
                                position=position,
                                argument_text=arg_data.get("argument_text", ""),
                                legal_principle=arg_data.get("legal_principle", ""),
                                supporting_reasoning=arg_data.get("supporting_reasoning", ""),
                                factual_support=arg_data.get("factual_support", []),
                                jurisdiction=jurisdiction,
                                strength_score=0.8,  # High strength for primary arguments
                                confidence_score=0.85
                            )
                            
                            # Add supporting cases
                            argument.supporting_cases = precedents[:2]  # Top 2 precedents
                            
                            primary_arguments.append(argument)
                            
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"⚠️ Error parsing argument content: {e}")
                        continue
            
            logger.info(f"✅ Generated {len(primary_arguments)} primary arguments")
            return primary_arguments
            
        except Exception as e:
            logger.error(f"❌ Error generating primary arguments: {e}")
            return []
    
    async def _generate_supporting_arguments(self, primary_args: List[LegalArgument],
                                           precedents: List[Dict], legal_question: str) -> List[LegalArgument]:
        """Generate supporting arguments for primary arguments"""
        try:
            logger.info("🔗 Generating supporting arguments...")
            
            supporting_arguments = []
            
            for primary_arg in primary_args[:3]:  # Support top 3 primary arguments
                
                prompt = f"""
                Generate a supporting argument that reinforces this primary argument:
                
                Primary Argument: {primary_arg.argument_text[:300]}...
                Legal Principle: {primary_arg.legal_principle}
                Legal Question: {legal_question}
                
                Available Precedents:
                {self._format_precedents_for_prompt(precedents[2:5])}  # Use different precedents
                
                Create a supporting argument that:
                1. Reinforces the primary argument's reasoning
                2. Provides additional legal authority
                3. Addresses potential weaknesses
                4. Uses different precedents or authorities
                5. Strengthens the overall position
                
                Format as JSON:
                {{
                    "argument_text": "Supporting argument text",
                    "legal_principle": "Additional legal principle",
                    "supporting_reasoning": "Why this supports the primary argument",
                    "connection_to_primary": "How this connects to the primary argument"
                }}
                """
                
                support_content = await self._generate_content_with_ai(prompt, max_tokens=800)
                
                if support_content:
                    try:
                        json_match = re.search(r'\{.*\}', support_content, re.DOTALL)
                        if json_match:
                            support_data = json.loads(json_match.group())
                            
                            supporting_arg = LegalArgument(
                                argument_type=ArgumentType.SUPPORTING,
                                position=primary_arg.position,
                                argument_text=support_data.get("argument_text", ""),
                                legal_principle=support_data.get("legal_principle", ""),
                                supporting_reasoning=support_data.get("supporting_reasoning", ""),
                                jurisdiction=primary_arg.jurisdiction,
                                strength_score=0.7,  # Moderate strength for supporting
                                confidence_score=0.75
                            )
                            
                            # Link different precedents
                            supporting_arg.supporting_cases = precedents[2:4] if len(precedents) > 2 else []
                            
                            supporting_arguments.append(supporting_arg)
                            
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"⚠️ Error parsing supporting argument: {e}")
                        continue
            
            logger.info(f"✅ Generated {len(supporting_arguments)} supporting arguments")
            return supporting_arguments
            
        except Exception as e:
            logger.error(f"❌ Error generating supporting arguments: {e}")
            return []
    
    async def _generate_counterarguments(self, primary_args: List[LegalArgument],
                                       legal_question: str, precedents: List[Dict]) -> List[LegalArgument]:
        """Generate potential counterarguments"""
        try:
            logger.info("🔄 Generating counterarguments...")
            
            counterarguments = []
            
            # Get opposing position
            opposing_position = self._get_opposing_position(primary_args[0].position if primary_args else LegalPosition.PLAINTIFF)
            
            for primary_arg in primary_args[:self.max_counterarguments]:
                
                prompt = f"""
                Generate the strongest counterargument against this legal argument:
                
                Target Argument: {primary_arg.argument_text[:400]}...
                Legal Principle: {primary_arg.legal_principle}
                Legal Question: {legal_question}
                
                Create a counterargument from the {opposing_position.value} perspective that:
                1. Directly challenges the primary argument
                2. Identifies weaknesses in reasoning or precedents
                3. Presents alternative legal interpretation
                4. Uses distinguishing factors
                5. Provides opposing authorities if available
                
                Consider these common counterargument patterns:
                - Distinguishable precedents
                - Alternative legal interpretations
                - Factual disputes
                - Policy arguments
                - Jurisdictional differences
                
                Format as JSON:
                {{
                    "argument_text": "Counterargument text",
                    "legal_principle": "Opposing legal principle",
                    "challenge_points": ["weakness 1", "weakness 2"],
                    "distinguishing_factors": ["factor 1", "factor 2"]
                }}
                """
                
                counter_content = await self._generate_content_with_ai(prompt, max_tokens=800)
                
                if counter_content:
                    try:
                        json_match = re.search(r'\{.*\}', counter_content, re.DOTALL)
                        if json_match:
                            counter_data = json.loads(json_match.group())
                            
                            counterarg = LegalArgument(
                                argument_type=ArgumentType.COUNTERARGUMENT,
                                position=opposing_position,
                                argument_text=counter_data.get("argument_text", ""),
                                legal_principle=counter_data.get("legal_principle", ""),
                                distinguishing_factors=counter_data.get("distinguishing_factors", []),
                                jurisdiction=primary_arg.jurisdiction,
                                strength_score=0.6,  # Moderate strength
                                confidence_score=0.70
                            )
                            
                            counterarguments.append(counterarg)
                            
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"⚠️ Error parsing counterargument: {e}")
                        continue
            
            logger.info(f"✅ Generated {len(counterarguments)} counterarguments")
            return counterarguments
            
        except Exception as e:
            logger.error(f"❌ Error generating counterarguments: {e}")
            return []
    
    async def _generate_mitigation_strategies(self, counter_args: List[LegalArgument],
                                            primary_args: List[LegalArgument]) -> List[LegalArgument]:
        """Generate mitigation strategies for counterarguments"""
        try:
            logger.info("🛡️ Generating mitigation strategies...")
            
            mitigation_arguments = []
            
            for i, counter_arg in enumerate(counter_args):
                related_primary = primary_args[i] if i < len(primary_args) else primary_args[0]
                
                prompt = f"""
                Generate a mitigation strategy to address this counterargument:
                
                Counterargument: {counter_arg.argument_text[:300]}...
                Challenges: {counter_arg.distinguishing_factors}
                
                Original Argument: {related_primary.argument_text[:200]}...
                
                Create a mitigation argument that:
                1. Directly addresses the counterargument's challenges
                2. Reinforces the original position
                3. Distinguishes opposing precedents
                4. Provides additional support
                5. Minimizes the counterargument's impact
                
                Format as JSON:
                {{
                    "argument_text": "Mitigation argument text",
                    "mitigation_strategies": ["strategy 1", "strategy 2"],
                    "reinforcing_points": ["point 1", "point 2"]
                }}
                """
                
                mitigation_content = await self._generate_content_with_ai(prompt, max_tokens=600)
                
                if mitigation_content:
                    try:
                        json_match = re.search(r'\{.*\}', mitigation_content, re.DOTALL)
                        if json_match:
                            mit_data = json.loads(json_match.group())
                            
                            mitigation_arg = LegalArgument(
                                argument_type=ArgumentType.MITIGATION,
                                position=related_primary.position,
                                argument_text=mit_data.get("argument_text", ""),
                                mitigation_strategies=mit_data.get("mitigation_strategies", []),
                                jurisdiction=related_primary.jurisdiction,
                                strength_score=0.65,
                                confidence_score=0.70
                            )
                            
                            mitigation_arguments.append(mitigation_arg)
                            
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"⚠️ Error parsing mitigation strategy: {e}")
                        continue
            
            logger.info(f"✅ Generated {len(mitigation_arguments)} mitigation strategies")
            return mitigation_arguments
            
        except Exception as e:
            logger.error(f"❌ Error generating mitigation strategies: {e}")
            return []
    
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
            
            else:
                logger.warning("⚠️ No AI clients available for content generation")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Error generating AI content: {e}")
            return ""
    
    def _format_precedents_for_prompt(self, precedents: List[Dict]) -> str:
        """Format precedents for AI prompts"""
        try:
            formatted = []
            for i, precedent in enumerate(precedents, 1):
                title = precedent.get("case_title", "Unknown Case")
                citation = precedent.get("citation", "")
                relevance = precedent.get("relevance_score", 0.0)
                summary = precedent.get("case_summary", "")[:150]
                
                formatted.append(f"""
                {i}. {title}
                   Citation: {citation}
                   Relevance: {relevance:.2f}
                   Summary: {summary}...
                """)
            
            return "\n".join(formatted) if formatted else "No precedents available."
            
        except Exception:
            return "No precedents available."
    
    def _get_opposing_position(self, position: LegalPosition) -> LegalPosition:
        """Get the opposing legal position"""
        position_map = {
            LegalPosition.PLAINTIFF: LegalPosition.DEFENDANT,
            LegalPosition.DEFENDANT: LegalPosition.PLAINTIFF,
            LegalPosition.PETITIONER: LegalPosition.RESPONDENT,
            LegalPosition.RESPONDENT: LegalPosition.PETITIONER,
            LegalPosition.APPELLANT: LegalPosition.APPELLEE,
            LegalPosition.APPELLEE: LegalPosition.APPELLANT
        }
        return position_map.get(position, LegalPosition.DEFENDANT)
    
    async def _rank_supporting_precedents(self, precedents: List[Dict], 
                                        primary_args: List[LegalArgument]) -> List[Dict[str, Any]]:
        """Rank precedents by their support for the arguments"""
        try:
            logger.info("📊 Ranking supporting precedents...")
            
            ranked_precedents = []
            
            for precedent in precedents[:10]:  # Top 10
                relevance_score = precedent.get("relevance_score", 0.0)
                authority_score = precedent.get("authority_score", 0.0)
                
                # Calculate support score for arguments
                support_score = 0.0
                for arg in primary_args:
                    if any(case.get("case_title") == precedent.get("case_title") for case in arg.supporting_cases):
                        support_score += 0.3
                
                # Combined ranking score
                overall_score = (relevance_score * 0.4 + authority_score * 0.3 + support_score * 0.3)
                
                ranked_precedent = {
                    "case_id": precedent.get("case_id", ""),
                    "case_title": precedent.get("case_title", ""),
                    "citation": precedent.get("citation", ""),
                    "court": precedent.get("court", ""),
                    "relevance_score": relevance_score,
                    "authority_score": authority_score,
                    "support_score": support_score,
                    "overall_ranking": overall_score,
                    "key_holding": precedent.get("holdings", [""])[0] if precedent.get("holdings") else ""
                }
                
                ranked_precedents.append(ranked_precedent)
            
            # Sort by overall ranking
            ranked_precedents.sort(key=lambda x: x["overall_ranking"], reverse=True)
            
            return ranked_precedents
            
        except Exception as e:
            logger.error(f"❌ Error ranking precedents: {e}")
            return []
    
    async def _organize_precedent_hierarchy(self, precedents: List[Dict]) -> List[Dict[str, Any]]:
        """Organize precedents into hierarchy by authority level"""
        try:
            hierarchy = {
                "supreme_court": [],
                "appellate_court": [],
                "trial_court": [],
                "other": []
            }
            
            for precedent in precedents:
                court = precedent.get("court", "").lower()
                
                if "supreme" in court:
                    hierarchy["supreme_court"].append(precedent)
                elif any(term in court for term in ["appellate", "appeal", "circuit"]):
                    hierarchy["appellate_court"].append(precedent)
                elif any(term in court for term in ["district", "trial", "superior"]):
                    hierarchy["trial_court"].append(precedent)
                else:
                    hierarchy["other"].append(precedent)
            
            return [{"court_level": level, "cases": cases} for level, cases in hierarchy.items() if cases]
            
        except Exception as e:
            logger.error(f"❌ Error organizing precedent hierarchy: {e}")
            return []
    
    async def _assess_argument_strength(self, structure: ArgumentStructure):
        """Assess overall argument strength"""
        try:
            logger.info("💪 Assessing argument strength...")
            
            # Calculate strength for each argument type
            primary_strength = 0.0
            if structure.primary_arguments:
                primary_strength = sum(arg.strength_score for arg in structure.primary_arguments) / len(structure.primary_arguments)
            
            supporting_strength = 0.0
            if structure.supporting_arguments:
                supporting_strength = sum(arg.strength_score for arg in structure.supporting_arguments) / len(structure.supporting_arguments)
            
            counter_weakness = 0.0
            if structure.counterarguments:
                counter_weakness = sum(arg.strength_score for arg in structure.counterarguments) / len(structure.counterarguments)
            
            mitigation_strength = 0.0
            if structure.mitigation_arguments:
                mitigation_strength = sum(arg.strength_score for arg in structure.mitigation_arguments) / len(structure.mitigation_arguments)
            
            # Overall strength calculation
            structure.overall_strength = (
                primary_strength * 0.5 +
                supporting_strength * 0.2 +
                mitigation_strength * 0.2 -
                counter_weakness * 0.1
            )
            
            # Persuasiveness rating (based on authority and reasoning quality)
            authority_bonus = 0.0
            if structure.key_precedents:
                avg_authority = sum(p.get("authority_score", 0.0) for p in structure.key_precedents[:3]) / min(3, len(structure.key_precedents))
                authority_bonus = avg_authority * 0.2
            
            structure.persuasiveness_rating = min(1.0, structure.overall_strength + authority_bonus)
            
            # Completeness rating
            completeness_factors = [
                1.0 if structure.primary_arguments else 0.0,
                1.0 if structure.supporting_arguments else 0.0,
                1.0 if structure.counterarguments else 0.0,
                1.0 if structure.mitigation_arguments else 0.0,
                1.0 if structure.key_precedents else 0.0
            ]
            structure.completeness_rating = sum(completeness_factors) / len(completeness_factors)
            
            # Store detailed strength assessment
            structure.strength_assessment = {
                "primary_strength": primary_strength,
                "supporting_strength": supporting_strength,
                "counter_weakness": counter_weakness,
                "mitigation_strength": mitigation_strength,
                "authority_bonus": authority_bonus,
                "overall_assessment": self._categorize_strength(structure.overall_strength)
            }
            
            logger.info(f"✅ Argument strength assessed - Overall: {structure.overall_strength:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error assessing argument strength: {e}")
    
    def _categorize_strength(self, strength_score: float) -> str:
        """Categorize argument strength"""
        if strength_score >= 0.85:
            return "Very Strong"
        elif strength_score >= 0.75:
            return "Strong"
        elif strength_score >= 0.65:
            return "Moderate"
        elif strength_score >= 0.50:
            return "Weak"
        else:
            return "Very Weak"
    
    async def _generate_argument_strategy(self, structure: ArgumentStructure) -> str:
        """Generate overall argument strategy"""
        try:
            prompt = f"""
            Based on this legal argument structure, recommend an overall litigation strategy:
            
            Case: {structure.legal_question}
            Overall Strength: {structure.overall_strength:.2f}
            Primary Arguments: {len(structure.primary_arguments)}
            Counterarguments: {len(structure.counterarguments)}
            Key Precedents: {len(structure.key_precedents)}
            
            Strength Assessment:
            - Primary Argument Strength: {structure.strength_assessment.get('primary_strength', 0.0):.2f}
            - Supporting Evidence: {len(structure.supporting_arguments)} arguments
            - Potential Vulnerabilities: {len(structure.counterarguments)} counterarguments
            - Mitigation Coverage: {len(structure.mitigation_arguments)} strategies
            
            Provide a strategic recommendation that addresses:
            1. Overall approach (aggressive, moderate, defensive)
            2. Key strengths to emphasize
            3. Vulnerabilities to address
            4. Sequencing of arguments
            5. Settlement considerations
            6. Risk mitigation strategies
            
            Keep the strategy concise and actionable.
            """
            
            strategy = await self._generate_content_with_ai(prompt, max_tokens=600)
            return strategy or "Strategy analysis pending."
            
        except Exception as e:
            logger.error(f"❌ Error generating argument strategy: {e}")
            return "Strategy analysis could not be completed."
    
    async def _perform_risk_analysis(self, structure: ArgumentStructure) -> Dict[str, Any]:
        """Perform comprehensive risk analysis"""
        try:
            risk_analysis = {
                "overall_risk_level": "moderate",
                "key_risks": [],
                "mitigation_recommendations": [],
                "success_factors": [],
                "concern_areas": []
            }
            
            # Assess risk level based on argument strength
            if structure.overall_strength >= 0.8:
                risk_analysis["overall_risk_level"] = "low"
            elif structure.overall_strength >= 0.6:
                risk_analysis["overall_risk_level"] = "moderate" 
            else:
                risk_analysis["overall_risk_level"] = "high"
            
            # Identify key risks
            if len(structure.counterarguments) > len(structure.mitigation_arguments):
                risk_analysis["key_risks"].append("Insufficient counterargument mitigation")
            
            if not structure.key_precedents:
                risk_analysis["key_risks"].append("Limited precedential support")
            
            if structure.overall_strength < 0.6:
                risk_analysis["key_risks"].append("Weak argument foundation")
            
            # Success factors
            if structure.overall_strength >= 0.75:
                risk_analysis["success_factors"].append("Strong primary arguments")
            
            if len(structure.key_precedents) >= 3:
                risk_analysis["success_factors"].append("Solid precedential support")
            
            if len(structure.mitigation_arguments) >= len(structure.counterarguments):
                risk_analysis["success_factors"].append("Good counterargument coverage")
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"❌ Error performing risk analysis: {e}")
            return {"overall_risk_level": "unknown", "key_risks": [], "mitigation_recommendations": []}
    
    async def _estimate_success_probability(self, structure: ArgumentStructure) -> float:
        """Estimate probability of success based on argument analysis"""
        try:
            # Base probability from argument strength
            base_probability = structure.overall_strength * 0.6
            
            # Precedent support bonus
            precedent_bonus = 0.0
            if structure.key_precedents:
                strong_precedents = sum(1 for p in structure.key_precedents if p.get("authority_score", 0.0) > 0.7)
                precedent_bonus = min(0.2, strong_precedents * 0.05)
            
            # Counterargument penalty
            counter_penalty = 0.0
            if structure.counterarguments:
                unmitigated_counters = max(0, len(structure.counterarguments) - len(structure.mitigation_arguments))
                counter_penalty = min(0.15, unmitigated_counters * 0.05)
            
            # Completeness bonus
            completeness_bonus = structure.completeness_rating * 0.1
            
            # Calculate final probability
            success_probability = base_probability + precedent_bonus + completeness_bonus - counter_penalty
            
            return min(0.95, max(0.05, success_probability))  # Cap between 5% and 95%
            
        except Exception as e:
            logger.error(f"❌ Error estimating success probability: {e}")
            return 0.5  # Default 50% if calculation fails
    
    def _serialize_argument_structure(self, structure: ArgumentStructure) -> List[Dict[str, Any]]:
        """Convert argument structure to serializable format"""
        try:
            serialized_args = []
            
            # Primary arguments
            for arg in structure.primary_arguments:
                serialized_args.append({
                    "argument_id": arg.argument_id,
                    "type": "primary",
                    "position": arg.position.value,
                    "argument_text": arg.argument_text,
                    "legal_principle": arg.legal_principle,
                    "supporting_reasoning": arg.supporting_reasoning,
                    "supporting_cases": arg.supporting_cases,
                    "strength_score": arg.strength_score,
                    "confidence_score": arg.confidence_score
                })
            
            # Supporting arguments
            for arg in structure.supporting_arguments:
                serialized_args.append({
                    "argument_id": arg.argument_id,
                    "type": "supporting",
                    "position": arg.position.value,
                    "argument_text": arg.argument_text,
                    "legal_principle": arg.legal_principle,
                    "supporting_reasoning": arg.supporting_reasoning,
                    "strength_score": arg.strength_score,
                    "confidence_score": arg.confidence_score
                })
            
            # Counterarguments
            for arg in structure.counterarguments:
                serialized_args.append({
                    "argument_id": arg.argument_id,
                    "type": "counterargument",
                    "position": arg.position.value,
                    "argument_text": arg.argument_text,
                    "legal_principle": arg.legal_principle,
                    "distinguishing_factors": arg.distinguishing_factors,
                    "strength_score": arg.strength_score,
                    "confidence_score": arg.confidence_score
                })
            
            # Mitigation arguments
            for arg in structure.mitigation_arguments:
                serialized_args.append({
                    "argument_id": arg.argument_id,
                    "type": "mitigation",
                    "position": arg.position.value,
                    "argument_text": arg.argument_text,
                    "mitigation_strategies": arg.mitigation_strategies,
                    "strength_score": arg.strength_score,
                    "confidence_score": arg.confidence_score
                })
            
            # Add structure metadata
            structure_metadata = {
                "structure_id": structure.structure_id,
                "type": "structure_summary",
                "legal_question": structure.legal_question,
                "jurisdiction": structure.jurisdiction,
                "overall_strength": structure.overall_strength,
                "persuasiveness_rating": structure.persuasiveness_rating,
                "completeness_rating": structure.completeness_rating,
                "estimated_success_probability": structure.estimated_success_probability,
                "argument_strategy": structure.argument_strategy,
                "risk_analysis": structure.risk_analysis,
                "key_precedents": structure.key_precedents,
                "precedent_hierarchy": structure.precedent_hierarchy,
                "created_at": structure.created_at.isoformat()
            }
            
            serialized_args.append(structure_metadata)
            
            return serialized_args
            
        except Exception as e:
            logger.error(f"❌ Error serializing argument structure: {e}")
            return []
    
    def _update_performance_metrics(self, processing_time: float, structure: ArgumentStructure):
        """Update system performance metrics"""
        try:
            self.performance_metrics["arguments_structured"] += 1
            self.performance_metrics["success_predictions"].append(structure.estimated_success_probability)
            
            # Update average structure time
            current_avg = self.performance_metrics["average_structure_time"]
            total_structures = self.performance_metrics["arguments_structured"]
            
            new_avg = ((current_avg * (total_structures - 1)) + processing_time) / total_structures
            self.performance_metrics["average_structure_time"] = new_avg
            
            # Update average strength score
            current_strength_avg = self.performance_metrics["average_strength_score"]
            new_strength_avg = ((current_strength_avg * (total_structures - 1)) + structure.overall_strength) / total_structures
            self.performance_metrics["average_strength_score"] = new_strength_avg
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            stats = {
                "system_status": "operational",
                "performance_metrics": self.performance_metrics,
                "cache_stats": {
                    "argument_cache_size": len(self.argument_cache),
                    "precedent_cache_size": len(self.precedent_cache)
                },
                "configuration": {
                    "max_arguments_per_type": self.max_arguments_per_type,
                    "min_strength_threshold": self.min_strength_threshold,
                    "max_counterarguments": self.max_counterarguments
                },
                "available_templates": list(self.argument_templates.keys()),
                "ai_clients": {
                    "gemini_available": self.gemini_api_key is not None,
                    "groq_available": self.groq_client is not None
                }
            }
            
            # Add success rate statistics
            if self.performance_metrics["success_predictions"]:
                predictions = self.performance_metrics["success_predictions"]
                stats["success_statistics"] = {
                    "average_predicted_success": sum(predictions) / len(predictions),
                    "high_confidence_cases": len([p for p in predictions if p >= 0.8]),
                    "low_confidence_cases": len([p for p in predictions if p <= 0.4])
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting system stats: {e}")
            return {"error": str(e)}


# Global instance
_argument_structurer = None

async def get_argument_structurer() -> LegalArgumentStructurer:
    """Get initialized legal argument structurer"""
    global _argument_structurer
    
    if _argument_structurer is None:
        _argument_structurer = LegalArgumentStructurer()
        await _argument_structurer.initialize()
    
    return _argument_structurer


if __name__ == "__main__":
    # Test the legal argument structurer
    async def test_structurer():
        structurer = await get_argument_structurer()
        
        argument_data = {
            "legal_question": "Can plaintiff recover damages for breach of web development contract?",
            "jurisdiction": "CA",
            "case_type": "contract_breach",
            "position": "plaintiff",
            "precedents": [
                {
                    "case_title": "Test v. Developer",
                    "citation": "123 Cal. App. 4th 456 (2020)",
                    "relevance_score": 0.85,
                    "authority_score": 0.75
                }
            ]
        }
        
        structured_args = await structurer.structure_legal_arguments(argument_data)
        print(f"Structured {len(structured_args)} arguments")
        
        for arg in structured_args[:3]:
            if arg.get("type") != "structure_summary":
                print(f"Type: {arg.get('type')}, Strength: {arg.get('strength_score', 0.0):.2f}")
    
    asyncio.run(test_structurer())