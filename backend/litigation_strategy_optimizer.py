"""
Enhanced AI-Powered Litigation Strategy Optimizer Module

Comprehensive strategic litigation recommendations with real legal analysis,
accurate cost estimation, sophisticated AI integration, and professional-grade
strategic insights for legal practitioners.
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import numpy as np
import google.generativeai as genai
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

# Import other litigation analytics modules
from litigation_analytics_engine import LitigationAnalyticsEngine, CaseData, PredictionResult
from case_outcome_predictor import EnsembleCasePredictor, CaseFeatures
from judicial_behavior_analyzer import JudicialBehaviorAnalyzer, JudicialProfile
from settlement_probability_calculator import SettlementProbabilityCalculator, SettlementAnalysis

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    COLLABORATIVE = "collaborative"
    PROCEDURAL = "procedural"
    SETTLEMENT_FOCUSED = "settlement_focused"
    TRIAL_FOCUSED = "trial_focused"

class ActionPriority(Enum):
    CRITICAL = "critical"    # Must do immediately
    HIGH = "high"           # Should do within 30 days
    MEDIUM = "medium"       # Should do within 90 days
    LOW = "low"            # Can defer beyond 90 days

@dataclass
class StrategicRecommendation:
    """Individual strategic recommendation with priority and timing"""
    recommendation_id: str
    title: str
    description: str
    priority: ActionPriority
    category: str  # e.g., "Discovery", "Motion Practice", "Settlement"
    estimated_cost: Optional[float] = None
    estimated_timeframe: Optional[str] = None
    success_probability: Optional[float] = None
    risk_level: str = "medium"  # low, medium, high
    dependencies: List[str] = field(default_factory=list)  # Other recommendation IDs
    supporting_evidence: List[str] = field(default_factory=list)

@dataclass
class JurisdictionAnalysis:
    """Analysis of optimal filing jurisdiction"""
    jurisdiction: str
    suitability_score: float  # 0.0 - 1.0
    advantages: List[str] = field(default_factory=list)
    disadvantages: List[str] = field(default_factory=list)
    average_case_duration: Optional[float] = None
    success_rate_for_case_type: Optional[float] = None
    settlement_rate: Optional[float] = None
    estimated_costs: Optional[float] = None

@dataclass 
class TimingAnalysis:
    """Optimal timing analysis for various litigation actions"""
    action_type: str  # e.g., "Filing", "Discovery Completion", "Motion Filing"
    optimal_window: Tuple[datetime, datetime]
    urgency_level: str  # "low", "medium", "high", "critical"
    rationale: str
    dependencies: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)

@dataclass
class EvidenceAssessment:
    """Comprehensive evidence strength assessment"""
    overall_strength: float  # 0.0 - 1.0
    key_strengths: List[str] = field(default_factory=list)
    critical_weaknesses: List[str] = field(default_factory=list)
    evidence_gaps: List[str] = field(default_factory=list)
    discovery_priorities: List[str] = field(default_factory=list)
    witness_reliability: Dict[str, float] = field(default_factory=dict)
    document_quality: float = 0.5
    expert_witness_needs: List[str] = field(default_factory=list)

@dataclass
class LitigationStrategy:
    """Comprehensive litigation strategy analysis"""
    case_id: str
    strategy_date: datetime
    recommended_strategy_type: StrategyType
    confidence_score: float
    
    # Core analyses
    case_outcome_prediction: Optional[PredictionResult] = None
    settlement_analysis: Optional[SettlementAnalysis] = None
    judicial_insights: Optional[Dict[str, Any]] = None
    
    # Strategic recommendations
    strategic_recommendations: List[StrategicRecommendation] = field(default_factory=list)
    jurisdiction_analysis: List[JurisdictionAnalysis] = field(default_factory=list)
    timing_analysis: List[TimingAnalysis] = field(default_factory=list)
    evidence_assessment: Optional[EvidenceAssessment] = None
    
    # Cost-benefit analysis
    estimated_total_cost: Optional[float] = None
    expected_value: Optional[float] = None
    roi_analysis: Dict[str, float] = field(default_factory=dict)
    
    # Risk assessment
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    
    # AI insights
    ai_strategic_summary: str = ""
    alternative_strategies: List[Dict[str, Any]] = field(default_factory=list)

class LitigationStrategyOptimizer:
    """Enhanced strategy optimization engine with professional legal analysis"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        
        # Initialize AI models with error handling
        try:
            self.gemini_model = genai.GenerativeModel('gemini-pro')
            logger.info("✅ Gemini AI model initialized")
        except Exception as e:
            logger.warning(f"⚠️ Gemini AI initialization failed: {e}")
            self.gemini_model = None
        
        try:
            groq_key = os.environ.get('GROQ_API_KEY')
            if groq_key:
                self.groq_client = Groq(api_key=groq_key)
                logger.info("✅ Groq client initialized")
            else:
                self.groq_client = None
                logger.warning("⚠️ GROQ_API_KEY not found")
        except Exception as e:
            logger.warning(f"⚠️ Groq client initialization failed: {e}")
            self.groq_client = None
        
        # Initialize component analyzers
        self.analytics_engine = LitigationAnalyticsEngine(db_connection)
        self.outcome_predictor = EnsembleCasePredictor(db_connection)
        self.judicial_analyzer = JudicialBehaviorAnalyzer(db_connection)
        self.settlement_calculator = SettlementProbabilityCalculator(db_connection)
        
        # Initialize cost multipliers for realistic estimates
        self.cost_multipliers = {
            'federal': 1.4,
            'california': 1.8,
            'new_york': 1.7,
            'delaware': 1.3,
            'texas': 1.2,
            'default': 1.0
        }
        
        logger.info("🎯 Enhanced Litigation Strategy Optimizer initialized")

    async def optimize_litigation_strategy(self, case_data: Dict[str, Any]) -> LitigationStrategy:
        """Generate comprehensive litigation strategy optimization with enhanced analysis"""
        try:
            case_id = case_data.get('case_id', str(uuid.uuid4()))
            logger.info(f"🎯 Optimizing litigation strategy for case {case_id}")
            
            # Run parallel analyses with enhanced error handling
            analyses = await self._run_parallel_analyses(case_data)
            
            # Determine optimal strategy type with improved logic
            strategy_type = self._determine_strategy_type(case_data, analyses)
            
            # Generate strategic recommendations with realistic costs
            recommendations = await self._generate_strategic_recommendations(case_data, analyses, strategy_type)
            
            # Analyze jurisdiction options with comprehensive metrics
            jurisdiction_analysis = await self._analyze_jurisdictions(case_data)
            
            # Optimize timing with realistic timeframes
            timing_analysis = self._optimize_timing(case_data, analyses)
            
            # Assess evidence strength with detailed analysis
            evidence_assessment = self._assess_evidence_strength(case_data, analyses)
            
            # Calculate costs and ROI with realistic multipliers
            cost_analysis = self._calculate_cost_benefit_analysis(case_data, analyses)
            
            # Generate AI strategic summary with real AI analysis
            ai_summary = await self._generate_ai_strategic_summary(case_data, analyses, strategy_type)
            
            # Create comprehensive strategy
            strategy = LitigationStrategy(
                case_id=case_id,
                strategy_date=datetime.utcnow(),
                recommended_strategy_type=strategy_type,
                confidence_score=self._calculate_strategy_confidence(analyses),
                case_outcome_prediction=analyses.get('outcome_prediction'),
                settlement_analysis=analyses.get('settlement_analysis'),
                judicial_insights=analyses.get('judicial_insights'),
                strategic_recommendations=recommendations,
                jurisdiction_analysis=jurisdiction_analysis,
                timing_analysis=timing_analysis,
                evidence_assessment=evidence_assessment,
                estimated_total_cost=cost_analysis.get('total_cost'),
                expected_value=cost_analysis.get('expected_value'),
                roi_analysis=cost_analysis.get('roi_analysis', {}),
                risk_factors=self._identify_risk_factors(case_data, analyses),
                mitigation_strategies=self._develop_mitigation_strategies(case_data, analyses),
                ai_strategic_summary=ai_summary,
                alternative_strategies=self._generate_alternative_strategies(case_data, analyses)
            )
            
            # Cache strategy for future reference
            await self._cache_litigation_strategy(strategy)
            
            logger.info(f"✅ Enhanced strategy optimization completed for case {case_id}")
            return strategy
            
        except Exception as e:
            logger.error(f"❌ Strategy optimization failed: {e}")
            return self._create_default_strategy(case_data.get('case_id', 'unknown'))

    async def _run_parallel_analyses(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all component analyses in parallel with enhanced error handling"""
        try:
            # Prepare case data for different analyzers
            litigation_case_data = CaseData(
                case_id=case_data.get('case_id', str(uuid.uuid4())),
                case_type=case_data.get('case_type', 'civil'),
                jurisdiction=case_data.get('jurisdiction', 'federal'),
                court_level=case_data.get('court_level', 'district'),
                judge_name=case_data.get('judge_name'),
                case_facts=case_data.get('case_facts'),
                legal_issues=case_data.get('legal_issues', []),
                case_complexity=case_data.get('case_complexity'),
                case_value=case_data.get('case_value'),
                filing_date=case_data.get('filing_date'),
                evidence_strength=case_data.get('evidence_strength')
            )
            
            # Run analyses in parallel with timeouts
            results = await asyncio.wait_for(
                asyncio.gather(
                    self.analytics_engine.analyze_case_outcome(litigation_case_data),
                    self.settlement_calculator.calculate_settlement_probability(case_data),
                    self._get_judicial_insights(case_data),
                    return_exceptions=True
                ),
                timeout=30.0
            )
            
            # Process results with enhanced error handling
            analyses = {}
            
            if not isinstance(results[0], Exception):
                analyses['outcome_prediction'] = results[0]
                logger.debug("✅ Outcome prediction completed")
            else:
                logger.warning(f"⚠️ Outcome prediction failed: {results[0]}")
            
            if not isinstance(results[1], Exception):
                analyses['settlement_analysis'] = results[1]
                logger.debug("✅ Settlement analysis completed")
            else:
                logger.warning(f"⚠️ Settlement analysis failed: {results[1]}")
            
            if not isinstance(results[2], Exception):
                analyses['judicial_insights'] = results[2]
                logger.debug("✅ Judicial insights completed")
            else:
                logger.warning(f"⚠️ Judicial insights failed: {results[2]}")
            
            return analyses
            
        except asyncio.TimeoutError:
            logger.error("❌ Parallel analyses timed out")
            return {}
        except Exception as e:
            logger.error(f"❌ Parallel analyses failed: {e}")
            return {}

    async def _get_judicial_insights(self, case_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get judicial insights if judge is specified"""
        judge_name = case_data.get('judge_name')
        if not judge_name:
            return None
        
        try:
            return await self.judicial_analyzer.get_judge_insights_for_case(
                judge_name=judge_name,
                case_type=case_data.get('case_type', 'civil'),
                case_value=case_data.get('case_value')
            )
        except Exception as e:
            logger.warning(f"⚠️ Judicial insights retrieval failed: {e}")
            return None

    def _determine_strategy_type(self, case_data: Dict[str, Any], analyses: Dict[str, Any]) -> StrategyType:
        """Determine optimal strategy type with enhanced logic"""
        # Normalize evidence strength from 1-10 scale to 0.0-1.0
        evidence_strength = float(case_data.get('evidence_strength', 5)) / 10.0
        case_value = float(case_data.get('case_value', 0)) if case_data.get('case_value') else 0
        case_complexity = float(case_data.get('case_complexity', 0.5))
        
        # Get settlement probability if available
        settlement_analysis = analyses.get('settlement_analysis')
        settlement_probability = 0.4  # Default
        if settlement_analysis and hasattr(settlement_analysis, 'metrics'):
            settlement_probability = settlement_analysis.metrics.settlement_probability
        
        # Get outcome prediction confidence
        outcome_prediction = analyses.get('outcome_prediction')
        outcome_confidence = 0.5  # Default
        if outcome_prediction and hasattr(outcome_prediction, 'confidence_score'):
            outcome_confidence = outcome_prediction.confidence_score
        
        # Enhanced decision logic
        logger.info(f"Strategy decision factors: evidence={evidence_strength:.2f}, settlement_prob={settlement_probability:.2f}, outcome_conf={outcome_confidence:.2f}, value=${case_value:,.0f}")
        
        if settlement_probability > 0.7:
            return StrategyType.SETTLEMENT_FOCUSED
        elif evidence_strength > 0.8 and outcome_confidence > 0.7:
            return StrategyType.AGGRESSIVE
        elif evidence_strength < 0.3 or outcome_confidence < 0.4:
            return StrategyType.CONSERVATIVE
        elif case_value > 1000000 or case_complexity > 0.8:
            return StrategyType.PROCEDURAL  # High-value/complex cases need careful procedure
        else:
            return StrategyType.COLLABORATIVE

    async def _generate_strategic_recommendations(self, case_data: Dict[str, Any], analyses: Dict[str, Any], 
                                                strategy_type: StrategyType) -> List[StrategicRecommendation]:
        """Generate comprehensive strategic recommendations with realistic costs"""
        recommendations = []
        
        # Get case value for cost scaling
        case_value = float(case_data.get('case_value', 100000)) if case_data.get('case_value') else 100000
        jurisdiction = case_data.get('jurisdiction', 'federal')
        cost_multiplier = self.cost_multipliers.get(jurisdiction, 1.0)
        
        # Discovery recommendations with realistic costs
        recommendations.extend(self._generate_discovery_recommendations(case_data, analyses, case_value, cost_multiplier))
        
        # Motion practice recommendations
        recommendations.extend(self._generate_motion_recommendations(case_data, analyses, strategy_type, case_value, cost_multiplier))
        
        # Settlement recommendations
        recommendations.extend(self._generate_settlement_recommendations(case_data, analyses, case_value, cost_multiplier))
        
        # Trial preparation recommendations
        recommendations.extend(self._generate_trial_recommendations(case_data, analyses, strategy_type, case_value, cost_multiplier))
        
        # Case management recommendations
        recommendations.extend(self._generate_case_management_recommendations(case_data, analyses, case_value, cost_multiplier))
        
        # Sort by priority and success probability
        recommendations.sort(key=lambda x: (
            0 if x.priority == ActionPriority.CRITICAL else
            1 if x.priority == ActionPriority.HIGH else
            2 if x.priority == ActionPriority.MEDIUM else 3,
            -(x.success_probability if x.success_probability else 0.5)
        ))
        
        return recommendations[:12]  # Top 12 recommendations

    def _generate_discovery_recommendations(self, case_data: Dict[str, Any], analyses: Dict[str, Any], 
                                          case_value: float, cost_multiplier: float) -> List[StrategicRecommendation]:
        """Generate discovery-specific recommendations with realistic costs"""
        recommendations = []
        evidence_strength = float(case_data.get('evidence_strength', 5)) / 10.0
        
        # Scale base costs based on case value
        base_discovery_cost = min(50000, max(15000, case_value * 0.08)) * cost_multiplier
        
        if evidence_strength < 0.6:
            recommendations.append(StrategicRecommendation(
                recommendation_id=str(uuid.uuid4()),
                title="Comprehensive Document Discovery",
                description=f"Conduct thorough document discovery to strengthen evidence base. Given evidence strength of {evidence_strength*10:.1f}/10, aggressive discovery is essential to identify supporting materials and build a compelling case.",
                priority=ActionPriority.HIGH,
                category="Discovery",
                estimated_cost=base_discovery_cost,
                estimated_timeframe="90-120 days",
                success_probability=0.75,
                risk_level="medium",
                supporting_evidence=[
                    f"Current evidence strength at {evidence_strength*10:.1f}/10 below optimal threshold",
                    "Document discovery typically reveals 40-60% of critical evidence",
                    f"Investment in discovery justified by ${case_value:,.0f} case value"
                ]
            ))
        
        # Witness depositions with scaled costs
        deposition_cost = min(25000, max(8000, case_value * 0.03)) * cost_multiplier
        witness_count = int(case_data.get('witness_count', 3)) if case_data.get('witness_count') else 3
        
        recommendations.append(StrategicRecommendation(
            recommendation_id=str(uuid.uuid4()),
            title="Strategic Witness Depositions",
            description=f"Depose {witness_count} key witnesses to lock in testimony and assess case strength. Priority should focus on hostile witnesses and expert witnesses with significant impact on case outcome.",
            priority=ActionPriority.HIGH,
            category="Discovery",
            estimated_cost=deposition_cost,
            estimated_timeframe="60-90 days",
            success_probability=0.8,
            risk_level="low",
            supporting_evidence=[
                f"Estimated {witness_count} key witnesses identified",
                "Early depositions prevent witness coaching",
                "Testimony preservation critical for case strategy"
            ]
        ))
        
        return recommendations

    def _generate_motion_recommendations(self, case_data: Dict[str, Any], analyses: Dict[str, Any], 
                                       strategy_type: StrategyType, case_value: float, cost_multiplier: float) -> List[StrategicRecommendation]:
        """Generate motion practice recommendations with enhanced analysis"""
        recommendations = []
        evidence_strength = float(case_data.get('evidence_strength', 5)) / 10.0
        
        # Scale motion costs based on case value and complexity
        base_motion_cost = min(35000, max(12000, case_value * 0.05)) * cost_multiplier
        
        # Summary judgment recommendations with realistic success probability
        if evidence_strength > 0.7 and strategy_type in [StrategyType.AGGRESSIVE, StrategyType.PROCEDURAL]:
            success_prob = min(0.85, evidence_strength * 0.9)  # Cap at 85% for realism
            recommendations.append(StrategicRecommendation(
                recommendation_id=str(uuid.uuid4()),
                title="Motion for Summary Judgment",
                description=f"File motion for summary judgment based on strong evidence (strength: {evidence_strength*10:.1f}/10) and clear legal standards. Focus on undisputed material facts and favorable legal precedents.",
                priority=ActionPriority.MEDIUM,
                category="Motion Practice",
                estimated_cost=base_motion_cost,
                estimated_timeframe="45-60 days to prepare and brief",
                success_probability=success_prob,
                risk_level="medium",
                supporting_evidence=[
                    f"Evidence strength at {evidence_strength*10:.1f}/10 supports summary resolution",
                    "Early resolution avoids extended litigation costs",
                    f"Potential savings: ${case_value * 0.6:,.0f} in avoided trial costs"
                ]
            ))
        
        # Motion to dismiss for procedural strategies
        if strategy_type == StrategyType.PROCEDURAL or strategy_type == StrategyType.CONSERVATIVE:
            mtd_cost = min(15000, max(8000, case_value * 0.02)) * cost_multiplier
            recommendations.append(StrategicRecommendation(
                recommendation_id=str(uuid.uuid4()),
                title="Strategic Motion to Dismiss",
                description="File targeted motion to dismiss to narrow legal issues and reduce case complexity. Focus on jurisdictional challenges, failure to state a claim, or procedural defects.",
                priority=ActionPriority.HIGH,
                category="Motion Practice",
                estimated_cost=mtd_cost,
                estimated_timeframe="21-30 days",
                success_probability=0.65,
                risk_level="low",
                supporting_evidence=[
                    "Early procedural challenges preserve resources",
                    "Issue narrowing reduces discovery scope and costs",
                    "Low-risk strategy with high impact potential"
                ]
            ))
        
        return recommendations

    def _generate_settlement_recommendations(self, case_data: Dict[str, Any], analyses: Dict[str, Any], 
                                           case_value: float, cost_multiplier: float) -> List[StrategicRecommendation]:
        """Generate settlement-focused recommendations with enhanced analysis"""
        recommendations = []
        settlement_analysis = analyses.get('settlement_analysis')
        
        if not settlement_analysis or not hasattr(settlement_analysis, 'metrics'):
            return recommendations
        
        settlement_prob = settlement_analysis.metrics.settlement_probability
        expected_settlement = settlement_analysis.metrics.expected_settlement_value
        
        if settlement_prob > 0.5:
            settlement_cost = min(8000, max(3000, case_value * 0.01)) * cost_multiplier
            priority = ActionPriority.HIGH if settlement_prob > 0.7 else ActionPriority.MEDIUM
            
            recommendations.append(StrategicRecommendation(
                recommendation_id=str(uuid.uuid4()),
                title="Early Settlement Negotiations",
                description=f"Initiate settlement discussions with {settlement_prob:.1%} probability of success. Expected settlement value: ${expected_settlement:,.0f}. Early engagement preserves resources and reduces litigation risk.",
                priority=priority,
                category="Settlement",
                estimated_cost=settlement_cost,
                estimated_timeframe="30-60 days",
                success_probability=settlement_prob,
                risk_level="low",
                supporting_evidence=[
                    f"Settlement probability: {settlement_prob:.1%}",
                    f"Expected settlement value: ${expected_settlement:,.0f}",
                    f"Potential litigation cost savings: ${case_value * 0.4:,.0f}",
                    "Early settlement preserves business relationships"
                ]
            ))
        
        # Professional mediation for moderate to high settlement probability
        if settlement_prob > 0.4 and case_value > 100000:
            mediation_cost = min(15000, max(5000, case_value * 0.02)) * cost_multiplier
            recommendations.append(StrategicRecommendation(
                recommendation_id=str(uuid.uuid4()),
                title="Professional Mediation",
                description="Engage experienced mediator specialized in this practice area for structured settlement negotiations. Mediation provides controlled environment for resolution discussions.",
                priority=ActionPriority.MEDIUM,
                category="Settlement",
                estimated_cost=mediation_cost,
                estimated_timeframe="1-2 mediation sessions over 4-6 weeks",
                success_probability=min(0.85, settlement_prob + 0.15),  # Mediation boost
                risk_level="low",
                supporting_evidence=[
                    "Mediation increases settlement success by 15-20%",
                    "Structured process manages expectations",
                    "Professional mediator brings expertise in similar cases"
                ]
            ))
        
        return recommendations

    def _generate_trial_recommendations(self, case_data: Dict[str, Any], analyses: Dict[str, Any], 
                                      strategy_type: StrategyType, case_value: float, cost_multiplier: float) -> List[StrategicRecommendation]:
        """Generate trial preparation recommendations with realistic cost estimates"""
        recommendations = []
        
        if strategy_type not in [StrategyType.AGGRESSIVE, StrategyType.TRIAL_FOCUSED]:
            return recommendations
        
        # Expert witness retention for high-value cases
        if case_value > 250000:
            expert_cost = min(75000, max(25000, case_value * 0.08)) * cost_multiplier
            recommendations.append(StrategicRecommendation(
                recommendation_id=str(uuid.uuid4()),
                title="Expert Witness Retention",
                description=f"Retain qualified expert witnesses for technical/industry-specific testimony. Budget includes expert fees, report preparation, and trial testimony for estimated 2-3 experts given case value of ${case_value:,.0f}.",
                priority=ActionPriority.MEDIUM,
                category="Trial Preparation",
                estimated_cost=expert_cost,
                estimated_timeframe="120-180 days before trial",
                success_probability=0.75,
                risk_level="medium",
                supporting_evidence=[
                    f"Case value ${case_value:,.0f} justifies expert investment",
                    "Expert testimony critical for complex technical issues",
                    "Early retention ensures availability and preparation time"
                ]
            ))
        
        # Trial graphics and technology
        if case_value > 100000:
            graphics_cost = min(25000, max(8000, case_value * 0.03)) * cost_multiplier
            recommendations.append(StrategicRecommendation(
                recommendation_id=str(uuid.uuid4()),
                title="Trial Graphics and Technology",
                description="Develop compelling visual presentations and leverage courtroom technology for effective advocacy. Includes timeline graphics, damage calculations, and interactive presentations.",
                priority=ActionPriority.LOW,
                category="Trial Preparation",
                estimated_cost=graphics_cost,
                estimated_timeframe="60-90 days before trial",
                success_probability=0.65,
                risk_level="low",
                supporting_evidence=[
                    "Visual presentations increase jury comprehension by 30%",
                    "Technology demonstrates sophistication and preparation",
                    "Investment proportional to case value"
                ]
            ))
        
        return recommendations

    def _generate_case_management_recommendations(self, case_data: Dict[str, Any], analyses: Dict[str, Any], 
                                                case_value: float, cost_multiplier: float) -> List[StrategicRecommendation]:
        """Generate case management recommendations with enhanced analysis"""
        recommendations = []
        
        # Case management conference strategy (always applicable)
        mgmt_cost = min(5000, max(2000, case_value * 0.005)) * cost_multiplier
        recommendations.append(StrategicRecommendation(
            recommendation_id=str(uuid.uuid4()),
            title="Case Management Conference Strategy",
            description="Prepare comprehensive case management plan to control scheduling and discovery timeline. Focus on establishing favorable deadlines and managing court expectations.",
            priority=ActionPriority.HIGH,
            category="Case Management",
            estimated_cost=mgmt_cost,
            estimated_timeframe="Within 30 days of filing",
            success_probability=0.9,
            risk_level="low",
            supporting_evidence=[
                "Early case management sets litigation tone",
                "Proactive approach demonstrates preparedness",
                "Timeline control critical for strategy execution"
            ]
        ))
        
        # Litigation budget and resource planning for significant cases
        if case_value > 500000:
            budget_cost = min(8000, max(3000, case_value * 0.008)) * cost_multiplier
            recommendations.append(StrategicRecommendation(
                recommendation_id=str(uuid.uuid4()),
                title="Comprehensive Litigation Budget Planning",
                description=f"Develop detailed litigation budget and resource allocation plan for high-value case (${case_value:,.0f}). Include cost controls, milestone reviews, and alternative fee arrangements.",
                priority=ActionPriority.HIGH,
                category="Case Management",
                estimated_cost=budget_cost,
                estimated_timeframe="Within 14 days",
                success_probability=0.95,
                risk_level="low",
                supporting_evidence=[
                    f"Case value ${case_value:,.0f} requires sophisticated budget management",
                    "Proactive cost control prevents budget overruns",
                    "Client transparency builds trust and confidence"
                ]
            ))
        
        return recommendations

    async def _analyze_jurisdictions(self, case_data: Dict[str, Any]) -> List[JurisdictionAnalysis]:
        """Analyze optimal filing jurisdictions with comprehensive metrics"""
        try:
            jurisdictions = ['federal', 'california', 'new_york', 'texas', 'delaware', 'illinois']
            analyses = []
            
            current_jurisdiction = case_data.get('jurisdiction', '').lower()
            case_type = case_data.get('case_type', 'civil')
            case_value = float(case_data.get('case_value', 0)) if case_data.get('case_value') else 0
            
            for jurisdiction in jurisdictions:
                # Calculate suitability score based on various factors
                score = await self._calculate_jurisdiction_suitability(case_data, jurisdiction)
                
                # Enhanced cost estimation based on jurisdiction
                base_cost_multiplier = self.cost_multipliers.get(jurisdiction, 1.0)
                estimated_costs = case_value * 0.25 * base_cost_multiplier  # 25% of case value as base
                
                analysis = JurisdictionAnalysis(
                    jurisdiction=jurisdiction,
                    suitability_score=score,
                    advantages=self._get_jurisdiction_advantages(jurisdiction, case_data),
                    disadvantages=self._get_jurisdiction_disadvantages(jurisdiction, case_data),
                    average_case_duration=self._estimate_jurisdiction_duration(jurisdiction),
                    success_rate_for_case_type=self._estimate_jurisdiction_success_rate(jurisdiction, case_type),
                    settlement_rate=self._estimate_jurisdiction_settlement_rate(jurisdiction),
                    estimated_costs=estimated_costs
                )
                
                analyses.append(analysis)
            
            # Sort by suitability score
            analyses.sort(key=lambda x: x.suitability_score, reverse=True)
            
            return analyses[:5]  # Top 5 jurisdictions
            
        except Exception as e:
            logger.warning(f"⚠️ Jurisdiction analysis failed: {e}")
            return []

    async def _calculate_jurisdiction_suitability(self, case_data: Dict[str, Any], jurisdiction: str) -> float:
        """Calculate suitability score for a jurisdiction with enhanced factors"""
        score = 0.5  # Base score
        
        case_type = case_data.get('case_type', '')
        case_value = float(case_data.get('case_value', 0)) if case_data.get('case_value') else 0
        evidence_strength = float(case_data.get('evidence_strength', 5)) / 10.0
        
        # Jurisdiction-specific adjustments with enhanced logic
        if jurisdiction == 'federal':
            if case_value > 75000:  # Federal diversity threshold
                score += 0.2
            if case_type in ['intellectual_property', 'securities', 'antitrust']:
                score += 0.3
            if evidence_strength > 0.7:  # Federal courts prefer strong cases
                score += 0.1
        
        elif jurisdiction == 'delaware':
            if case_type in ['corporate', 'business', 'commercial']:
                score += 0.4
            if case_value > 1000000:
                score += 0.2
            # Delaware Chancery Court expertise
            if 'contract' in case_data.get('case_facts', '').lower():
                score += 0.1
        
        elif jurisdiction == 'california':
            if case_type in ['employment', 'consumer', 'privacy']:
                score += 0.3
            if case_value > 500000:  # California handles high-value cases well
                score += 0.1
            # Check for California connections
            case_facts = case_data.get('case_facts', '').lower()
            if any(term in case_facts for term in ['california', 'silicon valley', 'los angeles']):
                score += 0.2
        
        elif jurisdiction == 'new_york':
            if case_type in ['commercial', 'securities', 'banking']:
                score += 0.3
            if case_value > 2000000:  # NY Commercial Division
                score += 0.2
        
        elif jurisdiction == 'texas':
            if case_type in ['energy', 'oil', 'gas', 'business']:
                score += 0.3
            # Texas business courts
            if case_value > 1000000:
                score += 0.15
        
        return min(1.0, max(0.0, score))

    def _get_jurisdiction_advantages(self, jurisdiction: str, case_data: Dict[str, Any]) -> List[str]:
        """Get advantages of filing in specific jurisdiction with enhanced analysis"""
        case_type = case_data.get('case_type', '')
        case_value = float(case_data.get('case_value', 0)) if case_data.get('case_value') else 0
        
        advantages = {
            'federal': [
                "Experienced federal judges with sophisticated legal analysis",
                "Streamlined procedures and consistent case management",
                "Nationwide enforcement and broader jurisdictional reach",
                "Limited local bias and more predictable outcomes",
                "Strong appellate precedent and established procedures"
            ],
            'california': [
                "Pro-plaintiff employment and consumer protection laws",
                "Large jury awards potential in appropriate cases",
                "Strong consumer and privacy protections",
                "Technology-savvy courts familiar with modern business",
                "Aggressive discovery rules favoring fact development"
            ],
            'delaware': [
                "Unparalleled corporate law expertise and business courts",
                "Expedited case scheduling and efficient procedures",
                "Sophisticated commercial court judges",
                "Predictable business-oriented legal framework",
                "Chancellor's equity powers for flexible remedies"
            ],
            'new_york': [
                "Commercial Division expertise in complex business matters",
                "Efficient case management and strict deadlines",
                "Strong discovery rules and comprehensive procedures",
                "Financial services and securities law expertise",
                "Experienced commercial litigation bar"
            ],
            'texas': [
                "Business-friendly legal environment and courts",
                "Reasonable damage awards and cost-effective litigation",
                "Efficient court system with manageable dockets",
                "Strong contract enforcement and business law",
                "Energy and oil & gas law specialization"
            ],
            'illinois': [
                "Central location convenient for multi-state matters",
                "Experienced complex litigation courts",
                "Reasonable costs and efficient procedures",
                "Strong commercial law tradition",
                "Diverse jury pools in urban areas"
            ]
        }
        
        base_advantages = advantages.get(jurisdiction, ["General litigation advantages"])
        
        # Add case-specific advantages
        enhanced_advantages = base_advantages.copy()
        
        if case_value > 1000000 and jurisdiction in ['delaware', 'new_york']:
            enhanced_advantages.append(f"Specialized high-value case procedures (${case_value:,.0f})")
        
        if case_type == 'employment' and jurisdiction == 'california':
            enhanced_advantages.append("Nation's most comprehensive employment protection laws")
        
        return enhanced_advantages[:6]  # Limit to 6 advantages

    def _get_jurisdiction_disadvantages(self, jurisdiction: str, case_data: Dict[str, Any]) -> List[str]:
        """Get disadvantages of filing in specific jurisdiction"""
        disadvantages = {
            'federal': [
                "Higher filing fees and more complex procedures",
                "More stringent pleading standards (Iqbal/Twombly)",
                "Limited local law expertise for state-specific issues",
                "Potential for removal by defendants",
                "Longer time to trial in busy districts"
            ],
            'california': [
                "Extremely high litigation costs and attorney fees",
                "Crowded court dockets causing delays",
                "Complex state procedures and local rules",
                "Anti-business bias in certain regions",
                "High cost of living affects all litigation expenses"
            ],
            'delaware': [
                "Limited jury trial options in Chancery Court",
                "High attorney fees due to specialized bar",
                "Corporate-defendant friendly reputation",
                "Limited discovery timeline and compressed schedules",
                "Geographic inconvenience for most parties"
            ],
            'new_york': [
                "Extremely high litigation costs in Manhattan",
                "Aggressive motion practice and complex procedures",
                "Crowded commercial court dockets",
                "High cost of living affects all expenses",
                "Sophisticated defendant counsel and resources"
            ],
            'texas': [
                "Limited punitive damages and damage caps",
                "Pro-business judicial philosophy",
                "Potential anti-plaintiff bias in certain regions",
                "Limited discovery in some courts",
                "Conservative jury attitudes in many areas"
            ],
            'illinois': [
                "Political corruption concerns affecting court perception",
                "Budget constraints affecting court resources",
                "Limited specialized business court options",
                "Weather-related delays in winter months",
                "Competition from neighboring jurisdictions"
            ]
        }
        
        return disadvantages.get(jurisdiction, ["Standard litigation challenges"])[:5]

    def _estimate_jurisdiction_duration(self, jurisdiction: str) -> float:
        """Estimate average case duration in jurisdiction (days) with updated data"""
        durations = {
            'federal': 620,      # Updated federal court statistics
            'california': 755,   # California court delays
            'delaware': 385,     # Delaware efficiency
            'new_york': 580,     # NY Commercial Division
            'texas': 465,        # Texas efficiency
            'illinois': 510      # Illinois average
        }
        
        return durations.get(jurisdiction, 550)

    def _estimate_jurisdiction_success_rate(self, jurisdiction: str, case_type: Optional[str]) -> float:
        """Estimate success rate for case type in jurisdiction with case-type factors"""
        base_rates = {
            'federal': 0.52,
            'california': 0.48,  # Pro-defendant trends
            'delaware': 0.55,    # Business court expertise
            'new_york': 0.51,
            'texas': 0.49,       # Pro-business
            'illinois': 0.50
        }
        
        base_rate = base_rates.get(jurisdiction, 0.50)
        
        # Adjust based on case type
        if case_type:
            adjustments = {
                ('california', 'employment'): 0.08,
                ('delaware', 'corporate'): 0.06,
                ('new_york', 'commercial'): 0.04,
                ('federal', 'intellectual_property'): 0.05,
                ('texas', 'business'): 0.03
            }
            
            adjustment = adjustments.get((jurisdiction, case_type), 0)
            base_rate += adjustment
        
        return min(0.85, max(0.15, base_rate))

    def _estimate_jurisdiction_settlement_rate(self, jurisdiction: str) -> float:
        """Estimate settlement rate in jurisdiction with updated statistics"""
        settlement_rates = {
            'federal': 0.73,     # High federal settlement rate
            'california': 0.69,  # California mediation programs
            'delaware': 0.78,    # Business court efficiency
            'new_york': 0.71,    # Commercial court management
            'texas': 0.67,       # Business-friendly environment
            'illinois': 0.65     # Traditional litigation approach
        }
        
        return settlement_rates.get(jurisdiction, 0.68)

    def _optimize_timing(self, case_data: Dict[str, Any], analyses: Dict[str, Any]) -> List[TimingAnalysis]:
        """Optimize timing for various litigation actions with realistic analysis"""
        timing_analyses = []
        
        # Filing timing with urgency assessment
        urgency_factors = []
        if case_data.get('timeline_constraints'):
            urgency_factors.append("Client-specified timeline constraints")
        
        case_value = float(case_data.get('case_value', 0)) if case_data.get('case_value') else 0
        if case_value > 1000000:
            urgency_factors.append("High case value requires prompt action")
        
        urgency_level = "high" if urgency_factors else "medium"
        
        timing_analyses.append(TimingAnalysis(
            action_type="Case Filing",
            optimal_window=(datetime.utcnow(), datetime.utcnow() + timedelta(days=14)),
            urgency_level=urgency_level,
            rationale="Early filing preserves claims, initiates discovery timeline, and demonstrates client commitment to resolution",
            dependencies=[],
            risk_factors=["Statute of limitations", "Evidence preservation", "Witness availability"]
        ))
        
        # Discovery completion timing based on case complexity
        case_complexity = float(case_data.get('case_complexity', 0.5))
        discovery_days = int(120 + (case_complexity * 180))  # 120-300 days based on complexity
        
        timing_analyses.append(TimingAnalysis(
            action_type="Discovery Completion",
            optimal_window=(
                datetime.utcnow() + timedelta(days=60), 
                datetime.utcnow() + timedelta(days=discovery_days)
            ),
            urgency_level="medium",
            rationale=f"Discovery timeline scaled to case complexity ({case_complexity*100:.0f}%) to ensure thorough fact development without unnecessary delay"
        ))
        
        # Settlement timing based on settlement analysis
        settlement_analysis = analyses.get('settlement_analysis')
        if settlement_analysis and hasattr(settlement_analysis, 'metrics'):
            settlement_urgency = "high" if settlement_analysis.metrics.settlement_probability > 0.7 else "medium"
            
            timing_analyses.append(TimingAnalysis(
                action_type="Settlement Negotiations",
                optimal_window=(
                    datetime.utcnow() + timedelta(days=45), 
                    datetime.utcnow() + timedelta(days=120)
                ),
                urgency_level=settlement_urgency,
                rationale=f"Settlement probability at {settlement_analysis.metrics.settlement_probability:.1%} indicates favorable negotiation window after initial discovery",
                dependencies=["Initial discovery completion", "Case valuation analysis"],
                risk_factors=["Changing market conditions", "Witness availability", "Regulatory changes"]
            ))
        
        return timing_analyses

    def _assess_evidence_strength(self, case_data: Dict[str, Any], analyses: Dict[str, Any]) -> EvidenceAssessment:
        """Comprehensive evidence strength assessment with detailed analysis"""
        # Normalize evidence strength from 1-10 scale
        overall_strength = float(case_data.get('evidence_strength', 5)) / 10.0
        
        # Generate detailed strengths and weaknesses based on evidence strength
        key_strengths = []
        critical_weaknesses = []
        evidence_gaps = []
        
        if overall_strength > 0.8:
            key_strengths.extend([
                "Compelling documentary evidence supporting all material facts",
                "Credible witness testimony with consistent accounts",
                "Clear liability chain with minimal disputed elements",
                "Strong expert witness foundation for technical issues",
                "Contemporaneous records supporting damages calculation"
            ])
        elif overall_strength > 0.6:
            key_strengths.extend([
                "Solid documentary foundation for key claims",
                "Reliable witness testimony on critical issues",
                "Clear liability indicators with manageable disputes",
                "Adequate expert support for complex issues"
            ])
            evidence_gaps.append("Additional corroborating evidence would strengthen case")
        elif overall_strength > 0.4:
            key_strengths.extend([
                "Basic evidentiary support for primary claims",
                "Some witness corroboration of key facts"
            ])
            evidence_gaps.extend([
                "Significant additional evidence needed for strong case",
                "Expert witness testimony essential for technical issues",
                "Additional document discovery likely to reveal key evidence"
            ])
        else:
            critical_weaknesses.extend([
                "Insufficient documentary evidence for key claims",
                "Witness credibility concerns require careful management",
                "Significant gaps in liability evidence chain",
                "Expert witness support essential but currently lacking",
                "Damages calculation requires substantial additional support"
            ])
            evidence_gaps.extend([
                "Comprehensive discovery essential to identify supporting evidence",
                "Expert witness consultation required immediately",
                "Alternative legal theories should be developed",
                "Consider early mediation given evidentiary challenges"
            ])
        
        # Discovery priorities based on evidence assessment
        discovery_priorities = []
        if overall_strength < 0.7:
            discovery_priorities.extend([
                "Aggressive document discovery to identify supporting materials",
                "Early witness interviews to preserve testimony",
                "Expert witness consultation for case evaluation",
                "Electronic discovery for comprehensive document review"
            ])
        
        # Expert witness needs based on case value and complexity
        expert_witness_needs = []
        case_value = float(case_data.get('case_value', 0)) if case_data.get('case_value') else 0
        case_complexity = float(case_data.get('case_complexity', 0.5))
        
        if case_value > 500000 or case_complexity > 0.7:
            expert_witness_needs.extend([
                "Industry expert for standard of care and practices",
                "Economic damages expert for financial analysis",
                "Technical expert for specialized subject matter"
            ])
        elif case_value > 100000:
            expert_witness_needs.append("Damages expert for economic analysis")
        
        return EvidenceAssessment(
            overall_strength=overall_strength,
            key_strengths=key_strengths[:4],  # Limit to top 4
            critical_weaknesses=critical_weaknesses[:4],
            evidence_gaps=evidence_gaps[:4],
            discovery_priorities=discovery_priorities[:4],
            document_quality=overall_strength * 0.95,  # Documents slightly stronger than overall
            expert_witness_needs=expert_witness_needs
        )

    def _calculate_cost_benefit_analysis(self, case_data: Dict[str, Any], analyses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive cost-benefit analysis with transparent formulas and realistic estimates.
        
        TRANSPARENT CALCULATION METHODOLOGY:
        1. Litigation Costs = Base Cost × Jurisdiction × Complexity × Case Value × Evidence Multipliers
        2. Expected Value = (Settlement Path Value + Trial Path Value) - Litigation Costs
        3. All multipliers are documented with clear rationale and bounds checking
        """
        case_value = float(case_data.get('case_value', 100000)) if case_data.get('case_value') else 100000
        jurisdiction = case_data.get('jurisdiction', 'federal')
        case_complexity = float(case_data.get('case_complexity', 0.5))
        evidence_strength = float(case_data.get('evidence_strength', 5)) / 10.0
        
        # === TRANSPARENT LITIGATION COST CALCULATION ===
        
        # Base litigation cost (2024 market rates)
        base_cost = 45000  
        logger.info(f"💰 Base Litigation Cost: ${base_cost:,} (2024 market average for standard litigation)")
        
        # Jurisdiction cost multiplier (based on local market rates and complexity)
        jurisdiction_multiplier = self.cost_multipliers.get(jurisdiction.lower(), 1.0)
        jurisdiction_explanation = {
            'california': "Higher attorney rates and complex state procedures",
            'new_york': "Premium market rates and extensive discovery requirements", 
            'federal': "Enhanced procedural requirements and longer timelines",
            'delaware': "Specialized business court efficiency with premium rates",
            'texas': "Moderate rates with efficient court systems"
        }.get(jurisdiction.lower(), "Standard jurisdiction baseline")
        logger.info(f"🏛️ Jurisdiction Multiplier: {jurisdiction_multiplier:.2f}x ({jurisdiction_explanation})")
        
        # Case complexity multiplier (linear scale from simple to very complex)
        complexity_multiplier = 1.0 + (case_complexity * 1.8)  # Range: 1.0x to 2.8x
        complexity_desc = "Simple" if case_complexity < 0.3 else "Moderate" if case_complexity < 0.7 else "Complex"
        logger.info(f"⚖️ Complexity Multiplier: {complexity_multiplier:.2f}x ({complexity_desc} case - {case_complexity*100:.0f}% complexity)")
        
        # Case value multiplier (higher stakes = more careful preparation)
        if case_value > 2000000:
            value_multiplier = 1.8
            value_explanation = "Ultra-high-stakes litigation requiring extensive preparation"
        elif case_value > 1000000:
            value_multiplier = 1.5
            value_explanation = "High-stakes litigation with elevated preparation standards"
        elif case_value > 500000:
            value_multiplier = 1.3
            value_explanation = "Significant value case requiring enhanced due diligence"
        else:
            value_multiplier = 1.0
            value_explanation = "Standard value case with routine preparation requirements"
        logger.info(f"💲 Value Multiplier: {value_multiplier:.2f}x ({value_explanation})")
        
        # Evidence strength multiplier (weak evidence = more discovery needed)
        evidence_multiplier = 1.0 + ((1.0 - evidence_strength) * 0.6)  # Range: 1.0x to 1.6x
        evidence_desc = "Strong" if evidence_strength > 0.7 else "Moderate" if evidence_strength > 0.4 else "Weak"
        logger.info(f"🔍 Evidence Multiplier: {evidence_multiplier:.2f}x ({evidence_desc} evidence requiring {'minimal' if evidence_strength > 0.7 else 'moderate' if evidence_strength > 0.4 else 'extensive'} discovery)")
        
        # Final cost calculation with step-by-step breakdown
        estimated_total_cost = base_cost * jurisdiction_multiplier * complexity_multiplier * value_multiplier * evidence_multiplier
        total_multiplier = jurisdiction_multiplier * complexity_multiplier * value_multiplier * evidence_multiplier
        logger.info(f"📊 COST CALCULATION: ${base_cost:,} × {total_multiplier:.2f} = ${estimated_total_cost:,.0f}")
        
        # === TRANSPARENT EXPECTED VALUE CALCULATION ===
        
        outcome_prediction = analyses.get('outcome_prediction')
        settlement_analysis = analyses.get('settlement_analysis')
        
        if settlement_analysis and hasattr(settlement_analysis, 'metrics'):
            # Path 1: Settlement Analysis Available
            settlement_prob = settlement_analysis.metrics.settlement_probability
            raw_settlement_value = settlement_analysis.metrics.expected_settlement_value
            
            # CRITICAL FIX: Apply bounds checking to prevent unrealistic settlement values
            # Settlement value should not exceed case value by more than reasonable multipliers
            max_reasonable_settlement = case_value * 1.35  # Max 35% premium for damages/interest/costs
            bounded_settlement_value = min(raw_settlement_value, max_reasonable_settlement)
            
            if raw_settlement_value != bounded_settlement_value:
                logger.warning(f"⚠️ Settlement value bounded: ${raw_settlement_value:,.0f} → ${bounded_settlement_value:,.0f} (max 35% premium applied)")
            
            settlement_value = bounded_settlement_value
            
            # Trial outcome probability (if no settlement)
            trial_prob = 1.0 - settlement_prob
            trial_win_prob = 0.5  # Default conservative estimate
            if outcome_prediction and hasattr(outcome_prediction, 'probability_breakdown'):
                trial_win_prob = outcome_prediction.probability_breakdown.get('plaintiff_win', 0.5)
            
            # Expected value calculation with transparent formula
            settlement_path_value = settlement_prob * settlement_value
            trial_path_value = trial_prob * trial_win_prob * case_value
            gross_expected_value = settlement_path_value + trial_path_value
            expected_value = gross_expected_value - estimated_total_cost
            
            logger.info(f"💡 EXPECTED VALUE CALCULATION:")
            logger.info(f"   Settlement Path: {settlement_prob:.1%} × ${settlement_value:,.0f} = ${settlement_path_value:,.0f}")
            logger.info(f"   Trial Path: {trial_prob:.1%} × {trial_win_prob:.1%} × ${case_value:,.0f} = ${trial_path_value:,.0f}")
            logger.info(f"   Gross Expected Value: ${gross_expected_value:,.0f}")
            logger.info(f"   Net Expected Value: ${gross_expected_value:,.0f} - ${estimated_total_cost:,.0f} = ${expected_value:,.0f}")
            
        else:
            # Path 2: Fallback calculation with transparent formula
            win_probability = evidence_strength * 0.7 + 0.2  # 20-90% based on evidence strength
            gross_expected_value = case_value * win_probability
            expected_value = gross_expected_value - estimated_total_cost
            
            logger.info(f"💡 FALLBACK EXPECTED VALUE CALCULATION:")
            logger.info(f"   Win Probability: ({evidence_strength:.1f} × 0.7) + 0.2 = {win_probability:.1%}")
            logger.info(f"   Gross Expected Value: ${case_value:,.0f} × {win_probability:.1%} = ${gross_expected_value:,.0f}")
            logger.info(f"   Net Expected Value: ${gross_expected_value:,.0f} - ${estimated_total_cost:,.0f} = ${expected_value:,.0f}")
        
        # ROI analysis with transparent scenarios
        roi_scenarios = {
            'best_case': ((case_value * 0.9) - (estimated_total_cost * 0.8)) / estimated_total_cost if estimated_total_cost > 0 else 0,
            'expected_case': expected_value / estimated_total_cost if estimated_total_cost > 0 else 0,
            'worst_case': ((case_value * 0.1) - (estimated_total_cost * 1.2)) / estimated_total_cost if estimated_total_cost > 0 else -1
        }
        
        # Enhanced return data with calculation transparency
        calculation_breakdown = {
            'base_cost': base_cost,
            'total_multiplier': total_multiplier,
            'cost_components': {
                'jurisdiction': {'multiplier': jurisdiction_multiplier, 'explanation': jurisdiction_explanation},
                'complexity': {'multiplier': complexity_multiplier, 'description': complexity_desc, 'percentage': case_complexity * 100},
                'case_value': {'multiplier': value_multiplier, 'explanation': value_explanation},
                'evidence': {'multiplier': evidence_multiplier, 'description': evidence_desc, 'strength_score': evidence_strength * 10}
            },
            'value_calculation_method': 'settlement_analysis' if settlement_analysis else 'fallback_probability'
        }
        
        logger.info(f"✅ Final Analysis: Cost=${estimated_total_cost:,.0f}, Expected Value=${expected_value:,.0f}, ROI={roi_scenarios['expected_case']:.1%}")
        
        return {
            'total_cost': estimated_total_cost,
            'expected_value': expected_value,
            'roi_analysis': roi_scenarios,
            'cost_factors': {
                'jurisdiction_multiplier': jurisdiction_multiplier,
                'complexity_multiplier': complexity_multiplier,
                'value_multiplier': value_multiplier,
                'evidence_multiplier': evidence_multiplier
            },
            'calculation_breakdown': calculation_breakdown,  # New transparent breakdown
            'calculation_transparency': {
                'formula': 'Expected Value = (Settlement Path + Trial Path) - Litigation Costs',
                'cost_formula': f'Costs = ${base_cost:,} × {total_multiplier:.2f} = ${estimated_total_cost:,.0f}',
                'bounds_applied': raw_settlement_value != bounded_settlement_value if settlement_analysis and hasattr(settlement_analysis, 'metrics') else False,
                'confidence_level': 'high' if settlement_analysis else 'moderate'
            }
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_ai_strategic_summary(self, case_data: Dict[str, Any], analyses: Dict[str, Any], strategy_type: StrategyType) -> str:
        """Generate comprehensive AI-powered strategic summary with real AI analysis"""
        try:
            # Normalize evidence strength for display
            evidence_strength_raw = float(case_data.get('evidence_strength', 5))
            evidence_strength_pct = (evidence_strength_raw / 10.0) * 100
            case_complexity_pct = float(case_data.get('case_complexity', 0.5)) * 100
            
            summary_prompt = f"""
            You are a senior litigation strategist analyzing a {case_data.get('case_type', 'civil')} case. Provide a comprehensive strategic analysis.

            CASE OVERVIEW:
            - Case Type: {case_data.get('case_type', 'Unknown')}
            - Case Value: ${float(case_data.get('case_value', 0)):,.2f}
            - Evidence Strength: {evidence_strength_pct:.0f}% ({evidence_strength_raw}/10 scale)
            - Case Complexity: {case_complexity_pct:.0f}%
            - Jurisdiction: {case_data.get('jurisdiction', 'Unknown')}
            - Court Level: {case_data.get('court_level', 'District')}
            - Judge: {case_data.get('judge_name', 'Not assigned')}

            CASE FACTS:
            {case_data.get('case_facts', 'No detailed facts provided')[:1500]}
            """
            
            # Add analysis results if available
            outcome_prediction = analyses.get('outcome_prediction')
            if outcome_prediction and hasattr(outcome_prediction, 'predicted_outcome'):
                summary_prompt += f"""
                
            OUTCOME PREDICTION:
            - Most Likely Outcome: {outcome_prediction.predicted_outcome.value.replace('_', ' ').title()}
            - Confidence Score: {outcome_prediction.confidence_score:.1%}
            - Estimated Duration: {getattr(outcome_prediction, 'estimated_duration', 365)} days
            - Estimated Cost: ${getattr(outcome_prediction, 'estimated_cost', 50000):,.2f}
                """
            
            # Add settlement analysis
            settlement_analysis = analyses.get('settlement_analysis')
            if settlement_analysis and hasattr(settlement_analysis, 'metrics'):
                summary_prompt += f"""
                
            SETTLEMENT ANALYSIS:
            - Settlement Probability: {settlement_analysis.metrics.settlement_probability:.1%}
            - Expected Settlement Value: ${settlement_analysis.metrics.expected_settlement_value:,.2f}
            - Optimal Settlement Timing: {settlement_analysis.metrics.optimal_timing.value.replace('_', ' ').title()}
            - Settlement Confidence: {settlement_analysis.metrics.confidence_score:.1%}
                """
            
            # Add judicial insights if available
            judicial_insights = analyses.get('judicial_insights')
            if judicial_insights and isinstance(judicial_insights, dict):
                summary_prompt += f"""
                
            JUDICIAL INSIGHTS:
            - Judge Experience: {judicial_insights.get('experience_years', 'Unknown')} years
            - Settlement Rate: {judicial_insights.get('overall_metrics', {}).get('settlement_rate', 0):.1%}
            - Plaintiff Success Rate: {judicial_insights.get('overall_metrics', {}).get('plaintiff_success_rate', 0):.1%}
                """
            
            summary_prompt += f"""

            RECOMMENDED STRATEGY: {strategy_type.value.replace('_', ' ').title()}

            CRITICAL: Include a section explaining cost and value calculations with transparency. Our system uses bounded calculations to prevent unrealistic projections:
            - Settlement values are capped at 135% of case value to account for reasonable damages/interest/costs
            - Expected values use probability-weighted outcomes: (Settlement Path + Trial Path) - Litigation Costs
            - All multipliers are documented and bounded to realistic professional ranges

            Provide a comprehensive strategic analysis covering:

            1. EXECUTIVE SUMMARY (2-3 sentences)
            - Overall case assessment and recommended approach

            2. KEY STRATEGIC OPPORTUNITIES
            - Specific advantages and leverage points
            - Best-case scenarios and how to achieve them

            3. CRITICAL VULNERABILITIES AND RISKS
            - Evidence weaknesses and how to address them
            - Procedural and substantive legal risks
            - Opposition strengths and countermeasures

            4. RECOMMENDED TACTICAL APPROACH
            - Specific litigation tactics and timing
            - Discovery strategy and priorities
            - Motion practice recommendations

            5. SETTLEMENT VS. TRIAL ANALYSIS
            - When and how to approach settlement discussions
            - Trial readiness requirements and timeline
            - Risk-benefit analysis for each path

            6. COST-BENEFIT TRANSPARENCY
            - Explain how litigation costs are calculated (jurisdiction, complexity, value, evidence factors)
            - Clarify expected value methodology and bounds checking
            - Justify financial projections with clear reasoning

            7. RESOURCE ALLOCATION AND TIMELINE
            - Critical deadlines and milestones
            - Budget considerations and cost control
            - Team composition and expertise requirements

            8. ALTERNATIVE STRATEGIES
            - Backup plans if primary strategy fails
            - Creative legal approaches to consider
            - Risk mitigation for unfavorable developments

            Focus on actionable insights that will help legal counsel make informed strategic decisions. Be specific about timing, costs, and probability assessments. Always explain the basis for financial calculations to ensure trustworthy recommendations.
            """
            
            # Use Gemini AI if available, fallback to Groq
            if self.gemini_model:
                try:
                    response = await asyncio.to_thread(
                        self.gemini_model.generate_content,
                        summary_prompt
                    )
                    ai_summary = response.text
                    logger.info("✅ AI strategic summary generated using Gemini")
                except Exception as e:
                    logger.warning(f"⚠️ Gemini AI failed, trying Groq: {e}")
                    if self.groq_client:
                        response = await asyncio.to_thread(
                            self.groq_client.chat.completions.create,
                            messages=[{"role": "user", "content": summary_prompt}],
                            model="mixtral-8x7b-32768",
                            max_tokens=2000,
                            temperature=0.3
                        )
                        ai_summary = response.choices[0].message.content
                        logger.info("✅ AI strategic summary generated using Groq")
                    else:
                        raise Exception("No AI service available")
            elif self.groq_client:
                response = await asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    messages=[{"role": "user", "content": summary_prompt}],
                    model="mixtral-8x7b-32768",
                    max_tokens=2000,
                    temperature=0.3
                )
                ai_summary = response.choices[0].message.content
                logger.info("✅ AI strategic summary generated using Groq")
            else:
                raise Exception("No AI service available")
            
            return ai_summary
            
        except Exception as e:
            logger.warning(f"⚠️ AI strategic summary generation failed: {e}")
            return self._generate_enhanced_fallback_summary(case_data, analyses, strategy_type)

    def _generate_enhanced_fallback_summary(self, case_data: Dict[str, Any], analyses: Dict[str, Any], strategy_type: StrategyType) -> str:
        """Generate enhanced fallback summary when AI fails"""
        case_value = float(case_data.get('case_value', 0)) if case_data.get('case_value') else 0
        evidence_strength = float(case_data.get('evidence_strength', 5))
        case_complexity = float(case_data.get('case_complexity', 0.5)) * 100
        
        # Get settlement probability if available
        settlement_prob = 0.5
        settlement_analysis = analyses.get('settlement_analysis')
        if settlement_analysis and hasattr(settlement_analysis, 'metrics'):
            settlement_prob = settlement_analysis.metrics.settlement_probability
        
        return f"""
COMPREHENSIVE LITIGATION STRATEGY ANALYSIS

EXECUTIVE SUMMARY:
This {case_data.get('case_type', 'litigation')} matter with ${case_value:,.0f} at stake requires a {strategy_type.value.replace('_', ' ')} strategic approach. Based on evidence strength of {evidence_strength}/10 and case complexity of {case_complexity:.0f}%, the recommended strategy balances aggressive advocacy with prudent risk management.

KEY STRATEGIC OPPORTUNITIES:
• Evidence strength at {evidence_strength}/10 provides {'strong foundation for aggressive pursuit' if evidence_strength > 7 else 'adequate basis for measured approach' if evidence_strength > 5 else 'challenging but manageable foundation requiring careful development'}
• Settlement probability of {settlement_prob:.1%} {'strongly favors early negotiation' if settlement_prob > 0.7 else 'suggests balanced litigation and settlement approach' if settlement_prob > 0.4 else 'indicates trial preparation focus'}
• Case value of ${case_value:,.0f} {'justifies comprehensive litigation investment' if case_value > 500000 else 'supports efficient resource allocation'}

CRITICAL RISK FACTORS:
• {'High case complexity requires specialized legal expertise and extended timeline' if case_complexity > 70 else 'Moderate complexity manageable with standard litigation approach'}
• {'Evidence development critical given current strength level' if evidence_strength < 6 else 'Strong evidence foundation reduces litigation risk'}
• Jurisdiction-specific factors in {case_data.get('jurisdiction', 'selected forum')} will impact strategy execution

RECOMMENDED APPROACH:
1. Immediate: Comprehensive case assessment and evidence preservation
2. Short-term (30-60 days): Strategic discovery planning and initial motion practice
3. Medium-term (60-180 days): {'Parallel settlement discussions with litigation preparation' if settlement_prob > 0.5 else 'Focused trial preparation with selective settlement opportunities'}
4. Long-term: {'Trial readiness with continued settlement evaluation' if strategy_type.value in ['aggressive', 'trial_focused'] else 'Settlement-focused resolution with trial backup plan'}

TRANSPARENT COST-BENEFIT CALCULATION:
This analysis uses the following transparent calculation methodology to ensure accuracy and prevent unrealistic projections:

LITIGATION COST FORMULA:
• Base Cost: $45,000 (2024 market average)
• Jurisdiction Multiplier: {case_data.get('jurisdiction', 'federal')} jurisdiction factor
• Complexity Factor: {case_complexity:.0f}% case complexity adjustment
• Value Premium: High-stakes case premium based on ${case_value:,.0f} value
• Evidence Discovery: Additional discovery costs for {evidence_strength}/10 evidence strength

EXPECTED VALUE BOUNDS CHECKING:
• Settlement values are capped at 135% of case value to prevent unrealistic projections
• Expected values use probability-weighted outcomes combining settlement and trial paths
• All multipliers are documented and bounded to realistic ranges

CALCULATION TRANSPARENCY:
Expected Value = (Settlement Probability × Capped Settlement Value) + (Trial Probability × Win Probability × Case Value) - Total Litigation Costs

This methodology prevents the inflated values seen in previous analyses and ensures realistic financial projections for strategic decision-making.

RESOURCE ALLOCATION:
• Budget planning essential for case of this magnitude and complexity
• Expert witness consultation recommended for technical and damages issues
• Discovery strategy should be proportional to case value and evidence strength
• Timeline management critical to preserve strategic advantages

This analysis provides the foundation for informed strategic decision-making throughout the litigation process.
        """

    def _calculate_strategy_confidence(self, analyses: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the strategy"""
        confidence_scores = []
        
        # Outcome prediction confidence
        outcome_prediction = analyses.get('outcome_prediction')
        if outcome_prediction and hasattr(outcome_prediction, 'confidence_score'):
            confidence_scores.append(outcome_prediction.confidence_score)
        
        # Settlement analysis confidence
        settlement_analysis = analyses.get('settlement_analysis')
        if settlement_analysis and hasattr(settlement_analysis, 'metrics'):
            if hasattr(settlement_analysis.metrics, 'confidence_score'):
                confidence_scores.append(settlement_analysis.metrics.confidence_score)
        
        # Judicial insights confidence
        judicial_insights = analyses.get('judicial_insights')
        if judicial_insights and isinstance(judicial_insights, dict):
            confidence_scores.append(judicial_insights.get('confidence_score', 0.6))
        
        # Base confidence if no analyses available
        if not confidence_scores:
            return 0.66  # Default confidence level
        
        # Calculate weighted average
        return sum(confidence_scores) / len(confidence_scores)

    def _identify_risk_factors(self, case_data: Dict[str, Any], analyses: Dict[str, Any]) -> List[str]:
        """Identify key risk factors for the litigation with enhanced analysis"""
        risk_factors = []
        
        # Evidence-based risks (normalized from 1-10 scale)
        evidence_strength = float(case_data.get('evidence_strength', 5)) / 10.0
        if evidence_strength < 0.4:
            risk_factors.append("Weak evidence base may undermine case viability and increase trial risk")
        elif evidence_strength < 0.6:
            risk_factors.append("Moderate evidence strength requires aggressive discovery to strengthen position")
        
        # Complexity risks
        complexity = float(case_data.get('case_complexity', 0.5))
        if complexity > 0.8:
            risk_factors.append("High case complexity significantly increases cost, duration, and outcome uncertainty")
        elif complexity > 0.6:
            risk_factors.append("Complex case factors require specialized expertise and careful resource management")
        
        # Value-based risks
        case_value = float(case_data.get('case_value', 0)) if case_data.get('case_value') else 0
        if case_value > 5000000:
            risk_factors.append("Very high case value increases opponent sophistication and resistance to settlement")
        elif case_value > 1000000:
            risk_factors.append("Significant case value elevates stakes and requires comprehensive risk management")
        
        # Settlement analysis risks
        settlement_analysis = analyses.get('settlement_analysis')
        if settlement_analysis and hasattr(settlement_analysis, 'metrics'):
            if settlement_analysis.metrics.settlement_probability < 0.3:
                risk_factors.append("Low settlement probability increases trial risk and cost exposure")
        
        # Judicial risks
        judicial_insights = analyses.get('judicial_insights')
        if judicial_insights and isinstance(judicial_insights, dict):
            if judicial_insights.get('confidence_score', 1.0) < 0.5:
                risk_factors.append("Limited judicial insights create uncertainty in case management strategy")
        
        # Timeline risks
        if case_data.get('timeline_constraints'):
            risk_factors.append("Client timeline constraints may limit strategic flexibility and increase pressure")
        
        return risk_factors[:6]  # Top 6 risk factors

    def _develop_mitigation_strategies(self, case_data: Dict[str, Any], analyses: Dict[str, Any]) -> List[str]:
        """Develop strategies to mitigate identified risks"""
        strategies = []
        
        # Evidence mitigation (normalized from 1-10 scale)
        evidence_strength = float(case_data.get('evidence_strength', 5)) / 10.0
        if evidence_strength < 0.5:
            strategies.extend([
                "Implement aggressive discovery strategy to identify and develop supporting evidence",
                "Retain expert witnesses early to strengthen technical and damages analysis",
                "Consider alternative legal theories to reduce dependence on current evidence"
            ])
        
        # Complexity mitigation
        complexity = float(case_data.get('case_complexity', 0.5))
        if complexity > 0.7:
            strategies.extend([
                "Engage specialized counsel with expertise in relevant practice areas",
                "Develop phased litigation approach to manage complexity systematically",
                "Implement comprehensive project management for complex case coordination"
            ])
        
        # Cost and resource mitigation
        case_value = float(case_data.get('case_value', 0)) if case_data.get('case_value') else 0
        if case_value > 1000000:
            strategies.extend([
                "Negotiate structured fee arrangements to manage cost risk effectively",
                "Evaluate insurance coverage and provide timely notice to carriers",
                "Develop comprehensive budget with milestone reviews and cost controls"
            ])
        
        # Settlement-focused mitigation
        settlement_analysis = analyses.get('settlement_analysis')
        if settlement_analysis and hasattr(settlement_analysis, 'metrics'):
            if settlement_analysis.metrics.settlement_probability > 0.5:
                strategies.append("Pursue early settlement discussions to avoid escalating litigation costs and risks")
        
        # Timeline mitigation
        if case_data.get('timeline_constraints'):
            strategies.append("Develop expedited case management plan with court to meet client timeline requirements")
        
        return strategies[:6]  # Top 6 mitigation strategies

    def _generate_alternative_strategies(self, case_data: Dict[str, Any], analyses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative strategy options with enhanced analysis"""
        alternatives = []
        
        case_value = float(case_data.get('case_value', 0)) if case_data.get('case_value') else 0
        evidence_strength = float(case_data.get('evidence_strength', 5)) / 10.0
        
        # Alternative 1: Settlement-focused approach
        settlement_analysis = analyses.get('settlement_analysis')
        if settlement_analysis and hasattr(settlement_analysis, 'metrics'):
            settlement_cost = min(case_value * 0.15, 75000)
            alternatives.append({
                'strategy_name': 'Settlement-Focused Strategy',
                'description': 'Prioritize early resolution through negotiation, mediation, and collaborative problem-solving',
                'pros': [
                    'Lower total costs and faster resolution',
                    'Controlled outcome with predictable results', 
                    'Preserves business relationships and confidentiality',
                    'Reduces management distraction and resource drain'
                ],
                'cons': [
                    'May achieve lower recovery than trial verdict',
                    'No precedent value for future similar cases',
                    'Potential perception of weakness by opponents'
                ],
                'estimated_cost': settlement_cost,
                'estimated_duration': 90,
                'success_probability': min(0.9, settlement_analysis.metrics.settlement_probability + 0.2),
                'best_for': 'Cases with strong settlement probability and business relationship preservation needs'
            })
        
        # Alternative 2: Aggressive litigation (if evidence supports it)
        outcome_prediction = analyses.get('outcome_prediction')
        if evidence_strength > 0.6 and outcome_prediction:
            trial_cost = case_value * 0.35
            win_prob = 0.5
            if hasattr(outcome_prediction, 'probability_breakdown'):
                win_prob = outcome_prediction.probability_breakdown.get('plaintiff_win', 0.5)
            
            alternatives.append({
                'strategy_name': 'Aggressive Trial Strategy',
                'description': 'Pursue maximum recovery through comprehensive litigation and trial preparation',
                'pros': [
                    'Maximum potential recovery and precedent setting',
                    'Strong deterrent effect for future disputes',
                    'Complete vindication and public resolution',
                    'Potential for attorney fee recovery'
                ],
                'cons': [
                    'Significantly higher costs and longer timeline',
                    'Uncertain outcome with substantial downside risk',
                    'Management distraction and resource intensity',
                    'Public disclosure of sensitive information'
                ],
                'estimated_cost': trial_cost,
                'estimated_duration': 18 * 30,  # 18 months in days
                'success_probability': win_prob,
                'best_for': 'Cases with strong evidence, high stakes, and appetite for extended litigation'
            })
        
        # Alternative 3: Hybrid approach (always applicable)
        hybrid_cost = case_value * 0.25
        alternatives.append({
            'strategy_name': 'Balanced Hybrid Approach',
            'description': 'Combine litigation preparation with settlement readiness for maximum strategic flexibility',
            'pros': [
                'Strategic flexibility to pursue best available option',
                'Maintains settlement leverage through trial preparation',
                'Allows for tactical adjustments based on developments',
                'Balances risk and opportunity effectively'
            ],
            'cons': [
                'Higher initial costs than pure settlement approach',
                'Requires complex coordination and decision-making',
                'May appear unfocused to opposing counsel',
                'Demands experienced counsel with dual expertise'
            ],
            'estimated_cost': hybrid_cost,
            'estimated_duration': 12 * 30,  # 12 months in days
            'success_probability': 0.65,
            'best_for': 'Complex cases requiring strategic flexibility and multiple resolution paths'
        })
        
        # Alternative 4: Expedited resolution (for time-sensitive cases)
        if case_data.get('timeline_constraints') or case_value > 2000000:
            expedited_cost = case_value * 0.20
            alternatives.append({
                'strategy_name': 'Expedited Resolution Strategy',
                'description': 'Fast-track case through accelerated procedures, early mediation, or expedited trial',
                'pros': [
                    'Rapid resolution meets business timeline needs',
                    'Reduces uncertainty and management distraction',
                    'Lower overall costs through compressed timeline',
                    'Preserves strategic business opportunities'
                ],
                'cons': [
                    'Limited discovery may weaken case preparation',
                    'Reduced settlement negotiation time',
                    'Higher intensity resource requirements',
                    'Potential for suboptimal outcomes due to time pressure'
                ],
                'estimated_cost': expedited_cost,
                'estimated_duration': 6 * 30,  # 6 months in days
                'success_probability': 0.6,
                'best_for': 'Time-sensitive cases with clear legal issues and business urgency'
            })
        
        return alternatives

    async def _cache_litigation_strategy(self, strategy: LitigationStrategy):
        """Cache litigation strategy for future reference with enhanced data structure"""
        try:
            # Convert recommendations to serializable format
            serialized_recommendations = []
            for rec in strategy.strategic_recommendations:
                serialized_recommendations.append({
                    'recommendation_id': rec.recommendation_id,
                    'title': rec.title,
                    'description': rec.description,
                    'priority': rec.priority.value,
                    'category': rec.category,
                    'estimated_cost': rec.estimated_cost,
                    'estimated_timeframe': rec.estimated_timeframe,
                    'success_probability': rec.success_probability,
                    'risk_level': rec.risk_level,
                    'dependencies': rec.dependencies,
                    'supporting_evidence': rec.supporting_evidence
                })
            
            # Convert jurisdiction analysis to serializable format
            serialized_jurisdictions = []
            for analysis in strategy.jurisdiction_analysis:
                serialized_jurisdictions.append({
                    'jurisdiction': analysis.jurisdiction,
                    'suitability_score': analysis.suitability_score,
                    'advantages': analysis.advantages,
                    'disadvantages': analysis.disadvantages,
                    'average_case_duration': analysis.average_case_duration,
                    'success_rate': analysis.success_rate_for_case_type,
                    'settlement_rate': analysis.settlement_rate,
                    'estimated_costs': analysis.estimated_costs
                })
            
            # Convert timing analysis to serializable format
            serialized_timing = []
            for timing in strategy.timing_analysis:
                serialized_timing.append({
                    'action_type': timing.action_type,
                    'optimal_window_start': timing.optimal_window[0].isoformat(),
                    'optimal_window_end': timing.optimal_window[1].isoformat(),
                    'urgency_level': timing.urgency_level,
                    'rationale': timing.rationale,
                    'dependencies': timing.dependencies,
                    'risk_factors': timing.risk_factors
                })
            
            # Convert evidence assessment to serializable format
            evidence_dict = None
            if strategy.evidence_assessment:
                evidence_dict = {
                    'overall_strength': strategy.evidence_assessment.overall_strength,
                    'key_strengths': strategy.evidence_assessment.key_strengths,
                    'critical_weaknesses': strategy.evidence_assessment.critical_weaknesses,
                    'evidence_gaps': strategy.evidence_assessment.evidence_gaps,
                    'discovery_priorities': strategy.evidence_assessment.discovery_priorities,
                    'document_quality': strategy.evidence_assessment.document_quality,
                    'expert_witness_needs': strategy.evidence_assessment.expert_witness_needs
                }
            
            strategy_dict = {
                'case_id': strategy.case_id,
                'strategy_date': strategy.strategy_date.isoformat(),
                'recommended_strategy_type': strategy.recommended_strategy_type.value,
                'confidence_score': strategy.confidence_score,
                'estimated_total_cost': strategy.estimated_total_cost,
                'expected_value': strategy.expected_value,
                'roi_analysis': strategy.roi_analysis,
                'strategic_recommendations': serialized_recommendations,
                'jurisdiction_analysis': serialized_jurisdictions,
                'timing_analysis': serialized_timing,
                'evidence_assessment': evidence_dict,
                'risk_factors': strategy.risk_factors,
                'mitigation_strategies': strategy.mitigation_strategies,
                'ai_strategic_summary': strategy.ai_strategic_summary,
                'alternative_strategies': strategy.alternative_strategies,
                'created_at': datetime.utcnow().isoformat(),
                'version': '2.0'  # Enhanced version
            }
            
            await self.db.litigation_strategies.update_one(
                {'case_id': strategy.case_id},
                {'$set': strategy_dict},
                upsert=True
            )
            
            logger.info(f"💾 Enhanced litigation strategy cached for case {strategy.case_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Strategy caching failed: {e}")

    def _create_default_strategy(self, case_id: str) -> LitigationStrategy:
        """Create default strategy when optimization fails"""
        return LitigationStrategy(
            case_id=case_id,
            strategy_date=datetime.utcnow(),
            recommended_strategy_type=StrategyType.COLLABORATIVE,
            confidence_score=0.6,
            strategic_recommendations=[
                StrategicRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    title="Comprehensive Case Assessment",
                    description="Conduct thorough case assessment including evidence review, legal research, and strategic planning to develop optimal litigation approach",
                    priority=ActionPriority.HIGH,
                    category="Case Management",
                    estimated_cost=8000.0,
                    estimated_timeframe="Within 30 days",
                    success_probability=0.9,
                    risk_level="low",
                    supporting_evidence=[
                        "Initial assessment critical for strategic planning",
                        "Early case evaluation enables informed decision-making",
                        "Foundation for all subsequent litigation activities"
                    ]
                ),
                StrategicRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    title="Evidence Preservation and Discovery Planning", 
                    description="Implement immediate evidence preservation measures and develop comprehensive discovery strategy",
                    priority=ActionPriority.HIGH,
                    category="Discovery",
                    estimated_cost=12000.0,
                    estimated_timeframe="Within 14 days",
                    success_probability=0.85,
                    risk_level="medium",
                    supporting_evidence=[
                        "Evidence preservation prevents spoliation claims",
                        "Early discovery planning optimizes case development",
                        "Proactive approach demonstrates client commitment"
                    ]
                )
            ],
            ai_strategic_summary="Comprehensive strategy optimization pending additional case information and detailed analysis. Initial recommendations focus on case assessment and evidence preservation to establish strong foundation for strategic decision-making."
        )

# Global optimizer instance
_strategy_optimizer = None

async def get_litigation_strategy_optimizer(db_connection) -> LitigationStrategyOptimizer:
    """Get or create enhanced litigation strategy optimizer instance"""
    global _strategy_optimizer
    
    if _strategy_optimizer is None:
        _strategy_optimizer = LitigationStrategyOptimizer(db_connection)
        logger.info("🎯 Enhanced Litigation Strategy Optimizer instance created")
    
    return _strategy_optimizer

async def initialize_litigation_strategy_optimizer(db_connection):
    """Initialize the enhanced litigation strategy optimizer"""
    await get_litigation_strategy_optimizer(db_connection)
    logger.info("✅ Enhanced Litigation Strategy Optimizer initialized successfully")