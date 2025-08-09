"""
AI-Powered Litigation Analytics Engine

This module provides comprehensive litigation analytics including case outcome prediction,
judicial behavior analysis, settlement probability calculations, and strategic litigation
recommendations for LegalMate AI.
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
import google.generativeai as genai
from groq import Groq
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class CaseType(Enum):
    CIVIL = "civil"
    CRIMINAL = "criminal"
    COMMERCIAL = "commercial"
    EMPLOYMENT = "employment"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    FAMILY = "family"
    PERSONAL_INJURY = "personal_injury"
    BANKRUPTCY = "bankruptcy"
    TAX = "tax"
    ENVIRONMENTAL = "environmental"

class CaseOutcome(Enum):
    PLAINTIFF_WIN = "plaintiff_win"
    DEFENDANT_WIN = "defendant_win"
    SETTLEMENT = "settlement"
    DISMISSED = "dismissed"
    APPEAL_PENDING = "appeal_pending"
    ONGOING = "ongoing"

class JudgeType(Enum):
    DISTRICT = "district"
    CIRCUIT = "circuit"
    SUPREME = "supreme"
    STATE = "state"
    MAGISTRATE = "magistrate"

@dataclass
class CaseData:
    """Structure for case data used in predictions"""
    case_id: str
    case_type: CaseType
    jurisdiction: str
    court_level: str
    judge_name: Optional[str] = None
    case_facts: Optional[str] = None
    legal_issues: List[str] = field(default_factory=list)
    case_complexity: Optional[float] = None
    case_value: Optional[float] = None
    filing_date: Optional[datetime] = None
    case_status: Optional[str] = None
    similar_cases: List[str] = field(default_factory=list)
    evidence_strength: Optional[float] = None
    witness_count: Optional[int] = None
    settlement_offers: List[float] = field(default_factory=list)

@dataclass
class AppealAnalysis:
    """Structure for appeal probability analysis"""
    appeal_probability: float
    appeal_confidence: float
    appeal_factors: List[str] = field(default_factory=list)
    appeal_timeline: Optional[int] = None  # days to file appeal
    appeal_cost_estimate: Optional[float] = None
    appeal_success_probability: Optional[float] = None
    preventive_measures: List[str] = field(default_factory=list)
    jurisdictional_appeal_rate: Optional[float] = None

@dataclass
class PredictionResult:
    """Structure for case outcome predictions"""
    case_id: str
    predicted_outcome: CaseOutcome
    confidence_score: float
    probability_breakdown: Dict[str, float]
    estimated_duration: Optional[int] = None  # days
    estimated_cost: Optional[float] = None
    settlement_probability: Optional[float] = None
    settlement_range: Optional[Tuple[float, float]] = None
    risk_factors: List[str] = field(default_factory=list)
    success_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    appeal_analysis: Optional[AppealAnalysis] = None  # NEW: Appeal probability analysis
    prediction_date: datetime = field(default_factory=datetime.utcnow)

@dataclass
class JudgeInsights:
    """Structure for judicial behavior analysis"""
    judge_name: str
    court: str
    judge_type: JudgeType
    total_cases: int
    outcome_patterns: Dict[str, float]
    average_case_duration: float
    settlement_rate: float
    appeal_rate: float
    specialty_areas: List[str]
    decision_tendencies: Dict[str, Any]
    recent_trends: Dict[str, Any]
    confidence_score: float

class LitigationAnalyticsEngine:
    """Main litigation analytics engine combining multiple AI models for comprehensive analysis"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        self.groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
        
        # Initialize CourtListener client
        self.courtlistener_client = httpx.AsyncClient(
            base_url="https://www.courtlistener.com/api/rest/v3",
            headers={"Authorization": f"Token {os.environ.get('COURTLISTENER_API_KEY')}"}
        )
        
        # Performance metrics
        self.prediction_accuracy = 0.0
        self.total_predictions = 0
        
        logger.info("✅ Litigation Analytics Engine initialized")

    async def initialize_collections(self):
        """Initialize MongoDB collections for litigation analytics"""
        try:
            # Create indexes for optimal performance
            await self.db.litigation_cases.create_index([("case_id", 1)], unique=True)
            await self.db.litigation_cases.create_index([("case_type", 1), ("jurisdiction", 1)])
            await self.db.litigation_cases.create_index([("judge_name", 1)])
            await self.db.litigation_cases.create_index([("filing_date", -1)])
            
            await self.db.judicial_decisions.create_index([("judge_name", 1)], unique=True)
            await self.db.judicial_decisions.create_index([("court", 1)])
            
            await self.db.settlement_data.create_index([("case_type", 1), ("case_value", 1)])
            await self.db.settlement_data.create_index([("settlement_date", -1)])
            
            await self.db.prediction_models.create_index([("model_type", 1)])
            await self.db.prediction_models.create_index([("created_at", -1)])
            
            await self.db.litigation_analytics.create_index([("case_id", 1)])
            await self.db.litigation_analytics.create_index([("created_at", -1)])
            
            logger.info("✅ Litigation analytics collections initialized with indexes")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize collections: {e}")
            raise

    async def analyze_case_outcome(self, case_data: CaseData) -> PredictionResult:
        """Comprehensive case outcome analysis using ensemble AI approach"""
        try:
            logger.info(f"🔍 Analyzing case outcome for case {case_data.case_id}")
            
            # Get similar historical cases
            similar_cases = await self._find_similar_cases(case_data)
            
            # Get judicial insights if judge is known
            judge_insights = None
            if case_data.judge_name:
                judge_insights = await self._get_judge_insights(case_data.judge_name)
            
            # Prepare AI analysis prompt
            analysis_prompt = self._build_case_analysis_prompt(case_data, similar_cases, judge_insights)
            
            # Run parallel AI analysis
            gemini_analysis, groq_analysis = await asyncio.gather(
                self._get_gemini_analysis(analysis_prompt),
                self._get_groq_analysis(analysis_prompt)
            )
            
            # Combine and validate results
            prediction_result = await self._ensemble_prediction(
                case_data, gemini_analysis, groq_analysis, similar_cases, judge_insights
            )
            
            # Store prediction for future accuracy tracking
            await self._store_prediction(prediction_result)
            
            logger.info(f"✅ Case analysis completed with {prediction_result.confidence_score:.2%} confidence")
            return prediction_result
            
        except Exception as e:
            logger.error(f"❌ Case outcome analysis failed: {e}")
            raise

    async def _find_similar_cases(self, case_data: CaseData, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar historical cases for pattern analysis"""
        try:
            # Build query for similar cases
            query = {
                "case_type": case_data.case_type.value,
                "jurisdiction": case_data.jurisdiction,
                "outcome": {"$in": ["plaintiff_win", "defendant_win", "settlement"]}
            }
            
            # Add case value range if available
            if case_data.case_value:
                value_range = case_data.case_value * 0.5  # 50% range
                query["case_value"] = {
                    "$gte": case_data.case_value - value_range,
                    "$lte": case_data.case_value + value_range
                }
            
            # Find similar cases
            similar_cases = await self.db.litigation_cases.find(query).limit(limit).to_list(limit)
            
            logger.info(f"📊 Found {len(similar_cases)} similar cases for analysis")
            return similar_cases
            
        except Exception as e:
            logger.warning(f"⚠️ Similar case search failed: {e}")
            return []

    def _build_case_analysis_prompt(self, case_data: CaseData, similar_cases: List[Dict], judge_insights: Optional[JudgeInsights]) -> str:
        """Build comprehensive AI analysis prompt"""
        prompt = f"""
        LITIGATION ANALYTICS - CASE OUTCOME PREDICTION
        
        CASE DETAILS:
        - Case Type: {case_data.case_type.value}
        - Jurisdiction: {case_data.jurisdiction}
        - Court Level: {case_data.court_level}
        - Judge: {case_data.judge_name or 'Not specified'}
        - Case Value: {f'${case_data.case_value:,.2f}' if case_data.case_value else 'Not specified'}
        - Legal Issues: {', '.join(case_data.legal_issues) if case_data.legal_issues else 'Not specified'}
        - Case Facts: {case_data.case_facts or 'Not provided'}
        - Evidence Strength: {case_data.evidence_strength or 'Not rated'}/10
        - Witness Count: {case_data.witness_count or 'Not specified'}
        
        HISTORICAL PATTERNS:
        Found {len(similar_cases)} similar cases with outcomes:
        """
        
        # Add similar case outcomes
        if similar_cases:
            outcomes = {}
            for case in similar_cases:
                outcome = case.get('outcome', 'unknown')
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
            
            for outcome, count in outcomes.items():
                percentage = (count / len(similar_cases)) * 100
                prompt += f"- {outcome.replace('_', ' ').title()}: {count} cases ({percentage:.1f}%)\n"
        
        # Add judge insights if available
        if judge_insights:
            prompt += f"""
        JUDICIAL INSIGHTS:
        - Judge {judge_insights.judge_name} has presided over {judge_insights.total_cases} cases
        - Settlement Rate: {judge_insights.settlement_rate:.1%}
        - Appeal Rate: {judge_insights.appeal_rate:.1%}
        - Average Case Duration: {judge_insights.average_case_duration:.0f} days
        - Specialty Areas: {', '.join(judge_insights.specialty_areas)}
        """
        
        prompt += """
        
        ANALYSIS REQUIRED:
        1. Predict the most likely outcome (plaintiff_win, defendant_win, settlement, dismissed)
        2. Provide confidence score (0.0-1.0)
        3. Break down probability for each possible outcome
        4. Estimate case duration in days
        5. Estimate total litigation costs
        6. Calculate settlement probability and range
        7. Identify key risk factors
        8. Identify success factors
        9. Provide strategic recommendations
        
        Respond with detailed analysis considering legal precedents, judicial tendencies, case complexity, and statistical patterns from similar cases.
        """
        
        return prompt

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _get_gemini_analysis(self, prompt: str) -> Dict[str, Any]:
        """Get case analysis from Gemini AI"""
        try:
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt + "\n\nProvide response as structured analysis with clear sections for each required element."
            )
            
            return {
                "source": "gemini",
                "analysis": response.text,
                "confidence": 0.85  # Base confidence for Gemini
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Gemini analysis failed: {e}")
            return {"source": "gemini", "analysis": "", "confidence": 0.0}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _get_groq_analysis(self, prompt: str) -> Dict[str, Any]:
        """Get case analysis from Groq AI"""
        try:
            response = await asyncio.to_thread(
                self.groq_client.chat.completions.create,
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": prompt + "\n\nProvide detailed legal analysis with specific predictions and numerical assessments."}],
                temperature=0.1
            )
            
            return {
                "source": "groq",
                "analysis": response.choices[0].message.content,
                "confidence": 0.80  # Base confidence for Groq
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Groq analysis failed: {e}")
            return {"source": "groq", "analysis": "", "confidence": 0.0}

    async def _ensemble_prediction(self, case_data: CaseData, gemini_analysis: Dict, groq_analysis: Dict, 
                                 similar_cases: List[Dict], judge_insights: Optional[JudgeInsights]) -> PredictionResult:
        """Combine AI analyses into final ensemble prediction"""
        try:
            # Extract predictions from AI analyses
            gemini_prediction = self._parse_ai_analysis(gemini_analysis["analysis"], "gemini")
            groq_prediction = self._parse_ai_analysis(groq_analysis["analysis"], "groq")
            
            # Calculate historical baseline from similar cases
            historical_baseline = self._calculate_historical_baseline(similar_cases)
            
            # Combine predictions using weighted ensemble
            ensemble_weights = {
                "gemini": 0.35,
                "groq": 0.35,
                "historical": 0.30
            }
            
            # Combine outcome probabilities
            combined_probabilities = {}
            for outcome in ["plaintiff_win", "defendant_win", "settlement", "dismissed"]:
                combined_prob = (
                    gemini_prediction.get("probabilities", {}).get(outcome, 0.25) * ensemble_weights["gemini"] +
                    groq_prediction.get("probabilities", {}).get(outcome, 0.25) * ensemble_weights["groq"] +
                    historical_baseline.get("probabilities", {}).get(outcome, 0.25) * ensemble_weights["historical"]
                )
                combined_probabilities[outcome] = combined_prob
            
            # Determine most likely outcome
            predicted_outcome = max(combined_probabilities, key=combined_probabilities.get)
            confidence_score = combined_probabilities[predicted_outcome]
            
            # Estimate duration and cost
            estimated_duration = self._estimate_case_duration(case_data, similar_cases, judge_insights)
            estimated_cost = self._estimate_litigation_cost(case_data, estimated_duration)
            
            # Calculate settlement metrics
            settlement_probability = combined_probabilities.get("settlement", 0.0)
            settlement_range = self._estimate_settlement_range(case_data, similar_cases)
            
            # Combine risk and success factors
            risk_factors = list(set(
                gemini_prediction.get("risk_factors", []) + 
                groq_prediction.get("risk_factors", [])
            ))[:5]  # Top 5
            
            success_factors = list(set(
                gemini_prediction.get("success_factors", []) + 
                groq_prediction.get("success_factors", [])
            ))[:5]  # Top 5
            
            # Combine recommendations
            recommendations = list(set(
                gemini_prediction.get("recommendations", []) + 
                groq_prediction.get("recommendations", [])
            ))[:7]  # Top 7
            
            # Generate appeal analysis
            appeal_analysis = await self._predict_appeal_probability(
                case_data, predicted_outcome, confidence_score, similar_cases, judge_insights
            )
            
            return PredictionResult(
                case_id=case_data.case_id,
                predicted_outcome=CaseOutcome(predicted_outcome),
                confidence_score=confidence_score,
                probability_breakdown=combined_probabilities,
                estimated_duration=estimated_duration,
                estimated_cost=estimated_cost,
                settlement_probability=settlement_probability,
                settlement_range=settlement_range,
                risk_factors=risk_factors,
                success_factors=success_factors,
                recommendations=recommendations,
                appeal_analysis=appeal_analysis,
                prediction_date=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"❌ Ensemble prediction failed: {e}")
            # Return default prediction with basic appeal analysis
            default_appeal = AppealAnalysis(
                appeal_probability=0.3,  # Default 30% appeal probability
                appeal_confidence=0.5,
                appeal_factors=["Insufficient data for detailed appeal analysis"],
                appeal_timeline=30,
                appeal_cost_estimate=25000,
                appeal_success_probability=0.4,
                preventive_measures=["Ensure thorough trial record preparation"],
                jurisdictional_appeal_rate=0.25
            )
            
            return PredictionResult(
                case_id=case_data.case_id,
                predicted_outcome=CaseOutcome.ONGOING,
                confidence_score=0.5,
                probability_breakdown={"plaintiff_win": 0.25, "defendant_win": 0.25, "settlement": 0.25, "dismissed": 0.25},
                appeal_analysis=default_appeal
            )

    async def _predict_appeal_probability(self, case_data: CaseData, predicted_outcome: str,
                                         confidence_score: float, similar_cases: List[Dict],
                                         judge_insights: Optional[JudgeInsights]) -> AppealAnalysis:
        """
        Comprehensive appeal probability analysis using ensemble AI approach
        
        Factors considered:
        - Case outcome (losses more likely to be appealed)
        - Case value (high-value cases more likely appealed)
        - Evidence strength (weak evidence = higher appeal probability)
        - Jurisdiction appeal rates
        - Judge-specific appeal patterns
        - Case complexity
        - Legal precedent strength
        """
        try:
            logger.info(f"📋 Analyzing appeal probability for case {case_data.case_id}")
            
            # Base appeal probability calculation
            base_appeal_prob = 0.15  # Base 15% appeal rate
            
            # Factor 1: Case Outcome Impact (most important factor)
            outcome_multiplier = self._get_outcome_appeal_multiplier(predicted_outcome)
            
            # Factor 2: Case Value Impact
            value_multiplier = self._get_value_appeal_multiplier(case_data.case_value)
            
            # Factor 3: Evidence Strength Impact (inverse relationship)
            evidence_multiplier = self._get_evidence_appeal_multiplier(case_data.evidence_strength)
            
            # Factor 4: Jurisdictional Appeal Rate
            jurisdiction_multiplier = self._get_jurisdiction_appeal_multiplier(case_data.jurisdiction)
            
            # Factor 5: Case Complexity Impact
            complexity_multiplier = self._get_complexity_appeal_multiplier(case_data.case_complexity)
            
            # Factor 6: Judge-Specific Appeal Pattern
            judge_multiplier = self._get_judge_appeal_multiplier(judge_insights)
            
            # Factor 7: Confidence Score Impact (low confidence = higher appeal probability, reduced impact)
            confidence_multiplier = 1.0 + (0.3 * (1.0 - confidence_score))  # Reduced from 0.5 to 0.3
            
            # Calculate combined appeal probability
            appeal_probability = base_appeal_prob * outcome_multiplier * value_multiplier * \
                               evidence_multiplier * jurisdiction_multiplier * complexity_multiplier * \
                               judge_multiplier * confidence_multiplier
            
            # Cap probability between 5% and 95%
            appeal_probability = max(0.05, min(0.95, appeal_probability))
            
            # Generate AI-enhanced analysis
            appeal_analysis_prompt = await self._build_appeal_analysis_prompt(
                case_data, predicted_outcome, appeal_probability, similar_cases
            )
            
            # Get AI insights for appeal factors and preventive measures
            ai_appeal_analysis = await self._get_ai_appeal_analysis(appeal_analysis_prompt)
            
            # Calculate appeal success probability
            appeal_success_prob = await self._calculate_appeal_success_probability(
                case_data, predicted_outcome, appeal_probability
            )
            
            # Estimate appeal costs and timeline
            appeal_cost = self._estimate_appeal_costs(case_data.case_value, case_data.jurisdiction)
            appeal_timeline = self._estimate_appeal_timeline(case_data.jurisdiction, case_data.court_level)
            
            # Get jurisdictional appeal rate for comparison
            jurisdictional_rate = self._get_jurisdictional_appeal_statistics(case_data.jurisdiction)
            
            # Combine all analysis
            return AppealAnalysis(
                appeal_probability=appeal_probability,
                appeal_confidence=0.85,  # High confidence in appeal prediction
                appeal_factors=ai_appeal_analysis.get("appeal_factors", 
                    self._generate_default_appeal_factors(case_data, predicted_outcome, appeal_probability)
                )[:5],
                appeal_timeline=appeal_timeline,
                appeal_cost_estimate=appeal_cost,
                appeal_success_probability=appeal_success_prob,
                preventive_measures=ai_appeal_analysis.get("preventive_measures", 
                    self._generate_default_preventive_measures(case_data, predicted_outcome)
                )[:5],
                jurisdictional_appeal_rate=jurisdictional_rate
            )
            
        except Exception as e:
            logger.error(f"❌ Appeal probability analysis failed: {e}")
            # Return conservative default analysis
            return AppealAnalysis(
                appeal_probability=0.25,
                appeal_confidence=0.6,
                appeal_factors=["Standard appeal risk factors"],
                appeal_timeline=30,
                appeal_cost_estimate=50000,
                appeal_success_probability=0.35,
                preventive_measures=["Prepare thorough trial record"],
                jurisdictional_appeal_rate=0.2
            )

    def _get_outcome_appeal_multiplier(self, predicted_outcome: str) -> float:
        """Get appeal probability multiplier based on predicted outcome"""
        multipliers = {
            "plaintiff_win": 1.8,    # Defendants more likely to appeal losses (reduced from 2.8)
            "defendant_win": 0.7,    # Plaintiffs less likely to appeal (cost considerations)
            "settlement": 0.2,       # Very low appeal probability for settlements
            "dismissed": 1.3         # Moderate appeal probability for dismissals (reduced from 1.5)
        }
        return multipliers.get(predicted_outcome, 1.0)

    def _get_value_appeal_multiplier(self, case_value: Optional[float]) -> float:
        """Get appeal probability multiplier based on case value"""
        if not case_value:
            return 1.0
        
        if case_value >= 10000000:    # $10M+
            return 2.2  # Reduced from 3.2
        elif case_value >= 5000000:   # $5M-$10M  
            return 1.8  # Reduced from 2.5
        elif case_value >= 1000000:   # $1M-$5M
            return 1.5  # Reduced from 1.8
        elif case_value >= 500000:    # $500K-$1M
            return 1.2  # Reduced from 1.3
        elif case_value >= 100000:    # $100K-$500K
            return 1.0
        else:                         # Under $100K
            return 0.7  # Slightly reduced from 0.6

    def _get_evidence_appeal_multiplier(self, evidence_strength: Optional[float]) -> float:
        """Get appeal probability multiplier based on evidence strength (inverse relationship)"""
        if not evidence_strength:
            return 1.1  # Reduced from 1.2
        
        # Normalize to 0-1 scale if needed
        if evidence_strength > 1:
            evidence_strength = evidence_strength / 10.0
        
        # Inverse relationship: weak evidence = higher appeal probability (reduced impact)
        return 1.0 + (0.5 * (1.0 - evidence_strength))  # Reduced from 0.8 to 0.5

    def _get_jurisdiction_appeal_multiplier(self, jurisdiction: str) -> float:
        """Get appeal probability multiplier based on jurisdiction appeal patterns"""
        multipliers = {
            "federal": 1.15,     # Reduced from 1.2
            "california": 1.05,  # Reduced from 1.1
            "new_york": 1.0,     # Baseline appeal rate
            "texas": 0.95,       # Reduced from 0.9
            "delaware": 1.2,     # Reduced from 1.3
            "florida": 0.97,     # Slightly reduced from 0.95
            "illinois": 1.02     # Reduced from 1.05
        }
        return multipliers.get(jurisdiction.lower(), 1.0)

    def _get_complexity_appeal_multiplier(self, case_complexity: Optional[float]) -> float:
        """Get appeal probability multiplier based on case complexity"""
        if not case_complexity:
            return 1.0
        
        # Higher complexity = higher appeal probability (reduced impact)
        return 1.0 + (case_complexity * 0.4)  # Reduced from 0.6 to 0.4

    def _get_judge_appeal_multiplier(self, judge_insights: Optional[JudgeInsights]) -> float:
        """Get appeal probability multiplier based on judge's appeal rate history"""
        if not judge_insights:
            return 1.0
        
        # Use judge's historical appeal rate if available
        if hasattr(judge_insights, 'appeal_rate') and judge_insights.appeal_rate:
            # Normalize to multiplier (baseline 20% appeal rate)
            return judge_insights.appeal_rate / 0.20
        
        return 1.0

    async def _build_appeal_analysis_prompt(self, case_data: CaseData, predicted_outcome: str,
                                          appeal_probability: float, similar_cases: List[Dict]) -> str:
        """Build comprehensive prompt for AI appeal analysis"""
        prompt = f"""
        APPEAL PROBABILITY ANALYSIS - LEGAL CASE ASSESSMENT
        
        CASE DETAILS:
        - Case Type: {case_data.case_type.value}
        - Predicted Outcome: {predicted_outcome.replace('_', ' ').title()}
        - Calculated Appeal Probability: {appeal_probability:.1%}
        - Case Value: {f'${case_data.case_value:,.2f}' if case_data.case_value else 'Not specified'}
        - Evidence Strength: {case_data.evidence_strength or 'Not rated'}/10
        - Case Complexity: {case_data.case_complexity*100:.0f}% if case_data.case_complexity else 'Not specified'
        - Jurisdiction: {case_data.jurisdiction}
        - Court Level: {case_data.court_level}
        
        HISTORICAL CONTEXT:
        Found {len(similar_cases)} similar cases for pattern analysis.
        
        ANALYSIS REQUIRED:
        1. Identify specific appeal risk factors for this case
        2. Recommend preventive measures to reduce appeal probability
        3. Assess potential grounds for appeal based on case characteristics
        4. Suggest trial strategies to minimize appellate issues
        
        Provide detailed analysis focusing on:
        - Legal issues most likely to be challenged on appeal
        - Procedural safeguards to implement during trial
        - Settlement considerations given appeal risk
        - Post-trial motion strategies
        
        Respond with structured analysis including specific, actionable recommendations.
        """
        return prompt

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _get_ai_appeal_analysis(self, prompt: str) -> Dict[str, Any]:
        """Get AI-powered appeal analysis from ensemble approach"""
        try:
            logger.info("🤖 Starting AI appeal analysis...")
            
            # Use both Gemini and Groq for comprehensive analysis
            gemini_response = None
            groq_response = None
            
            # Try Gemini first
            try:
                if hasattr(self, 'gemini_model') and self.gemini_model:
                    logger.info("📊 Calling Gemini for appeal analysis...")
                    gemini_response = await asyncio.to_thread(
                        self.gemini_model.generate_content,
                        prompt + "\n\nProvide specific appeal factors and preventive measures in clear bullet points."
                    )
                    logger.info("✅ Gemini appeal analysis completed")
            except Exception as gemini_error:
                logger.warning(f"⚠️ Gemini appeal analysis failed: {gemini_error}")
            
            # Try Groq second
            try:
                if hasattr(self, 'groq_client') and self.groq_client:
                    logger.info("🚀 Calling Groq for appeal analysis...")
                    groq_response = await asyncio.to_thread(
                        self.groq_client.chat.completions.create,
                        model="mixtral-8x7b-32768",
                        messages=[{"role": "user", "content": prompt + "\n\nFocus on practical appellate risk assessment and mitigation strategies. Provide clear bullet points."}],
                        temperature=0.1,
                        max_tokens=1500
                    )
                    logger.info("✅ Groq appeal analysis completed")
            except Exception as groq_error:
                logger.warning(f"⚠️ Groq appeal analysis failed: {groq_error}")
            
            # If we have at least one response, process it
            if gemini_response or groq_response:
                gemini_text = gemini_response.text if gemini_response else ""
                groq_text = groq_response.choices[0].message.content if groq_response else ""
                
                # Parse and combine responses
                appeal_factors = self._extract_appeal_factors(gemini_text, groq_text)
                preventive_measures = self._extract_preventive_measures(gemini_text, groq_text)
                
                logger.info(f"✅ Successfully extracted {len(appeal_factors)} appeal factors and {len(preventive_measures)} preventive measures")
                
                return {
                    "appeal_factors": appeal_factors,
                    "preventive_measures": preventive_measures
                }
            else:
                logger.warning("⚠️ Both AI services failed, using enhanced default analysis")
                raise Exception("Both Gemini and Groq services unavailable")
            
        except Exception as e:
            logger.warning(f"⚠️ AI appeal analysis failed completely: {e}")
            return {
                "appeal_factors": ["AI analysis unavailable - using enhanced default factors"],
                "preventive_measures": ["AI analysis unavailable - using enhanced default measures"]
            }

    def _extract_appeal_factors(self, gemini_text: str, groq_text: str) -> List[str]:
        """Extract appeal risk factors from AI responses"""
        factors = []
        
        # Extract from both responses using pattern matching
        import re
        
        # Look for appeal-related factors
        factor_patterns = [
            r'appeal.*?factor[s]?[:\-]\s*([^\n\.]+)',
            r'risk.*?factor[s]?[:\-]\s*([^\n\.]+)',
            r'ground[s]?.*?appeal[:\-]\s*([^\n\.]+)',
            r'likely.*?challenge[d]?[:\-]\s*([^\n\.]+)'
        ]
        
        combined_text = f"{gemini_text} {groq_text}".lower()
        
        for pattern in factor_patterns:
            matches = re.findall(pattern, combined_text)
            for match in matches:
                factor = match.strip().capitalize()
                if len(factor) > 10 and factor not in factors:
                    factors.append(factor)
        
        # Default factors if none extracted
        if not factors:
            factors = [
                "Evidence sufficiency challenges",
                "Legal standard interpretation disputes", 
                "Procedural error claims",
                "Damages calculation disputes",
                "Jury instruction challenges"
            ]
        
        return factors[:5]  # Top 5 factors

    def _extract_preventive_measures(self, gemini_text: str, groq_text: str) -> List[str]:
        """Extract preventive measures from AI responses"""
        measures = []
        
        import re
        
        # Look for preventive/mitigation measures
        measure_patterns = [
            r'prevent[ive]?.*?measure[s]?[:\-]\s*([^\n\.]+)',
            r'mitigat[e|ion].*?[:\-]\s*([^\n\.]+)',
            r'recommend[ation]?[s]?[:\-]\s*([^\n\.]+)',
            r'strategy.*?[:\-]\s*([^\n\.]+)',
            r'safeguard[s]?[:\-]\s*([^\n\.]+)'
        ]
        
        combined_text = f"{gemini_text} {groq_text}".lower()
        
        for pattern in measure_patterns:
            matches = re.findall(pattern, combined_text)
            for match in matches:
                measure = match.strip().capitalize()
                if len(measure) > 15 and measure not in measures:
                    measures.append(measure)
        
        # Default measures if none extracted
        if not measures:
            measures = [
                "Maintain detailed trial record with comprehensive objections",
                "Prepare thorough post-trial motions addressing key legal issues", 
                "Ensure proper preservation of all appellate issues during trial",
                "Consider settlement negotiations factoring appeal risk",
                "Retain experienced appellate counsel for consultation"
            ]
        
        return measures[:5]  # Top 5 measures

    async def _calculate_appeal_success_probability(self, case_data: CaseData,
                                                 predicted_outcome: str, appeal_probability: float) -> float:
        """Calculate probability of success if case is appealed"""
        base_success_rate = 0.35  # Historical average appeal success rate ~35%
        
        # Adjust based on factors
        success_multiplier = 1.0
        
        # Evidence strength impact (weak evidence = higher appeal success)
        if case_data.evidence_strength:
            evidence_score = case_data.evidence_strength / 10.0 if case_data.evidence_strength > 1 else case_data.evidence_strength
            success_multiplier *= 1.0 + (0.4 * (1.0 - evidence_score))
        
        # Case complexity (more complex = more appeal grounds)
        if case_data.case_complexity:
            success_multiplier *= 1.0 + (case_data.case_complexity * 0.3)
        
        # Outcome impact (losses more likely to succeed on appeal)
        if predicted_outcome == "defendant_win":
            success_multiplier *= 1.2  # Plaintiffs slightly more successful on appeal
        elif predicted_outcome == "plaintiff_win":  
            success_multiplier *= 0.9  # Defendants slightly less successful
        
        # Jurisdiction impact
        jurisdiction_success_rates = {
            "federal": 0.38,     # Federal appellate courts
            "california": 0.33,  # California appellate courts
            "new_york": 0.35,    # New York appellate courts
            "texas": 0.32,       # Texas appellate courts
            "delaware": 0.40,    # Delaware Supreme Court
        }
        
        jurisdiction_base = jurisdiction_success_rates.get(case_data.jurisdiction.lower(), base_success_rate)
        
        # Calculate final success probability
        appeal_success_prob = jurisdiction_base * success_multiplier
        
        # Cap between 10% and 70%
        return max(0.10, min(0.70, appeal_success_prob))

    def _estimate_appeal_costs(self, case_value: Optional[float], jurisdiction: str) -> float:
        """Estimate costs of appellate proceedings"""
        base_appeal_cost = 75000  # Base appellate cost
        
        # Adjust based on case value
        if case_value:
            if case_value >= 10000000:
                value_multiplier = 2.5
            elif case_value >= 5000000:
                value_multiplier = 2.0
            elif case_value >= 1000000:
                value_multiplier = 1.5
            else:
                value_multiplier = 1.0
        else:
            value_multiplier = 1.0
        
        # Jurisdiction cost adjustments
        jurisdiction_multipliers = {
            "federal": 1.3,      # Higher federal appellate costs
            "california": 1.4,   # High California costs
            "new_york": 1.5,     # Highest costs in NY
            "delaware": 1.2,     # Moderate Delaware costs
            "texas": 0.9,        # Lower Texas costs
        }
        
        jurisdiction_multiplier = jurisdiction_multipliers.get(jurisdiction.lower(), 1.0)
        
        return base_appeal_cost * value_multiplier * jurisdiction_multiplier

    def _estimate_appeal_timeline(self, jurisdiction: str, court_level: str) -> int:
        """Estimate timeline to file appeal (days from judgment)"""
        # Standard appeal deadlines
        base_timeline = 30  # Default 30 days
        
        # Jurisdiction-specific deadlines
        deadlines = {
            "federal": 30,       # Federal Rule of Civil Procedure
            "california": 60,    # California Rules of Court
            "new_york": 30,      # New York Civil Practice Law
            "texas": 30,         # Texas Rules of Civil Procedure
            "delaware": 30,      # Delaware Court Rules
            "florida": 30,       # Florida Rules of Civil Procedure
        }
        
        # Court level adjustments
        if court_level.lower() == "supreme":
            return deadlines.get(jurisdiction.lower(), base_timeline) + 60  # Additional time for supreme court appeals
        
        return deadlines.get(jurisdiction.lower(), base_timeline)

    def _get_jurisdictional_appeal_statistics(self, jurisdiction: str) -> float:
        """Get historical appeal rates for jurisdiction"""
        appeal_rates = {
            "federal": 0.18,     # Federal court appeal rate
            "california": 0.16,  # California appeal rate
            "new_york": 0.15,    # New York appeal rate
            "texas": 0.12,       # Texas appeal rate
            "delaware": 0.22,    # Delaware appeal rate (business cases)
            "florida": 0.14,     # Florida appeal rate
            "illinois": 0.17     # Illinois appeal rate
        }
        return appeal_rates.get(jurisdiction.lower(), 0.15)
    
    def _generate_default_appeal_factors(self, case_data: CaseData, predicted_outcome: str, appeal_probability: float) -> List[str]:
        """Generate contextual appeal factors when AI analysis is unavailable"""
        factors = []
        
        # Add outcome-based factor
        if predicted_outcome == "plaintiff_win":
            factors.append("Defendant likely to appeal unfavorable judgment")
        elif predicted_outcome == "defendant_win":
            factors.append("Plaintiff may appeal adverse ruling if significant damages at stake")
        elif predicted_outcome == "dismissed":
            factors.append("Plaintiff may appeal dismissal on procedural or legal grounds")
        else:
            factors.append(f"Case outcome: {predicted_outcome.replace('_', ' ').title()}")
        
        # Add value-based factor
        if case_data.case_value:
            if case_data.case_value >= 1000000:
                factors.append(f"High case value (${case_data.case_value:,.0f}) increases appeal likelihood")
            elif case_data.case_value >= 500000:
                factors.append(f"Significant case value (${case_data.case_value:,.0f}) warrants appeal consideration")
            else:
                factors.append(f"Case value (${case_data.case_value:,.0f}) may limit appeal attractiveness")
        
        # Add evidence-based factor
        if case_data.evidence_strength:
            evidence_score = case_data.evidence_strength / 10.0 if case_data.evidence_strength > 1 else case_data.evidence_strength
            if evidence_score < 0.4:
                factors.append("Weak evidence strength creates potential appeal grounds")
            elif evidence_score < 0.7:
                factors.append("Moderate evidence strength allows for appeal consideration")
            else:
                factors.append("Strong evidence may discourage appeals")
        
        # Add complexity factor
        if case_data.case_complexity and case_data.case_complexity > 0.6:
            factors.append("High case complexity creates multiple potential appeal issues")
        
        # Add jurisdictional factor
        jurisdiction_name = case_data.jurisdiction.replace('_', ' ').title()
        factors.append(f"{jurisdiction_name} jurisdiction has {self._get_jurisdictional_appeal_statistics(case_data.jurisdiction)*100:.0f}% appeal rate")
        
        return factors[:5]
    
    def _generate_default_preventive_measures(self, case_data: CaseData, predicted_outcome: str) -> List[str]:
        """Generate contextual preventive measures when AI analysis is unavailable"""
        measures = [
            "Maintain comprehensive trial record with detailed objections and rulings",
            "Prepare thorough post-trial motions to address potential appeal grounds"
        ]
        
        # Add outcome-specific measures
        if predicted_outcome == "plaintiff_win":
            measures.append("Ensure judgment is well-supported by evidence and legal precedent")
        elif predicted_outcome == "defendant_win":
            measures.append("Document strong legal basis for defense verdict")
        elif predicted_outcome == "dismissed":
            measures.append("Ensure dismissal follows proper procedural requirements")
        
        # Add case-specific measures
        if case_data.case_value and case_data.case_value >= 1000000:
            measures.append("Consider settlement negotiations to avoid costly appeal process")
        
        if case_data.case_complexity and case_data.case_complexity > 0.6:
            measures.append("Retain experienced appellate counsel for complex legal issues")
        
        measures.append("Document all legal arguments and supporting authorities thoroughly")
        
        return measures[:5]

    def _parse_ai_analysis(self, analysis_text: str, source: str) -> Dict[str, Any]:
        """Parse AI analysis text into structured data"""
        # This is a simplified parser - in production, you'd want more sophisticated NLP
        parsed = {
            "probabilities": {},
            "risk_factors": [],
            "success_factors": [],
            "recommendations": []
        }
        
        try:
            # Extract probabilities (looking for percentage patterns)
            import re
            
            # Look for outcome probabilities
            outcome_patterns = {
                "plaintiff_win": r"plaintiff.*?(\d+\.?\d*)%",
                "defendant_win": r"defendant.*?(\d+\.?\d*)%", 
                "settlement": r"settlement.*?(\d+\.?\d*)%",
                "dismissed": r"dismiss.*?(\d+\.?\d*)%"
            }
            
            for outcome, pattern in outcome_patterns.items():
                match = re.search(pattern, analysis_text.lower())
                if match:
                    parsed["probabilities"][outcome] = float(match.group(1)) / 100.0
            
            # If no specific probabilities found, use default distribution
            if not parsed["probabilities"]:
                parsed["probabilities"] = {
                    "plaintiff_win": 0.30,
                    "defendant_win": 0.30, 
                    "settlement": 0.30,
                    "dismissed": 0.10
                }
            
            # Extract risk factors (look for bullet points or numbered lists)
            risk_matches = re.findall(r'risk.*?[:\-]\s*([^\n\.]+)', analysis_text.lower())
            parsed["risk_factors"] = [risk.strip() for risk in risk_matches[:5]]
            
            # Extract success factors
            success_matches = re.findall(r'success.*?[:\-]\s*([^\n\.]+)', analysis_text.lower())
            parsed["success_factors"] = [success.strip() for success in success_matches[:5]]
            
            # Extract recommendations
            rec_matches = re.findall(r'recommend.*?[:\-]\s*([^\n\.]+)', analysis_text.lower())
            parsed["recommendations"] = [rec.strip() for rec in rec_matches[:7]]
            
            return parsed
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse {source} analysis: {e}")
            return parsed

    def _calculate_historical_baseline(self, similar_cases: List[Dict]) -> Dict[str, Any]:
        """Calculate historical baseline from similar cases"""
        if not similar_cases:
            return {
                "probabilities": {"plaintiff_win": 0.25, "defendant_win": 0.25, "settlement": 0.25, "dismissed": 0.25}
            }
        
        # Count outcomes
        outcomes = {}
        for case in similar_cases:
            outcome = case.get("outcome", "unknown")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        # Calculate probabilities
        total = len(similar_cases)
        probabilities = {}
        for outcome in ["plaintiff_win", "defendant_win", "settlement", "dismissed"]:
            probabilities[outcome] = outcomes.get(outcome, 0) / total
        
        return {"probabilities": probabilities}

    def _estimate_case_duration(self, case_data: CaseData, similar_cases: List[Dict], judge_insights: Optional[JudgeInsights]) -> int:
        """Estimate case duration in days"""
        base_duration = 365  # 1 year default
        
        # Adjust based on case type
        type_multipliers = {
            CaseType.COMMERCIAL: 1.5,
            CaseType.INTELLECTUAL_PROPERTY: 1.3,
            CaseType.PERSONAL_INJURY: 1.2,
            CaseType.EMPLOYMENT: 0.8,
            CaseType.FAMILY: 0.6
        }
        
        duration = base_duration * type_multipliers.get(case_data.case_type, 1.0)
        
        # Adjust based on case complexity
        if case_data.case_complexity:
            duration *= (1 + case_data.case_complexity)
        
        # Adjust based on judge insights
        if judge_insights:
            duration = judge_insights.average_case_duration
        
        # Adjust based on similar cases
        if similar_cases:
            avg_duration = sum(case.get("duration", 365) for case in similar_cases) / len(similar_cases)
            duration = (duration + avg_duration) / 2
        
        return int(duration)

    def _estimate_litigation_cost(self, case_data: CaseData, duration_days: int) -> float:
        """Estimate total litigation costs"""
        base_cost_per_day = 1000  # $1000 per day base rate
        
        # Adjust based on case value
        if case_data.case_value:
            # Cost typically 10-20% of case value
            value_based_cost = case_data.case_value * 0.15
            duration_based_cost = duration_days * base_cost_per_day
            
            # Use higher of the two estimates
            return max(value_based_cost, duration_based_cost)
        
        # Default to duration-based estimate
        return duration_days * base_cost_per_day

    def _estimate_settlement_range(self, case_data: CaseData, similar_cases: List[Dict]) -> Optional[Tuple[float, float]]:
        """Estimate potential settlement range"""
        if not case_data.case_value:
            return None
        
        # Base range: 40-80% of case value
        low_estimate = case_data.case_value * 0.40
        high_estimate = case_data.case_value * 0.80
        
        # Adjust based on similar cases
        if similar_cases:
            settlements = [case.get("settlement_amount") for case in similar_cases if case.get("settlement_amount")]
            if settlements:
                avg_settlement = sum(settlements) / len(settlements)
                # Weight towards historical data
                low_estimate = (low_estimate + avg_settlement * 0.7) / 2
                high_estimate = (high_estimate + avg_settlement * 1.3) / 2
        
        return (low_estimate, high_estimate)

    async def _get_judge_insights(self, judge_name: str) -> Optional[JudgeInsights]:
        """Get cached judicial insights or generate new ones"""
        try:
            # Check cache first
            cached_insights = await self.db.judicial_decisions.find_one({"judge_name": judge_name})
            
            if cached_insights and cached_insights.get("last_updated", datetime.min) > (datetime.utcnow() - timedelta(days=30)):
                return JudgeInsights(**cached_insights["insights"])
            
            # Generate new insights (placeholder for now)
            logger.info(f"📊 Generating judicial insights for Judge {judge_name}")
            
            # This would involve analyzing historical cases for this judge
            # For now, return None to indicate no specific insights available
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get judge insights: {e}")
            return None

    async def _store_prediction(self, prediction: PredictionResult):
        """Store prediction for accuracy tracking and caching"""
        try:
            prediction_doc = {
                "case_id": prediction.case_id,
                "predicted_outcome": prediction.predicted_outcome.value,
                "confidence_score": prediction.confidence_score,
                "probability_breakdown": prediction.probability_breakdown,
                "estimated_duration": prediction.estimated_duration,
                "estimated_cost": prediction.estimated_cost,
                "settlement_probability": prediction.settlement_probability,
                "settlement_range": prediction.settlement_range,
                "risk_factors": prediction.risk_factors,
                "success_factors": prediction.success_factors,
                "recommendations": prediction.recommendations,
                "prediction_date": prediction.prediction_date,
                "actual_outcome": None,  # To be filled when case concludes
                "accuracy_verified": False
            }
            
            await self.db.litigation_analytics.insert_one(prediction_doc)
            logger.info(f"✅ Prediction stored for case {prediction.case_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to store prediction: {e}")

    async def get_prediction_accuracy(self) -> Dict[str, Any]:
        """Get current model accuracy metrics"""
        try:
            total_predictions = await self.db.litigation_analytics.count_documents({})
            verified_predictions = await self.db.litigation_analytics.count_documents({"accuracy_verified": True})
            
            if verified_predictions == 0:
                return {
                    "accuracy_rate": 0.0,
                    "total_predictions": total_predictions,
                    "verified_predictions": 0,
                    "confidence": "building_baseline"
                }
            
            # Calculate accuracy from verified predictions
            correct_predictions = await self.db.litigation_analytics.count_documents({
                "accuracy_verified": True,
                "$expr": {"$eq": ["$predicted_outcome", "$actual_outcome"]}
            })
            
            accuracy_rate = correct_predictions / verified_predictions
            
            return {
                "accuracy_rate": accuracy_rate,
                "total_predictions": total_predictions,
                "verified_predictions": verified_predictions,
                "confidence": "high" if accuracy_rate > 0.75 else "medium" if accuracy_rate > 0.60 else "low"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate accuracy: {e}")
            return {"accuracy_rate": 0.0, "total_predictions": 0, "verified_predictions": 0, "confidence": "error"}

# Global engine instance
_litigation_engine = None

async def get_litigation_engine(db_connection) -> LitigationAnalyticsEngine:
    """Get or create litigation analytics engine instance"""
    global _litigation_engine
    
    if _litigation_engine is None:
        _litigation_engine = LitigationAnalyticsEngine(db_connection)
        await _litigation_engine.initialize_collections()
        logger.info("🚀 Litigation Analytics Engine instance created")
    
    return _litigation_engine

async def initialize_litigation_engine(db_connection):
    """Initialize the litigation analytics engine"""
    await get_litigation_engine(db_connection)
    logger.info("✅ Litigation Analytics Engine initialized successfully")