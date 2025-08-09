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
            # Return default prediction
            return PredictionResult(
                case_id=case_data.case_id,
                predicted_outcome=CaseOutcome.ONGOING,
                confidence_score=0.5,
                probability_breakdown={"plaintiff_win": 0.25, "defendant_win": 0.25, "settlement": 0.25, "dismissed": 0.25}
            )

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