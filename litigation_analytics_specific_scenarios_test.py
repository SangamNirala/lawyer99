#!/usr/bin/env python3
"""
Additional Specific Test Scenarios for Litigation Analytics

Testing specific scenarios mentioned in the review request:
- Commercial litigation case with high value ($2M+) 
- Employment law dispute with medium complexity
- Intellectual property case with strong evidence
- Personal injury case with settlement potential
- MongoDB collections verification
- AI model integration verification
"""

import asyncio
import aiohttp
import json
import os
import sys
import time
from datetime import datetime

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://legalengine.preview.emergentagent.com')
API_BASE = f"{BACKEND_URL}/api"

class SpecificScenariosTestSuite:
    """Test suite for specific scenarios mentioned in review request"""
    
    def __init__(self):
        self.session = None
        self.test_results = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log_result(self, test_name: str, success: bool, details: str, data: dict = None):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {details}")
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details,
            "data": data
        })
    
    async def test_commercial_litigation_high_value(self):
        """Test commercial litigation case with high value ($2M+)"""
        print("\n💼 Testing Commercial Litigation Case with High Value ($2M+)...")
        
        case_data = {
            "case_type": "commercial",
            "jurisdiction": "federal",
            "court_level": "district",
            "judge_name": "Judge Patricia Williams",
            "case_facts": "Major commercial dispute involving breach of multi-million dollar supply agreement. Plaintiff alleges defendant failed to deliver critical components causing production delays and lost revenue. Seeking damages for breach of contract, lost profits, and consequential damages.",
            "legal_issues": [
                "breach_of_contract",
                "lost_profits",
                "consequential_damages",
                "force_majeure_defense",
                "mitigation_of_damages"
            ],
            "case_complexity": 0.85,
            "case_value": 3500000.0,  # $3.5M - above $2M threshold
            "evidence_strength": 7.8,
            "witness_count": 18,
            "settlement_offers": [800000.0, 1200000.0, 1500000.0]
        }
        
        try:
            async with self.session.post(f"{API_BASE}/litigation/analyze-case", json=case_data) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Validate high-value case specific requirements
                    validation_passed = True
                    validation_details = []
                    
                    # Check confidence scores are realistic (0.0-1.0)
                    if not (0.0 <= data["confidence_score"] <= 1.0):
                        validation_passed = False
                        validation_details.append(f"Invalid confidence score: {data['confidence_score']}")
                    
                    # Check probability breakdown sums to reasonable range
                    prob_sum = sum(data["probability_breakdown"].values())
                    if not (0.8 <= prob_sum <= 1.2):  # Allow some tolerance
                        validation_passed = False
                        validation_details.append(f"Probability breakdown sum: {prob_sum}")
                    
                    # Check estimated cost is substantial for high-value case
                    if data["estimated_cost"] and data["estimated_cost"] < 100000:
                        validation_passed = False
                        validation_details.append(f"Low estimated cost for high-value case: ${data['estimated_cost']}")
                    
                    # Check settlement range is provided for high-value case
                    if not data.get("settlement_range"):
                        validation_passed = False
                        validation_details.append("Missing settlement range for high-value case")
                    
                    if validation_passed:
                        self.log_result(
                            "Commercial High-Value Case Analysis",
                            True,
                            f"Case value: ${case_data['case_value']:,.0f}, Predicted: {data['predicted_outcome']}, Confidence: {data['confidence_score']:.2%}, Cost: ${data['estimated_cost']:,.0f}",
                            data
                        )
                    else:
                        self.log_result(
                            "Commercial High-Value Case Analysis",
                            False,
                            f"Validation failed: {'; '.join(validation_details)}"
                        )
                else:
                    error_text = await response.text()
                    self.log_result(
                        "Commercial High-Value Case Analysis",
                        False,
                        f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_result(
                "Commercial High-Value Case Analysis",
                False,
                f"Exception: {str(e)}"
            )
    
    async def test_employment_law_medium_complexity(self):
        """Test employment law dispute with medium complexity"""
        print("\n👥 Testing Employment Law Dispute with Medium Complexity...")
        
        case_data = {
            "case_type": "employment",
            "jurisdiction": "california",
            "court_level": "state",
            "judge_name": "Judge Maria Rodriguez",
            "case_facts": "Employment discrimination and wrongful termination case. Employee alleges discrimination based on age and gender, hostile work environment, and retaliatory termination after filing internal complaints. Seeking reinstatement, back pay, front pay, and punitive damages.",
            "legal_issues": [
                "age_discrimination",
                "gender_discrimination", 
                "hostile_work_environment",
                "wrongful_termination",
                "retaliation"
            ],
            "case_complexity": 0.65,  # Medium complexity
            "case_value": 275000.0,
            "evidence_strength": 6.5,
            "witness_count": 12,
            "filing_date": "2024-02-01"
        }
        
        try:
            async with self.session.post(f"{API_BASE}/litigation/analyze-case", json=case_data) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Validate employment case specific aspects
                    validation_passed = True
                    validation_details = []
                    
                    # Check that recommendations include employment-specific advice
                    recommendations_text = " ".join(data.get("recommendations", [])).lower()
                    employment_keywords = ["employment", "discrimination", "workplace", "termination", "eeoc"]
                    
                    if not any(keyword in recommendations_text for keyword in employment_keywords):
                        validation_details.append("Recommendations lack employment-specific guidance")
                    
                    # Check risk factors are relevant to employment law
                    risk_factors_text = " ".join(data.get("risk_factors", [])).lower()
                    
                    # Validate duration is reasonable for employment case
                    if data.get("estimated_duration") and data["estimated_duration"] > 730:  # > 2 years
                        validation_details.append(f"Unusually long duration for employment case: {data['estimated_duration']} days")
                    
                    if len(validation_details) == 0:
                        self.log_result(
                            "Employment Medium Complexity Case",
                            True,
                            f"Complexity: {case_data['case_complexity']}, Predicted: {data['predicted_outcome']}, Duration: {data.get('estimated_duration', 'N/A')} days, Settlement prob: {data.get('settlement_probability', 0):.2%}",
                            data
                        )
                    else:
                        self.log_result(
                            "Employment Medium Complexity Case",
                            False,
                            f"Validation issues: {'; '.join(validation_details)}"
                        )
                else:
                    error_text = await response.text()
                    self.log_result(
                        "Employment Medium Complexity Case",
                        False,
                        f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_result(
                "Employment Medium Complexity Case",
                False,
                f"Exception: {str(e)}"
            )
    
    async def test_ip_case_strong_evidence(self):
        """Test intellectual property case with strong evidence"""
        print("\n🔬 Testing Intellectual Property Case with Strong Evidence...")
        
        case_data = {
            "case_type": "intellectual_property",
            "jurisdiction": "federal",
            "court_level": "district",
            "judge_name": "Judge Robert Chen",
            "case_facts": "Patent infringement case with clear evidence of unauthorized use of patented technology. Defendant's product contains identical implementation of plaintiff's patented algorithm. Strong documentary evidence including internal emails acknowledging the patent and deliberate copying.",
            "legal_issues": [
                "patent_infringement",
                "willful_infringement",
                "damages_calculation",
                "injunctive_relief",
                "attorney_fees"
            ],
            "case_complexity": 0.75,
            "case_value": 8500000.0,
            "evidence_strength": 9.2,  # Very strong evidence
            "witness_count": 22,
            "settlement_offers": [2000000.0, 3500000.0]
        }
        
        try:
            async with self.session.post(f"{API_BASE}/litigation/analyze-case", json=case_data) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Validate strong evidence case characteristics
                    validation_passed = True
                    validation_details = []
                    
                    # Strong evidence should lead to higher confidence
                    if data["confidence_score"] < 0.2:  # Minimum threshold for strong evidence
                        validation_details.append(f"Low confidence despite strong evidence: {data['confidence_score']:.2%}")
                    
                    # Check for IP-specific recommendations
                    recommendations_text = " ".join(data.get("recommendations", [])).lower()
                    ip_keywords = ["patent", "infringement", "injunction", "damages", "intellectual property"]
                    
                    if not any(keyword in recommendations_text for keyword in ip_keywords):
                        validation_details.append("Missing IP-specific recommendations")
                    
                    # Strong evidence should favor plaintiff
                    plaintiff_prob = data["probability_breakdown"].get("plaintiff_win", 0)
                    if plaintiff_prob < 0.2:  # Should have reasonable plaintiff win probability
                        validation_details.append(f"Low plaintiff probability despite strong evidence: {plaintiff_prob:.2%}")
                    
                    if len(validation_details) == 0:
                        self.log_result(
                            "IP Case Strong Evidence",
                            True,
                            f"Evidence strength: {case_data['evidence_strength']}/10, Confidence: {data['confidence_score']:.2%}, Plaintiff win prob: {plaintiff_prob:.2%}",
                            data
                        )
                    else:
                        self.log_result(
                            "IP Case Strong Evidence",
                            False,
                            f"Validation issues: {'; '.join(validation_details)}"
                        )
                else:
                    error_text = await response.text()
                    self.log_result(
                        "IP Case Strong Evidence",
                        False,
                        f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_result(
                "IP Case Strong Evidence",
                False,
                f"Exception: {str(e)}"
            )
    
    async def test_personal_injury_settlement_potential(self):
        """Test personal injury case with settlement potential"""
        print("\n🏥 Testing Personal Injury Case with Settlement Potential...")
        
        # First test case analysis
        case_data = {
            "case_type": "personal_injury",
            "jurisdiction": "texas",
            "court_level": "state",
            "judge_name": "Judge Susan Taylor",
            "case_facts": "Motor vehicle accident resulting in significant injuries. Clear liability with defendant running red light. Plaintiff suffered multiple fractures, required surgery, and has ongoing medical treatment. Strong medical documentation and witness testimony.",
            "legal_issues": [
                "negligence",
                "medical_expenses",
                "lost_wages",
                "pain_and_suffering",
                "future_medical_costs"
            ],
            "case_complexity": 0.4,  # Lower complexity for clear liability
            "case_value": 650000.0,
            "evidence_strength": 8.5,
            "witness_count": 8
        }
        
        try:
            async with self.session.post(f"{API_BASE}/litigation/analyze-case", json=case_data) as response:
                if response.status == 200:
                    case_analysis = await response.json()
                    
                    # Now test settlement probability for the same case
                    settlement_data = {
                        "case_type": "personal_injury",
                        "jurisdiction": "texas",
                        "case_value": 650000.0,
                        "evidence_strength": 8.5,
                        "case_complexity": 0.4
                    }
                    
                    async with self.session.post(f"{API_BASE}/litigation/settlement-probability", json=settlement_data) as settlement_response:
                        if settlement_response.status == 200:
                            settlement_analysis = await settlement_response.json()
                            
                            # Validate settlement potential
                            validation_passed = True
                            validation_details = []
                            
                            # Personal injury with strong evidence should have good settlement probability
                            settlement_prob = settlement_analysis["settlement_probability"]
                            if settlement_prob < 0.3:  # Should have decent settlement probability
                                validation_details.append(f"Low settlement probability for PI case: {settlement_prob:.2%}")
                            
                            # Check settlement ranges are reasonable
                            plaintiff_range = settlement_analysis["plaintiff_settlement_range"]
                            if plaintiff_range["high"] <= plaintiff_range["low"]:
                                validation_details.append("Invalid settlement range")
                            
                            # Settlement value should be substantial portion of case value
                            expected_settlement = settlement_analysis["expected_settlement_value"]
                            if expected_settlement < case_data["case_value"] * 0.2:  # At least 20% of case value
                                validation_details.append(f"Low expected settlement: ${expected_settlement:,.0f}")
                            
                            if len(validation_details) == 0:
                                self.log_result(
                                    "Personal Injury Settlement Potential",
                                    True,
                                    f"Settlement prob: {settlement_prob:.2%}, Expected: ${expected_settlement:,.0f}, Range: ${plaintiff_range['low']:,.0f}-${plaintiff_range['high']:,.0f}",
                                    {"case_analysis": case_analysis, "settlement_analysis": settlement_analysis}
                                )
                            else:
                                self.log_result(
                                    "Personal Injury Settlement Potential",
                                    False,
                                    f"Validation issues: {'; '.join(validation_details)}"
                                )
                        else:
                            error_text = await settlement_response.text()
                            self.log_result(
                                "Personal Injury Settlement Potential",
                                False,
                                f"Settlement analysis failed - HTTP {settlement_response.status}: {error_text}"
                            )
                else:
                    error_text = await response.text()
                    self.log_result(
                        "Personal Injury Settlement Potential",
                        False,
                        f"Case analysis failed - HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_result(
                "Personal Injury Settlement Potential",
                False,
                f"Exception: {str(e)}"
            )
    
    async def test_mongodb_collections_verification(self):
        """Verify MongoDB collections are created and populated"""
        print("\n🗄️ Testing MongoDB Collections Verification...")
        
        # Test analytics dashboard to verify database operations
        try:
            async with self.session.get(f"{API_BASE}/litigation/analytics-dashboard") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check if collections are being accessed
                    overview = data["overview"]
                    recent_activity = data["recent_activity"]
                    
                    validation_passed = True
                    validation_details = []
                    
                    # Verify database operations are working
                    if "total_cases_analyzed" not in overview:
                        validation_details.append("Missing total_cases_analyzed - litigation_cases collection issue")
                    
                    if "total_predictions_made" not in overview:
                        validation_details.append("Missing total_predictions_made - litigation_analytics collection issue")
                    
                    if "total_settlements_tracked" not in overview:
                        validation_details.append("Missing total_settlements_tracked - settlement_data collection issue")
                    
                    # Check if predictions are being stored (should increase with our tests)
                    predictions_count = overview.get("total_predictions_made", 0)
                    if predictions_count < 1:
                        validation_details.append("No predictions stored - database write operations may not be working")
                    
                    if len(validation_details) == 0:
                        self.log_result(
                            "MongoDB Collections Verification",
                            True,
                            f"Collections operational - Cases: {overview['total_cases_analyzed']}, Predictions: {overview['total_predictions_made']}, Settlements: {overview['total_settlements_tracked']}",
                            data
                        )
                    else:
                        self.log_result(
                            "MongoDB Collections Verification",
                            False,
                            f"Database issues: {'; '.join(validation_details)}"
                        )
                else:
                    error_text = await response.text()
                    self.log_result(
                        "MongoDB Collections Verification",
                        False,
                        f"Dashboard access failed - HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_result(
                "MongoDB Collections Verification",
                False,
                f"Exception: {str(e)}"
            )
    
    async def test_ai_model_integration_verification(self):
        """Verify AI model integration is working (Gemini/Groq APIs)"""
        print("\n🤖 Testing AI Model Integration Verification...")
        
        # Test with a case that should generate substantive AI analysis
        case_data = {
            "case_type": "commercial",
            "jurisdiction": "federal",
            "case_facts": "This is a complex multi-jurisdictional commercial dispute involving breach of contract, fraud allegations, and significant damages claims. The case involves multiple parties across different states with intricate contractual relationships and disputed performance obligations.",
            "legal_issues": [
                "breach_of_contract",
                "fraud",
                "multi_jurisdictional_issues",
                "complex_damages"
            ],
            "case_complexity": 0.9,
            "case_value": 5000000.0,
            "evidence_strength": 7.0
        }
        
        try:
            async with self.session.post(f"{API_BASE}/litigation/analyze-case", json=case_data) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Verify AI-generated content shows actual analysis
                    validation_passed = True
                    validation_details = []
                    
                    # Check for substantive recommendations (not placeholder text)
                    recommendations = data.get("recommendations", [])
                    if len(recommendations) < 3:
                        validation_details.append(f"Too few recommendations: {len(recommendations)}")
                    
                    # Check recommendations contain substantive content
                    rec_text = " ".join(recommendations).lower()
                    if "placeholder" in rec_text or "mock" in rec_text or "test" in rec_text:
                        validation_details.append("Recommendations contain placeholder/mock text")
                    
                    # Check for reasonable recommendation length (indicates real AI processing)
                    avg_rec_length = sum(len(rec) for rec in recommendations) / len(recommendations) if recommendations else 0
                    if avg_rec_length < 20:  # Very short recommendations might be mock data
                        validation_details.append(f"Recommendations too short (avg: {avg_rec_length} chars)")
                    
                    # Check risk factors are substantive
                    risk_factors = data.get("risk_factors", [])
                    if len(risk_factors) < 2:
                        validation_details.append(f"Too few risk factors: {len(risk_factors)}")
                    
                    # Check success factors are provided
                    success_factors = data.get("success_factors", [])
                    if len(success_factors) < 2:
                        validation_details.append(f"Too few success factors: {len(success_factors)}")
                    
                    # Confidence score should be reasonable (not exactly 0.5 which might indicate mock)
                    confidence = data["confidence_score"]
                    if confidence == 0.5:
                        validation_details.append("Confidence score exactly 0.5 - may indicate mock data")
                    
                    if len(validation_details) == 0:
                        self.log_result(
                            "AI Model Integration Verification",
                            True,
                            f"AI analysis appears genuine - {len(recommendations)} recommendations, {len(risk_factors)} risk factors, avg rec length: {avg_rec_length:.0f} chars",
                            data
                        )
                    else:
                        self.log_result(
                            "AI Model Integration Verification",
                            False,
                            f"AI integration issues: {'; '.join(validation_details)}"
                        )
                else:
                    error_text = await response.text()
                    self.log_result(
                        "AI Model Integration Verification",
                        False,
                        f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            self.log_result(
                "AI Model Integration Verification",
                False,
                f"Exception: {str(e)}"
            )
    
    async def run_specific_scenarios(self):
        """Run all specific scenario tests"""
        print("🎯 Starting Specific Litigation Analytics Scenarios Testing")
        print(f"🌐 Testing against: {API_BASE}")
        print("=" * 80)
        
        start_time = time.time()
        
        # Run specific scenario tests
        await self.test_commercial_litigation_high_value()
        await self.test_employment_law_medium_complexity()
        await self.test_ip_case_strong_evidence()
        await self.test_personal_injury_settlement_potential()
        await self.test_mongodb_collections_verification()
        await self.test_ai_model_integration_verification()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["success"])
        failed_tests = total_tests - passed_tests
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 SPECIFIC SCENARIOS TEST RESULTS")
        print("=" * 80)
        print(f"Total Scenario Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"⏱️ Duration: {duration:.2f} seconds")
        
        if failed_tests > 0:
            print("\n❌ FAILED SCENARIOS:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['details']}")
        
        return passed_tests, failed_tests

async def main():
    """Main test execution"""
    async with SpecificScenariosTestSuite() as test_suite:
        passed, failed = await test_suite.run_specific_scenarios()
        return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)