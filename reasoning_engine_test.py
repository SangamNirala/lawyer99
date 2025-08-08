#!/usr/bin/env python3
"""
Multi-Step Legal Reasoning Engine Test Suite
Tests the 3 core endpoints for the IRAC-based legal analysis system
"""

import requests
import sys
import json
from datetime import datetime

class MultiStepLegalReasoningTester:
    def __init__(self, base_url="https://8fa35272-f813-4f54-99e2-91d7f9fb1b50.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            print(f"   Status: {response.status_code}")
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'List with ' + str(len(response_data)) + ' items'}")
                    return True, response_data
                except:
                    return True, response.text
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timed out after {timeout} seconds")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_comprehensive_legal_analysis(self):
        """Test POST /api/legal-reasoning/comprehensive-analysis - IRAC-based comprehensive legal analysis"""
        print("\n🎯 Testing Comprehensive Legal Analysis (IRAC-based)")
        
        # Test with the exact scenario from the user request
        analysis_request = {
            "query": "A company terminated an employee without proper notice period as specified in their employment contract. The employee claims wrongful termination and seeks damages.",
            "jurisdiction": "US",
            "legal_domain": "employment_law",
            "user_type": "business_user",
            "context": {
                "contract_type": "employment_agreement",
                "termination_type": "without_cause",
                "notice_period_specified": "30_days",
                "actual_notice_given": "immediate"
            }
        }
        
        success, response = self.run_test(
            "Comprehensive Legal Analysis - Employment Termination",
            "POST",
            "legal-reasoning/comprehensive-analysis",
            200,
            analysis_request,
            timeout=120  # Allow more time for comprehensive analysis
        )
        
        if success and response:
            print(f"   Analysis ID: {response.get('analysis_id', 'N/A')}")
            print(f"   Query: {response.get('query', 'N/A')[:100]}...")
            print(f"   Jurisdiction: {response.get('jurisdiction', 'N/A')}")
            print(f"   User Type: {response.get('user_type', 'N/A')}")
            
            # Verify 7-step reasoning chain
            reasoning_chain = response.get('legal_reasoning_chain', [])
            print(f"   Reasoning Steps: {len(reasoning_chain)}")
            
            if len(reasoning_chain) >= 7:
                print(f"   ✅ All 7 reasoning steps executed")
                for i, step in enumerate(reasoning_chain[:7]):
                    step_name = step.get('step', f'Step {i+1}')
                    confidence = step.get('confidence', 0)
                    print(f"     Step {i+1}: {step_name} (Confidence: {confidence:.2f})")
            else:
                print(f"   ❌ Expected 7 reasoning steps, found {len(reasoning_chain)}")
            
            # Check legal issues identification
            legal_issues = response.get('legal_issues', [])
            print(f"   Legal Issues Identified: {len(legal_issues)}")
            
            if legal_issues:
                for issue in legal_issues[:3]:  # Show first 3
                    print(f"     - {issue.get('description', 'N/A')[:80]}...")
                print(f"   ✅ Legal issues identification working")
            else:
                print(f"   ❌ No legal issues identified")
            
            # Check concept extraction
            applicable_concepts = response.get('applicable_concepts', [])
            print(f"   Applicable Concepts: {len(applicable_concepts)}")
            
            if applicable_concepts:
                concept_names = [c.get('concept_name', 'Unknown') for c in applicable_concepts[:5]]
                print(f"     Concepts: {', '.join(concept_names)}")
                print(f"   ✅ Legal concept extraction working")
            else:
                print(f"   ❌ No applicable concepts identified")
            
            # Check precedent analysis
            controlling_precedents = response.get('controlling_precedents', [])
            print(f"   Controlling Precedents: {len(controlling_precedents)}")
            
            if controlling_precedents:
                for precedent in controlling_precedents[:2]:  # Show first 2
                    case_name = precedent.get('case_name', 'Unknown Case')
                    print(f"     - {case_name}")
                print(f"   ✅ Precedent analysis working")
            else:
                print(f"   ⚠️  No controlling precedents found")
            
            # Check legal conclusions
            legal_conclusions = response.get('legal_conclusions', [])
            print(f"   Legal Conclusions: {len(legal_conclusions)}")
            
            if legal_conclusions:
                for conclusion in legal_conclusions[:2]:  # Show first 2
                    conclusion_text = conclusion.get('conclusion', 'N/A')[:100]
                    confidence = conclusion.get('confidence', 0)
                    print(f"     - {conclusion_text}... (Confidence: {confidence:.2f})")
                print(f"   ✅ Legal conclusions generated")
            else:
                print(f"   ❌ No legal conclusions generated")
            
            # Check overall confidence and execution time
            confidence_score = response.get('confidence_score', 0)
            execution_time = response.get('execution_time', 0)
            print(f"   Overall Confidence: {confidence_score:.2f}")
            print(f"   Execution Time: {execution_time:.2f} seconds")
            
            if execution_time < 10:
                print(f"   ✅ Response time acceptable (<10 seconds)")
            else:
                print(f"   ⚠️  Response time high (>{execution_time:.1f} seconds)")
            
            # Verify response structure matches Pydantic model
            required_fields = [
                'analysis_id', 'query', 'jurisdiction', 'user_type', 'legal_issues',
                'applicable_concepts', 'controlling_precedents', 'applicable_statutes',
                'legal_reasoning_chain', 'risk_assessment', 'legal_conclusions',
                'alternative_analyses', 'confidence_score', 'expert_validation_status',
                'execution_time', 'created_at'
            ]
            
            missing_fields = [field for field in required_fields if field not in response]
            if not missing_fields:
                print(f"   ✅ All required response fields present")
            else:
                print(f"   ❌ Missing response fields: {missing_fields}")
        
        return success, response

    def test_legal_risk_assessment(self):
        """Test POST /api/legal-reasoning/risk-assessment - Legal risk assessment"""
        print("\n⚖️ Testing Legal Risk Assessment")
        
        # Test with the exact scenario from the user request
        risk_request = {
            "legal_scenario": "We want to launch a new software product that uses AI to analyze customer data for personalized recommendations. The data includes personal information and behavioral patterns.",
            "jurisdiction": "US",
            "legal_domains": ["intellectual_property", "privacy_law"],
            "context": {
                "business_type": "software_company",
                "data_types": ["personal_information", "behavioral_patterns"],
                "ai_usage": "personalized_recommendations",
                "target_market": "consumers"
            }
        }
        
        success, response = self.run_test(
            "Legal Risk Assessment - AI Software Product",
            "POST",
            "legal-reasoning/risk-assessment",
            200,
            risk_request,
            timeout=90  # Allow time for risk analysis
        )
        
        if success and response:
            print(f"   Assessment ID: {response.get('assessment_id', 'N/A')}")
            print(f"   Legal Scenario: {response.get('legal_scenario', 'N/A')[:100]}...")
            print(f"   Jurisdiction: {response.get('jurisdiction', 'N/A')}")
            
            # Check risk categories
            overall_risk_level = response.get('overall_risk_level', 'unknown')
            risk_probability = response.get('risk_probability', 'unknown')
            risk_impact = response.get('risk_impact', 'unknown')
            
            print(f"   Overall Risk Level: {overall_risk_level}")
            print(f"   Risk Probability: {risk_probability}")
            print(f"   Risk Impact: {risk_impact}")
            
            if overall_risk_level in ['low', 'medium', 'high', 'critical']:
                print(f"   ✅ Valid risk level assessment")
            else:
                print(f"   ❌ Invalid risk level: {overall_risk_level}")
            
            # Check risk categories
            risk_categories = response.get('risk_categories', [])
            print(f"   Risk Categories: {len(risk_categories)}")
            
            if risk_categories:
                print(f"     Categories: {', '.join(risk_categories)}")
                print(f"   ✅ Risk categories identified")
            else:
                print(f"   ❌ No risk categories identified")
            
            # Check detailed risks
            detailed_risks = response.get('detailed_risks', [])
            print(f"   Detailed Risks: {len(detailed_risks)}")
            
            if detailed_risks:
                for risk in detailed_risks[:3]:  # Show first 3
                    risk_type = risk.get('risk_type', 'Unknown')
                    description = risk.get('description', 'N/A')[:80]
                    probability = risk.get('probability', 'unknown')
                    impact = risk.get('impact', 'unknown')
                    print(f"     - {risk_type}: {description}... (P:{probability}, I:{impact})")
                print(f"   ✅ Detailed risk analysis provided")
            else:
                print(f"   ❌ No detailed risks provided")
            
            # Check mitigation strategies
            mitigation_strategies = response.get('mitigation_strategies', [])
            print(f"   Mitigation Strategies: {len(mitigation_strategies)}")
            
            if mitigation_strategies:
                for strategy in mitigation_strategies[:3]:  # Show first 3
                    print(f"     - {strategy[:100]}...")
                print(f"   ✅ Mitigation strategies provided")
            else:
                print(f"   ❌ No mitigation strategies provided")
            
            # Check compliance concerns
            compliance_concerns = response.get('compliance_concerns', [])
            print(f"   Compliance Concerns: {len(compliance_concerns)}")
            
            if compliance_concerns:
                for concern in compliance_concerns[:3]:  # Show first 3
                    print(f"     - {concern[:100]}...")
                print(f"   ✅ Compliance concerns identified")
            else:
                print(f"   ⚠️  No compliance concerns identified")
            
            # Check recommended actions
            recommended_actions = response.get('recommended_actions', [])
            print(f"   Recommended Actions: {len(recommended_actions)}")
            
            if recommended_actions:
                for action in recommended_actions[:3]:  # Show first 3
                    print(f"     - {action[:100]}...")
                print(f"   ✅ Recommended actions provided")
            else:
                print(f"   ❌ No recommended actions provided")
            
            # Check confidence score
            confidence_score = response.get('confidence_score', 0)
            print(f"   Confidence Score: {confidence_score:.2f}")
            
            if 0.5 <= confidence_score <= 1.0:
                print(f"   ✅ Reasonable confidence score")
            else:
                print(f"   ⚠️  Confidence score may be low: {confidence_score}")
            
            # Verify response structure
            required_fields = [
                'assessment_id', 'legal_scenario', 'jurisdiction', 'overall_risk_level',
                'risk_probability', 'risk_impact', 'risk_categories', 'detailed_risks',
                'mitigation_strategies', 'compliance_concerns', 'recommended_actions',
                'confidence_score', 'created_at'
            ]
            
            missing_fields = [field for field in required_fields if field not in response]
            if not missing_fields:
                print(f"   ✅ All required response fields present")
            else:
                print(f"   ❌ Missing response fields: {missing_fields}")
        
        return success, response

    def test_multi_jurisdiction_analysis(self):
        """Test POST /api/legal-reasoning/multi-jurisdiction-analysis - Cross-jurisdictional analysis"""
        print("\n🌍 Testing Multi-Jurisdiction Legal Analysis")
        
        # Test with the exact scenario from the user request
        jurisdiction_request = {
            "query": "A US-based company wants to expand operations to India and UK, requiring data processing agreements with local partners.",
            "primary_jurisdiction": "US",
            "comparison_jurisdictions": ["IN", "UK"],
            "legal_domain": "privacy_law",
            "context": {
                "business_type": "data_processing",
                "expansion_type": "international_operations",
                "partnership_requirements": "local_data_processing",
                "compliance_focus": "data_protection"
            }
        }
        
        success, response = self.run_test(
            "Multi-Jurisdiction Analysis - International Data Processing",
            "POST",
            "legal-reasoning/multi-jurisdiction-analysis",
            200,
            jurisdiction_request,
            timeout=150  # Allow more time for multi-jurisdiction analysis
        )
        
        if success and response:
            print(f"   Comparison ID: {response.get('comparison_id', 'N/A')}")
            print(f"   Query: {response.get('query', 'N/A')[:100]}...")
            print(f"   Primary Jurisdiction: {response.get('primary_jurisdiction', 'N/A')}")
            print(f"   Comparison Jurisdictions: {response.get('comparison_jurisdictions', [])}")
            
            # Check jurisdictional analysis
            jurisdictional_analysis = response.get('jurisdictional_analysis', {})
            print(f"   Jurisdictional Analysis: {len(jurisdictional_analysis)} jurisdictions")
            
            if jurisdictional_analysis:
                for jurisdiction, analysis in jurisdictional_analysis.items():
                    legal_framework = analysis.get('legal_framework', 'N/A')[:80]
                    key_statutes = analysis.get('key_statutes', [])
                    leading_cases = analysis.get('leading_cases', [])
                    
                    print(f"     {jurisdiction}:")
                    print(f"       Framework: {legal_framework}...")
                    print(f"       Statutes: {len(key_statutes)}")
                    print(f"       Cases: {len(leading_cases)}")
                
                print(f"   ✅ Jurisdictional analysis provided")
            else:
                print(f"   ❌ No jurisdictional analysis provided")
            
            # Check legal conflicts identification
            legal_conflicts = response.get('legal_conflicts', [])
            print(f"   Legal Conflicts: {len(legal_conflicts)}")
            
            if legal_conflicts:
                for conflict in legal_conflicts[:3]:  # Show first 3
                    conflict_type = conflict.get('conflict_type', 'Unknown')
                    jurisdictions_involved = conflict.get('jurisdictions_involved', [])
                    description = conflict.get('conflict_description', 'N/A')[:80]
                    
                    print(f"     - {conflict_type} ({', '.join(jurisdictions_involved)}): {description}...")
                
                print(f"   ✅ Legal conflicts identified")
            else:
                print(f"   ⚠️  No legal conflicts identified")
            
            # Check harmonization opportunities
            harmonization_opportunities = response.get('harmonization_opportunities', [])
            print(f"   Harmonization Opportunities: {len(harmonization_opportunities)}")
            
            if harmonization_opportunities:
                for opportunity in harmonization_opportunities[:3]:  # Show first 3
                    print(f"     - {opportunity[:100]}...")
                print(f"   ✅ Harmonization opportunities identified")
            else:
                print(f"   ⚠️  No harmonization opportunities identified")
            
            # Check forum shopping considerations
            forum_shopping = response.get('forum_shopping_considerations', [])
            print(f"   Forum Shopping Considerations: {len(forum_shopping)}")
            
            if forum_shopping:
                for consideration in forum_shopping[:2]:  # Show first 2
                    if isinstance(consideration, dict):
                        jurisdiction = consideration.get('jurisdiction', 'Unknown')
                        advantages = consideration.get('advantages', [])
                        likelihood = consideration.get('likelihood_of_jurisdiction', 'unknown')
                        print(f"     - {jurisdiction}: {len(advantages)} advantages (Likelihood: {likelihood})")
                    else:
                        print(f"     - {consideration[:100]}...")
                print(f"   ✅ Forum shopping analysis provided")
            else:
                print(f"   ⚠️  No forum shopping considerations provided")
            
            # Check choice of law recommendations
            choice_of_law = response.get('choice_of_law_recommendations', [])
            print(f"   Choice of Law Recommendations: {len(choice_of_law)}")
            
            if choice_of_law:
                for recommendation in choice_of_law[:2]:  # Show first 2
                    if isinstance(recommendation, dict):
                        scenario = recommendation.get('scenario', 'Unknown')[:50]
                        recommended_jurisdiction = recommendation.get('recommended_jurisdiction', 'Unknown')
                        print(f"     - {scenario}... → {recommended_jurisdiction}")
                    else:
                        print(f"     - {recommendation[:100]}...")
                print(f"   ✅ Choice of law recommendations provided")
            else:
                print(f"   ⚠️  No choice of law recommendations provided")
            
            # Check enforcement analysis
            enforcement_analysis = response.get('enforcement_analysis', {})
            print(f"   Enforcement Analysis: {len(enforcement_analysis)} aspects")
            
            if enforcement_analysis:
                cross_border = enforcement_analysis.get('cross_border_enforcement', 'N/A')[:80]
                treaty_considerations = enforcement_analysis.get('treaty_considerations', 'N/A')[:80]
                recognition_issues = enforcement_analysis.get('recognition_issues', 'N/A')[:80]
                
                print(f"     Cross-border: {cross_border}...")
                print(f"     Treaties: {treaty_considerations}...")
                print(f"     Recognition: {recognition_issues}...")
                print(f"   ✅ Enforcement analysis provided")
            else:
                print(f"   ❌ No enforcement analysis provided")
            
            # Check practical implications
            practical_implications = response.get('practical_implications', [])
            print(f"   Practical Implications: {len(practical_implications)}")
            
            if practical_implications:
                for implication in practical_implications[:3]:  # Show first 3
                    if isinstance(implication, dict):
                        implication_text = implication.get('implication', 'Unknown')[:80]
                        affected_jurisdictions = implication.get('affected_jurisdictions', [])
                        print(f"     - {implication_text}... (Affects: {', '.join(affected_jurisdictions)})")
                    else:
                        print(f"     - {implication[:100]}...")
                print(f"   ✅ Practical implications provided")
            else:
                print(f"   ⚠️  No practical implications provided")
            
            # Check confidence score
            confidence_score = response.get('confidence_score', 0)
            print(f"   Confidence Score: {confidence_score:.2f}")
            
            if 0.5 <= confidence_score <= 1.0:
                print(f"   ✅ Reasonable confidence score")
            else:
                print(f"   ⚠️  Confidence score may be low: {confidence_score}")
            
            # Verify response structure
            required_fields = [
                'comparison_id', 'query', 'primary_jurisdiction', 'comparison_jurisdictions',
                'jurisdictional_analysis', 'legal_conflicts', 'harmonization_opportunities',
                'forum_shopping_considerations', 'choice_of_law_recommendations',
                'enforcement_analysis', 'practical_implications', 'confidence_score', 'created_at'
            ]
            
            missing_fields = [field for field in required_fields if field not in response]
            if not missing_fields:
                print(f"   ✅ All required response fields present")
            else:
                print(f"   ❌ Missing response fields: {missing_fields}")
        
        return success, response

    def run_all_tests(self):
        """Run all Multi-Step Legal Reasoning Engine tests"""
        print("🚀 Starting Multi-Step Legal Reasoning Engine Testing...")
        print(f"Base URL: {self.base_url}")
        print("=" * 80)
        print("🧠 MULTI-STEP LEGAL REASONING ENGINE TESTING")
        print("=" * 80)
        print("Testing the 3 core endpoints that integrate all legal AI components")
        print("into sophisticated IRAC-based legal analysis")
        
        # Test 1: Comprehensive Legal Analysis
        self.test_comprehensive_legal_analysis()
        
        # Test 2: Legal Risk Assessment
        self.test_legal_risk_assessment()
        
        # Test 3: Multi-Jurisdiction Analysis
        self.test_multi_jurisdiction_analysis()
        
        print("=" * 80)
        print("🧠 MULTI-STEP LEGAL REASONING ENGINE TESTING COMPLETE")
        print("=" * 80)
        print(f"📊 Results: {self.tests_passed}/{self.tests_run} tests passed")
        print(f"✅ Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        print("=" * 80)

if __name__ == "__main__":
    tester = MultiStepLegalReasoningTester()
    tester.run_all_tests()
    
    print(f"\n📊 Final Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    print(f"✅ Success Rate: {(tester.tests_passed/tester.tests_run)*100:.1f}%")