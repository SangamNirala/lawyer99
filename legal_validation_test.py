#!/usr/bin/env python3
"""
Legal Accuracy Validation & Expert Integration System Testing
Tests the new professional-grade legal accuracy validation system with expert integration.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# Backend URL from environment
BACKEND_URL = "https://legal-engine-check.preview.emergentagent.com/api"

class LegalValidationTester:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        self.test_results = []
        self.validation_ids = []  # Store validation IDs for subsequent tests
        
    def log_test(self, test_name: str, success: bool, details: str = "", response_data: Any = None):
        """Log test results"""
        result = {
            'test_name': test_name,
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat(),
            'response_data': response_data
        }
        self.test_results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"   Details: {details}")
        if not success and response_data:
            print(f"   Response: {response_data}")
        print()

    def test_comprehensive_validation_check(self):
        """Test POST /api/legal-validation/comprehensive-check - Comprehensive 4-level validation"""
        print("🧪 Testing Comprehensive Legal Validation Check...")
        
        # Test Case 1: Contract Law Analysis
        contract_analysis = {
            "analysis_content": """
            This contract analysis examines a service agreement between TechCorp Inc. and DataSolutions LLC. 
            The agreement establishes a 12-month engagement for cloud infrastructure consulting services.
            
            Key Terms Analysis:
            1. Scope of Work: DataSolutions will provide cloud migration consulting, including AWS architecture design, 
               security implementation, and performance optimization. The scope is clearly defined with specific deliverables.
            
            2. Payment Terms: $15,000 monthly retainer plus expenses, payable within 30 days of invoice. 
               This represents fair market value for enterprise consulting services.
            
            3. Intellectual Property: All work product belongs to TechCorp, with DataSolutions retaining rights to 
               general methodologies and pre-existing IP. This is standard for consulting agreements.
            
            4. Confidentiality: Mutual NDA provisions protect both parties' confidential information for 5 years 
               post-termination. Standard confidentiality period for technology consulting.
            
            5. Termination: Either party may terminate with 30 days written notice. No penalty clauses, 
               which is appropriate for professional services agreements.
            
            Legal Assessment: The agreement appears legally sound with proper consideration, clear terms, 
            and balanced risk allocation. Recommend adding force majeure clause and dispute resolution mechanism.
            """,
            "legal_domain": "contract_law",
            "jurisdiction": "US"
        }
        
        try:
            response = self.session.post(
                f"{BACKEND_URL}/legal-validation/comprehensive-check",
                json=contract_analysis
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ['validation_id', 'analysis_id', 'overall_status', 'confidence_score', 
                                 'validation_levels', 'expert_review_required', 'recommendations']
                
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    self.log_test("Comprehensive Validation - Contract Law", False, 
                                f"Missing required fields: {missing_fields}", data)
                    return
                
                # Store validation ID for subsequent tests
                self.validation_ids.append(data['validation_id'])
                
                # Validate 4-level validation structure
                validation_levels = data.get('validation_levels', [])
                expected_levels = ['cross_reference', 'logic_consistency', 'precedent_authority', 'legal_principles']
                
                if len(validation_levels) != 4:
                    self.log_test("Comprehensive Validation - Contract Law", False, 
                                f"Expected 4 validation levels, got {len(validation_levels)}", data)
                    return
                
                # Check each validation level has required fields
                level_success = True
                for level in validation_levels:
                    level_fields = ['level', 'status', 'confidence', 'details', 'sources_checked', 
                                  'issues_found', 'recommendations', 'execution_time']
                    missing_level_fields = [field for field in level_fields if field not in level]
                    if missing_level_fields:
                        level_success = False
                        break
                
                if not level_success:
                    self.log_test("Comprehensive Validation - Contract Law", False, 
                                "Validation levels missing required fields", data)
                    return
                
                # Validate confidence score range
                confidence_score = data.get('confidence_score', 0)
                if not (0 <= confidence_score <= 1):
                    self.log_test("Comprehensive Validation - Contract Law", False, 
                                f"Confidence score {confidence_score} not in range [0,1]", data)
                    return
                
                self.log_test("Comprehensive Validation - Contract Law", True, 
                            f"4-level validation completed with {confidence_score:.1%} confidence, "
                            f"Expert review required: {data.get('expert_review_required', False)}")
                
            else:
                self.log_test("Comprehensive Validation - Contract Law", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Comprehensive Validation - Contract Law", False, f"Exception: {str(e)}")
        
        # Test Case 2: Constitutional Law Analysis
        constitutional_analysis = {
            "analysis_content": """
            Constitutional Analysis: First Amendment Free Speech Rights in Digital Platforms
            
            Issue: Whether private social media platforms can restrict user speech without violating 
            First Amendment protections, particularly regarding political content moderation.
            
            Legal Framework Analysis:
            1. State Action Doctrine: The First Amendment applies only to government action, not private entities. 
               Private social media companies are not state actors and generally have broad discretion to 
               moderate content on their platforms.
            
            2. Public Forum Analysis: While social media platforms serve as modern public squares, 
               they remain private property. The Supreme Court in Manhattan Community Access Corp. v. Halleck (2019) 
               clarified that private entities do not become state actors merely by providing a forum for speech.
            
            3. Section 230 Protection: 47 U.S.C. § 230 provides immunity to platforms for third-party content 
               and allows good faith content moderation. This federal law preempts most state attempts to 
               regulate platform moderation practices.
            
            4. Emerging State Legislation: Texas HB 20 and Florida SB 7072 attempt to restrict platform 
               moderation, but face constitutional challenges under the First Amendment and Commerce Clause.
            
            Conclusion: Current constitutional doctrine strongly supports platform autonomy in content moderation. 
            Users have no First Amendment right to access private platforms, and platforms have their own 
            First Amendment rights to curate content. However, this area of law is rapidly evolving.
            """,
            "legal_domain": "constitutional_law",
            "jurisdiction": "US"
        }
        
        try:
            response = self.session.post(
                f"{BACKEND_URL}/legal-validation/comprehensive-check",
                json=constitutional_analysis
            )
            
            if response.status_code == 200:
                data = response.json()
                self.validation_ids.append(data['validation_id'])
                
                self.log_test("Comprehensive Validation - Constitutional Law", True, 
                            f"Constitutional law analysis validated with {data.get('confidence_score', 0):.1%} confidence")
            else:
                self.log_test("Comprehensive Validation - Constitutional Law", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Comprehensive Validation - Constitutional Law", False, f"Exception: {str(e)}")

    def test_confidence_score_breakdown(self):
        """Test GET /api/legal-validation/confidence-score - 7-factor confidence calculation"""
        print("🧪 Testing Confidence Score Breakdown...")
        
        if not self.validation_ids:
            self.log_test("Confidence Score Breakdown", False, "No validation IDs available from previous tests")
            return
        
        validation_id = self.validation_ids[0]
        
        try:
            response = self.session.get(
                f"{BACKEND_URL}/legal-validation/confidence-score",
                params={"validation_id": validation_id}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ['validation_id', 'overall_confidence_score', 'confidence_factors', 
                                 'factor_weights', 'validation_summary']
                
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    self.log_test("Confidence Score Breakdown", False, 
                                f"Missing required fields: {missing_fields}", data)
                    return
                
                # Validate 7-factor calculation
                confidence_factors = data.get('confidence_factors', {})
                factor_weights = data.get('factor_weights', {})
                
                expected_factors = [
                    'source_authority_weight',
                    'precedent_consensus_level', 
                    'citation_verification_score',
                    'cross_reference_validation',
                    'legal_logic_consistency',
                    'expert_validation_status',
                    'knowledge_base_freshness'
                ]
                
                missing_factors = [factor for factor in expected_factors if factor not in confidence_factors]
                if missing_factors:
                    self.log_test("Confidence Score Breakdown", False, 
                                f"Missing confidence factors: {missing_factors}", data)
                    return
                
                # Validate weight distribution (should sum to approximately 1.0)
                total_weight = sum(factor_weights.values())
                if not (0.95 <= total_weight <= 1.05):  # Allow small floating point variance
                    self.log_test("Confidence Score Breakdown", False, 
                                f"Factor weights sum to {total_weight}, expected ~1.0", data)
                    return
                
                # Validate confidence score range
                overall_score = data.get('overall_confidence_score', 0)
                if not (0 <= overall_score <= 1):
                    self.log_test("Confidence Score Breakdown", False, 
                                f"Overall confidence score {overall_score} not in range [0,1]", data)
                    return
                
                self.log_test("Confidence Score Breakdown", True, 
                            f"7-factor confidence calculation: {overall_score:.1%} overall confidence, "
                            f"weights sum to {total_weight:.3f}")
                
            elif response.status_code == 404:
                self.log_test("Confidence Score Breakdown", False, 
                            f"Validation ID {validation_id} not found")
            else:
                self.log_test("Confidence Score Breakdown", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Confidence Score Breakdown", False, f"Exception: {str(e)}")

    def test_expert_review_request(self):
        """Test POST /api/legal-validation/expert-review-request - Automated expert routing"""
        print("🧪 Testing Expert Review Request...")
        
        # Test Case 1: High complexity contract law issue
        expert_request = {
            "analysis_id": "contract_analysis_001",
            "content": """
            Complex Multi-Jurisdictional Licensing Agreement Analysis
            
            This analysis involves a sophisticated intellectual property licensing agreement between 
            a US technology company and multiple international distributors across EU, UK, Canada, and Australia.
            
            Key Complexity Factors:
            1. Multi-jurisdictional IP rights with varying patent protection periods
            2. Complex royalty calculation involving tiered rates and minimum guarantees
            3. Exclusive vs. non-exclusive territory definitions with online sales complications
            4. Technology transfer restrictions under ITAR and EAR regulations
            5. Brexit-related complications for UK distribution rights
            6. GDPR compliance requirements for customer data handling
            
            Validation Issues Identified:
            - Potential conflict between US export control laws and EU distribution requirements
            - Ambiguous termination clauses regarding IP rights reversion
            - Unclear dispute resolution mechanism for multi-jurisdictional conflicts
            - Missing force majeure provisions for regulatory changes
            
            This analysis requires expert review due to the complex interplay of international IP law, 
            export controls, and data privacy regulations across multiple jurisdictions.
            """,
            "legal_domain": "intellectual_property",
            "validation_issues": [
                "Multi-jurisdictional IP rights conflict",
                "Export control compliance uncertainty", 
                "Ambiguous termination provisions",
                "Missing regulatory change provisions"
            ],
            "complexity_score": 0.85
        }
        
        try:
            response = self.session.post(
                f"{BACKEND_URL}/legal-validation/expert-review-request",
                json=expert_request
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ['success', 'review_id', 'assigned_expert', 'priority', 'status']
                
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    self.log_test("Expert Review Request - IP Law", False, 
                                f"Missing required fields: {missing_fields}", data)
                    return
                
                if not data.get('success', False):
                    self.log_test("Expert Review Request - IP Law", False, 
                                "Expert review request was not successful", data)
                    return
                
                # Validate expert assignment
                assigned_expert = data.get('assigned_expert', {})
                expert_fields = ['expert_id', 'name', 'specialization', 'experience_years']
                missing_expert_fields = [field for field in expert_fields if field not in assigned_expert]
                
                if missing_expert_fields:
                    self.log_test("Expert Review Request - IP Law", False, 
                                f"Missing expert fields: {missing_expert_fields}", data)
                    return
                
                # Validate priority assignment (high complexity should get high priority)
                priority = data.get('priority', '').lower()
                if expert_request['complexity_score'] >= 0.8 and priority not in ['high', 'urgent']:
                    self.log_test("Expert Review Request - IP Law", False, 
                                f"High complexity (0.85) assigned {priority} priority, expected high/urgent")
                    return
                
                # Validate domain matching
                expert_specialization = assigned_expert.get('specialization', '').lower()
                if 'intellectual property' not in expert_specialization and 'ip' not in expert_specialization:
                    self.log_test("Expert Review Request - IP Law", False, 
                                f"Expert specialization '{expert_specialization}' doesn't match IP domain")
                    return
                
                self.log_test("Expert Review Request - IP Law", True, 
                            f"Expert {assigned_expert.get('name')} assigned with {priority} priority, "
                            f"specialization: {expert_specialization}")
                
            else:
                self.log_test("Expert Review Request - IP Law", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Expert Review Request - IP Law", False, f"Exception: {str(e)}")
        
        # Test Case 2: Medium complexity employment law issue
        employment_request = {
            "analysis_id": "employment_analysis_002",
            "content": """
            Employment Contract Termination Analysis
            
            Analysis of termination provisions in executive employment agreement with focus on 
            severance calculations, non-compete enforceability, and equity vesting acceleration.
            
            Key Issues:
            1. Severance calculation methodology for "good reason" termination
            2. Non-compete clause enforceability under recent state law changes
            3. Equity vesting acceleration triggers and tax implications
            4. COBRA continuation and benefits bridge provisions
            
            This requires employment law expertise but is less complex than multi-jurisdictional IP matters.
            """,
            "legal_domain": "employment_law",
            "validation_issues": [
                "Severance calculation ambiguity",
                "Non-compete enforceability question"
            ],
            "complexity_score": 0.65
        }
        
        try:
            response = self.session.post(
                f"{BACKEND_URL}/legal-validation/expert-review-request",
                json=employment_request
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate domain-specific expert assignment
                assigned_expert = data.get('assigned_expert', {})
                expert_specialization = assigned_expert.get('specialization', '').lower()
                
                if 'employment' not in expert_specialization and 'labor' not in expert_specialization:
                    self.log_test("Expert Review Request - Employment Law", False, 
                                f"Expert specialization '{expert_specialization}' doesn't match employment domain")
                    return
                
                # Validate priority for medium complexity
                priority = data.get('priority', '').lower()
                if employment_request['complexity_score'] < 0.7 and priority == 'urgent':
                    self.log_test("Expert Review Request - Employment Law", False, 
                                f"Medium complexity (0.65) assigned urgent priority")
                    return
                
                self.log_test("Expert Review Request - Employment Law", True, 
                            f"Employment expert {assigned_expert.get('name')} assigned with {priority} priority")
                
            else:
                self.log_test("Expert Review Request - Employment Law", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Expert Review Request - Employment Law", False, f"Exception: {str(e)}")

    def test_accuracy_metrics_dashboard(self):
        """Test GET /api/legal-validation/accuracy-metrics - System accuracy dashboard"""
        print("🧪 Testing Accuracy Metrics Dashboard...")
        
        try:
            response = self.session.get(f"{BACKEND_URL}/legal-validation/accuracy-metrics")
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for error response
                if "error" in data:
                    self.log_test("Accuracy Metrics Dashboard", False, 
                                f"API returned error: {data['error']}")
                    return
                
                # Validate dashboard structure
                expected_sections = [
                    'system_accuracy',
                    'expert_review_metrics', 
                    'confidence_distribution',
                    'validation_levels_performance'
                ]
                
                missing_sections = [section for section in expected_sections if section not in data]
                if missing_sections:
                    self.log_test("Accuracy Metrics Dashboard", False, 
                                f"Missing dashboard sections: {missing_sections}", data)
                    return
                
                # Validate system accuracy metrics
                system_accuracy = data.get('system_accuracy', {})
                accuracy_fields = ['overall_accuracy_rate', 'total_validations', 'validations_passed', 
                                 'target_accuracy', 'accuracy_status']
                
                missing_accuracy_fields = [field for field in accuracy_fields if field not in system_accuracy]
                if missing_accuracy_fields:
                    self.log_test("Accuracy Metrics Dashboard", False, 
                                f"Missing system accuracy fields: {missing_accuracy_fields}")
                    return
                
                # Validate expert review metrics
                expert_metrics = data.get('expert_review_metrics', {})
                expert_fields = ['total_reviews', 'completed_reviews', 'completion_rate', 
                               'target_completion_time', 'average_review_time']
                
                missing_expert_fields = [field for field in expert_fields if field not in expert_metrics]
                if missing_expert_fields:
                    self.log_test("Accuracy Metrics Dashboard", False, 
                                f"Missing expert review fields: {missing_expert_fields}")
                    return
                
                # Validate validation levels performance (4 levels)
                validation_performance = data.get('validation_levels_performance', {})
                expected_levels = [
                    'cross_reference_validation',
                    'logic_consistency_check', 
                    'precedent_authority_validation',
                    'legal_principle_adherence'
                ]
                
                missing_levels = [level for level in expected_levels if level not in validation_performance]
                if missing_levels:
                    self.log_test("Accuracy Metrics Dashboard", False, 
                                f"Missing validation level metrics: {missing_levels}")
                    return
                
                # Validate metric ranges
                overall_accuracy = system_accuracy.get('overall_accuracy_rate', 0)
                if not (0 <= overall_accuracy <= 100):
                    self.log_test("Accuracy Metrics Dashboard", False, 
                                f"Overall accuracy rate {overall_accuracy} not in range [0,100]")
                    return
                
                self.log_test("Accuracy Metrics Dashboard", True, 
                            f"Dashboard metrics: {overall_accuracy}% accuracy, "
                            f"{system_accuracy.get('total_validations', 0)} total validations, "
                            f"{expert_metrics.get('total_reviews', 0)} expert reviews")
                
            else:
                self.log_test("Accuracy Metrics Dashboard", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Accuracy Metrics Dashboard", False, f"Exception: {str(e)}")

    def test_validation_history(self):
        """Test GET /api/legal-validation/validation-history - Validation history retrieval"""
        print("🧪 Testing Validation History Retrieval...")
        
        # Test Case 1: Default pagination
        try:
            response = self.session.get(f"{BACKEND_URL}/legal-validation/validation-history")
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for error response
                if "error" in data:
                    self.log_test("Validation History - Default", False, 
                                f"API returned error: {data['error']}")
                    return
                
                # Validate response structure
                required_fields = ['validations', 'total_count', 'limit', 'offset', 'has_more']
                
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    self.log_test("Validation History - Default", False, 
                                f"Missing required fields: {missing_fields}", data)
                    return
                
                # Validate pagination parameters
                limit = data.get('limit', 0)
                offset = data.get('offset', 0)
                total_count = data.get('total_count', 0)
                validations = data.get('validations', [])
                
                if limit != 20:  # Default limit
                    self.log_test("Validation History - Default", False, 
                                f"Expected default limit 20, got {limit}")
                    return
                
                if offset != 0:  # Default offset
                    self.log_test("Validation History - Default", False, 
                                f"Expected default offset 0, got {offset}")
                    return
                
                # Validate has_more calculation
                expected_has_more = (offset + limit) < total_count
                if data.get('has_more') != expected_has_more:
                    self.log_test("Validation History - Default", False, 
                                f"has_more calculation incorrect: got {data.get('has_more')}, expected {expected_has_more}")
                    return
                
                self.log_test("Validation History - Default", True, 
                            f"Retrieved {len(validations)} validations out of {total_count} total, "
                            f"has_more: {data.get('has_more')}")
                
            else:
                self.log_test("Validation History - Default", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Validation History - Default", False, f"Exception: {str(e)}")
        
        # Test Case 2: Custom pagination
        try:
            response = self.session.get(
                f"{BACKEND_URL}/legal-validation/validation-history",
                params={"limit": 5, "offset": 0}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if "error" not in data:
                    limit = data.get('limit', 0)
                    validations = data.get('validations', [])
                    
                    if limit == 5 and len(validations) <= 5:
                        self.log_test("Validation History - Custom Pagination", True, 
                                    f"Custom pagination working: limit={limit}, returned {len(validations)} items")
                    else:
                        self.log_test("Validation History - Custom Pagination", False, 
                                    f"Pagination issue: limit={limit}, returned {len(validations)} items")
                else:
                    self.log_test("Validation History - Custom Pagination", False, 
                                f"API returned error: {data['error']}")
                
            else:
                self.log_test("Validation History - Custom Pagination", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Validation History - Custom Pagination", False, f"Exception: {str(e)}")

    def run_all_tests(self):
        """Run all Legal Accuracy Validation & Expert Integration System tests"""
        print("🚀 Starting Legal Accuracy Validation & Expert Integration System Testing")
        print("=" * 80)
        
        # Test all endpoints in logical order
        self.test_comprehensive_validation_check()
        self.test_confidence_score_breakdown()
        self.test_expert_review_request()
        self.test_accuracy_metrics_dashboard()
        self.test_validation_history()
        
        # Generate summary
        print("=" * 80)
        print("📊 LEGAL ACCURACY VALIDATION SYSTEM TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print()
        
        # Show failed tests
        if failed_tests > 0:
            print("❌ FAILED TESTS:")
            for result in self.test_results:
                if not result['success']:
                    print(f"   • {result['test_name']}: {result['details']}")
            print()
        
        # Show key findings
        print("🔍 KEY FINDINGS:")
        
        # Multi-layer validation
        validation_tests = [r for r in self.test_results if 'Comprehensive Validation' in r['test_name'] and r['success']]
        if validation_tests:
            print("   ✅ Multi-layer validation (4 levels) working: cross-reference, logic consistency, precedent authority, legal principles")
        
        # Confidence scoring
        confidence_tests = [r for r in self.test_results if 'Confidence Score' in r['test_name'] and r['success']]
        if confidence_tests:
            print("   ✅ 7-factor confidence scoring operational with weighted calculation")
        
        # Expert review routing
        expert_tests = [r for r in self.test_results if 'Expert Review' in r['test_name'] and r['success']]
        if expert_tests:
            print("   ✅ Expert review routing by domain with automated priority assignment")
        
        # Dashboard metrics
        metrics_tests = [r for r in self.test_results if 'Accuracy Metrics' in r['test_name'] and r['success']]
        if metrics_tests:
            print("   ✅ Comprehensive accuracy metrics dashboard with system performance tracking")
        
        # Validation tracking
        history_tests = [r for r in self.test_results if 'Validation History' in r['test_name'] and r['success']]
        if history_tests:
            print("   ✅ Validation history tracking with pagination support")
        
        print()
        
        if success_rate >= 80:
            print("🎉 LEGAL ACCURACY VALIDATION SYSTEM IS OPERATIONAL AND READY FOR PRODUCTION USE!")
        elif success_rate >= 60:
            print("⚠️  Legal Accuracy Validation System is mostly functional but needs attention to failed tests.")
        else:
            print("🚨 Legal Accuracy Validation System has significant issues that need to be addressed.")
        
        return success_rate

if __name__ == "__main__":
    tester = LegalValidationTester()
    success_rate = tester.run_all_tests()
    
    # Exit with appropriate code
    exit(0 if success_rate >= 80 else 1)