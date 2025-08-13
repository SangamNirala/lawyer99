#!/usr/bin/env python3
"""
Advanced User Experience API Testing Suite
Tests the 6 new UX endpoints for sophisticated user interaction capabilities
"""

import requests
import sys
import json
import urllib.parse
from datetime import datetime

class AdvancedUXAPITester:
    def __init__(self, base_url="https://legal-engine-check.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def run_test(self, name, method, endpoint, expected_status, data=None, params=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, params=params, timeout=timeout)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, params=params, timeout=timeout)

            print(f"   Status: {response.status_code}")
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'List with ' + str(len(response_data)) + ' items'}")
                    self.test_results.append({"test": name, "status": "PASSED", "response": response_data})
                    return True, response_data
                except:
                    self.test_results.append({"test": name, "status": "PASSED", "response": response.text})
                    return True, response.text
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                    self.test_results.append({"test": name, "status": "FAILED", "error": error_data})
                except:
                    print(f"   Error: {response.text}")
                    self.test_results.append({"test": name, "status": "FAILED", "error": response.text})
                return False, None

        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            self.test_results.append({"test": name, "status": "ERROR", "error": str(e)})
            return False, None

    def test_adaptive_response_endpoint(self):
        """Test POST /api/ux/adaptive-response - Generate user-appropriate responses"""
        print("\n" + "="*80)
        print("🎯 TESTING ADAPTIVE RESPONSE ENDPOINT")
        print("="*80)
        
        # Test 1: Legal Professional Level Response
        test_data = {
            "content": "This contract contains a force majeure clause that may not be enforceable under current jurisdiction requirements.",
            "user_sophistication_level": "legal_professional",
            "legal_domain": "contract_law",
            "jurisdiction": "US",
            "context": {
                "contract_type": "commercial_agreement",
                "user_role": "attorney"
            }
        }
        
        success, response = self.run_test(
            "Adaptive Response - Legal Professional",
            "POST",
            "ux/adaptive-response",
            200,
            test_data
        )
        
        if success and response:
            # Verify response structure
            required_fields = ["adapted_content", "communication_style", "complexity_level", 
                             "interactive_elements", "follow_up_suggestions", "personalization_applied"]
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
            else:
                print(f"✅ All required response fields present")
                print(f"   Communication Style: {response.get('communication_style')}")
                print(f"   Complexity Level: {response.get('complexity_level')}")
                print(f"   Personalization Applied: {len(response.get('personalization_applied', []))} factors")
        
        # Test 2: General Consumer Level Response
        test_data = {
            "content": "This contract contains a force majeure clause that may not be enforceable under current jurisdiction requirements.",
            "user_sophistication_level": "general_consumer",
            "legal_domain": "contract_law",
            "jurisdiction": "US",
            "context": {
                "contract_type": "service_agreement",
                "user_role": "business_owner"
            }
        }
        
        success, response = self.run_test(
            "Adaptive Response - General Consumer",
            "POST",
            "ux/adaptive-response",
            200,
            test_data
        )
        
        # Test 3: Business Executive Level Response
        test_data = {
            "content": "The intellectual property assignment clause requires careful review for scope and limitations.",
            "user_sophistication_level": "business_executive",
            "legal_domain": "intellectual_property",
            "jurisdiction": "US",
            "context": {
                "contract_type": "employment_agreement",
                "user_role": "ceo"
            }
        }
        
        success, response = self.run_test(
            "Adaptive Response - Business Executive",
            "POST",
            "ux/adaptive-response",
            200,
            test_data
        )

    def test_user_sophistication_analysis_endpoint(self):
        """Test GET /api/ux/user-sophistication-analysis - Analyze user legal knowledge level"""
        print("\n" + "="*80)
        print("🧠 TESTING USER SOPHISTICATION ANALYSIS ENDPOINT")
        print("="*80)
        
        # Test 1: Legal Professional Query
        params = {
            "query_text": "I need to review the precedential value of this case law regarding promissory estoppel and its application to commercial contracts under the Uniform Commercial Code.",
            "user_context": json.dumps({
                "profession": "attorney",
                "years_experience": 10,
                "specialization": "commercial_law"
            }),
            "previous_queries": json.dumps([
                "What is the statute of limitations for breach of contract claims?",
                "How does the parol evidence rule apply to integrated contracts?",
                "What are the elements of a valid consideration?"
            ])
        }
        
        success, response = self.run_test(
            "Sophistication Analysis - Legal Professional",
            "GET",
            "ux/user-sophistication-analysis",
            200,
            params=params
        )
        
        if success and response:
            required_fields = ["sophistication_level", "confidence_score", "detected_indicators", 
                             "communication_preferences", "reasoning"]
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
            else:
                print(f"✅ All required response fields present")
                print(f"   Detected Level: {response.get('sophistication_level')}")
                print(f"   Confidence Score: {response.get('confidence_score')}")
                print(f"   Indicators Found: {len(response.get('detected_indicators', []))}")
        
        # Test 2: General Consumer Query
        params = {
            "query_text": "I need help understanding my lease agreement. What does it mean when it says I'm responsible for 'ordinary wear and tear'?",
            "user_context": json.dumps({
                "profession": "teacher",
                "legal_experience": "none"
            }),
            "previous_queries": json.dumps([
                "What is a security deposit?",
                "Can my landlord raise my rent?",
                "How much notice do I need to give before moving out?"
            ])
        }
        
        success, response = self.run_test(
            "Sophistication Analysis - General Consumer",
            "GET",
            "ux/user-sophistication-analysis",
            200,
            params=params
        )
        
        # Test 3: Business Executive Query
        params = {
            "query_text": "We're considering an acquisition and need to understand the due diligence requirements, regulatory approvals, and potential antitrust implications.",
            "user_context": json.dumps({
                "profession": "ceo",
                "company_size": "mid_market",
                "industry": "technology"
            }),
            "previous_queries": json.dumps([
                "What are the key terms in a merger agreement?",
                "How do we structure an earnout provision?",
                "What are the tax implications of an asset purchase vs stock purchase?"
            ])
        }
        
        success, response = self.run_test(
            "Sophistication Analysis - Business Executive",
            "GET",
            "ux/user-sophistication-analysis",
            200,
            params=params
        )

    def test_interactive_guidance_endpoint(self):
        """Test POST /api/ux/interactive-guidance - Provide step-by-step legal guidance"""
        print("\n" + "="*80)
        print("🗺️  TESTING INTERACTIVE GUIDANCE ENDPOINT")
        print("="*80)
        
        # Test 1: Contract Dispute Guidance
        test_data = {
            "legal_issue": "My contractor failed to complete work according to the contract specifications and is demanding full payment. I want to understand my options for resolving this dispute.",
            "user_sophistication_level": "general_consumer",
            "user_goals": [
                "Understand my legal rights",
                "Explore resolution options",
                "Minimize costs and time",
                "Preserve business relationship if possible"
            ],
            "jurisdiction": "US"
        }
        
        success, response = self.run_test(
            "Interactive Guidance - Contract Dispute",
            "POST",
            "ux/interactive-guidance",
            200,
            test_data
        )
        
        if success and response:
            required_fields = ["guidance_id", "legal_issue", "total_steps", "steps", 
                             "overall_complexity", "estimated_total_time", "key_considerations",
                             "professional_consultation_recommended"]
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
            else:
                print(f"✅ All required response fields present")
                print(f"   Total Steps: {response.get('total_steps')}")
                print(f"   Overall Complexity: {response.get('overall_complexity')}")
                print(f"   Estimated Time: {response.get('estimated_total_time')}")
                print(f"   Professional Consultation Recommended: {response.get('professional_consultation_recommended')}")
                
                # Verify steps structure
                steps = response.get('steps', [])
                if steps and len(steps) > 0:
                    first_step = steps[0]
                    step_fields = ["step_number", "title", "description", "action_items", 
                                 "resources", "risk_level", "estimated_time"]
                    missing_step_fields = [field for field in step_fields if field not in first_step]
                    if missing_step_fields:
                        print(f"⚠️  Missing step fields: {missing_step_fields}")
                    else:
                        print(f"✅ Step structure complete")
        
        # Test 2: Employment Law Issue
        test_data = {
            "legal_issue": "I believe I'm being discriminated against at work based on my age. My employer has been giving younger employees better assignments and I was passed over for a promotion.",
            "user_sophistication_level": "general_consumer",
            "user_goals": [
                "Understand if this constitutes discrimination",
                "Learn about documentation requirements",
                "Explore filing options with EEOC"
            ],
            "jurisdiction": "US"
        }
        
        success, response = self.run_test(
            "Interactive Guidance - Employment Discrimination",
            "POST",
            "ux/interactive-guidance",
            200,
            test_data
        )
        
        # Test 3: Business Formation Guidance
        test_data = {
            "legal_issue": "I want to start a technology consulting business and need to choose the right business structure, understand tax implications, and ensure proper legal compliance.",
            "user_sophistication_level": "business_executive",
            "user_goals": [
                "Choose optimal business structure",
                "Understand tax implications",
                "Ensure regulatory compliance",
                "Protect personal assets"
            ],
            "jurisdiction": "US"
        }
        
        success, response = self.run_test(
            "Interactive Guidance - Business Formation",
            "POST",
            "ux/interactive-guidance",
            200,
            test_data
        )

    def test_personalized_recommendations_endpoint(self):
        """Test GET /api/ux/personalized-recommendations - Generate personalized legal insights"""
        print("\n" + "="*80)
        print("🎯 TESTING PERSONALIZED RECOMMENDATIONS ENDPOINT")
        print("="*80)
        
        # Test 1: Active User with Contract History
        params = {
            "user_id": "user_12345",
            "legal_history": json.dumps([
                {"type": "contract_creation", "contract_type": "NDA", "date": "2024-01-15"},
                {"type": "contract_creation", "contract_type": "employment_agreement", "date": "2024-02-20"},
                {"type": "legal_research", "topic": "intellectual_property", "date": "2024-03-10"},
                {"type": "compliance_check", "area": "employment_law", "date": "2024-03-25"},
                {"type": "contract_analysis", "contract_type": "partnership_agreement", "date": "2024-04-05"}
            ]),
            "industry": "technology",
            "jurisdiction": "US",
            "interests": json.dumps([
                "intellectual_property",
                "employment_law",
                "contract_automation",
                "compliance_monitoring"
            ])
        }
        
        success, response = self.run_test(
            "Personalized Recommendations - Active Tech User",
            "GET",
            "ux/personalized-recommendations",
            200,
            params=params
        )
        
        if success and response:
            required_fields = ["user_profile_summary", "recommendations", "total_recommendations", 
                             "personalization_factors", "generated_at"]
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
            else:
                print(f"✅ All required response fields present")
                print(f"   Total Recommendations: {response.get('total_recommendations')}")
                print(f"   Personalization Factors: {len(response.get('personalization_factors', []))}")
                
                # Verify recommendations structure
                recommendations = response.get('recommendations', [])
                if recommendations and len(recommendations) > 0:
                    first_rec = recommendations[0]
                    rec_fields = ["recommendation_type", "title", "description", "relevance_score", 
                                "action_url", "priority", "estimated_value"]
                    missing_rec_fields = [field for field in rec_fields if field not in first_rec]
                    if missing_rec_fields:
                        print(f"⚠️  Missing recommendation fields: {missing_rec_fields}")
                    else:
                        print(f"✅ Recommendation structure complete")
                        print(f"   First Recommendation: {first_rec.get('title')}")
                        print(f"   Relevance Score: {first_rec.get('relevance_score')}")
        
        # Test 2: New User with Minimal History
        params = {
            "user_id": "new_user_789",
            "legal_history": json.dumps([]),
            "industry": "healthcare",
            "jurisdiction": "US",
            "interests": json.dumps([
                "compliance",
                "patient_privacy"
            ])
        }
        
        success, response = self.run_test(
            "Personalized Recommendations - New Healthcare User",
            "GET",
            "ux/personalized-recommendations",
            200,
            params=params
        )
        
        # Test 3: Anonymous User
        params = {
            "legal_history": json.dumps([
                {"type": "legal_research", "topic": "real_estate", "date": "2024-04-01"}
            ]),
            "industry": "real_estate",
            "jurisdiction": "US",
            "interests": json.dumps([
                "property_law",
                "lease_agreements"
            ])
        }
        
        success, response = self.run_test(
            "Personalized Recommendations - Anonymous Real Estate User",
            "GET",
            "ux/personalized-recommendations",
            200,
            params=params
        )

    def test_document_analysis_endpoint(self):
        """Test POST /api/ux/document-analysis - Analyze uploaded legal documents"""
        print("\n" + "="*80)
        print("📄 TESTING DOCUMENT ANALYSIS ENDPOINT")
        print("="*80)
        
        # Test 1: Standard Contract Analysis
        sample_contract = """
        INDEPENDENT CONTRACTOR AGREEMENT
        
        This Agreement is entered into on [Date] between ABC Company ("Company") and John Smith ("Contractor").
        
        1. SERVICES: Contractor agrees to provide web development services including website design, 
        development, and maintenance for the Company's e-commerce platform.
        
        2. COMPENSATION: Company agrees to pay Contractor $5,000 upon completion of the project, 
        with 50% due upon signing and 50% due upon delivery.
        
        3. TERM: This agreement shall commence on [Start Date] and continue until completion 
        of the services, estimated to be 3 months.
        
        4. INTELLECTUAL PROPERTY: All work product created by Contractor shall become the 
        exclusive property of Company.
        
        5. CONFIDENTIALITY: Contractor agrees to maintain confidentiality of all Company 
        information and trade secrets.
        
        6. TERMINATION: Either party may terminate this agreement with 30 days written notice.
        
        7. GOVERNING LAW: This agreement shall be governed by the laws of [State].
        """
        
        test_data = {
            "document_content": sample_contract,
            "document_type": "independent_contractor_agreement",
            "analysis_depth": "standard",
            "user_sophistication_level": "general_consumer"
        }
        
        success, response = self.run_test(
            "Document Analysis - Standard Contract",
            "POST",
            "ux/document-analysis",
            200,
            test_data
        )
        
        if success and response:
            required_fields = ["analysis_id", "document_summary", "key_findings", 
                             "legal_issues_identified", "recommendations", "complexity_assessment",
                             "requires_professional_review", "confidence_score"]
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
            else:
                print(f"✅ All required response fields present")
                print(f"   Complexity Assessment: {response.get('complexity_assessment')}")
                print(f"   Professional Review Required: {response.get('requires_professional_review')}")
                print(f"   Confidence Score: {response.get('confidence_score')}")
                print(f"   Key Findings: {len(response.get('key_findings', []))}")
                print(f"   Legal Issues: {len(response.get('legal_issues_identified', []))}")
        
        # Test 2: Complex Legal Document Analysis
        complex_document = """
        MERGER AGREEMENT
        
        WHEREAS, the parties hereto desire to effect a merger pursuant to the provisions of 
        Section 251 of the General Corporation Law of the State of Delaware;
        
        WHEREAS, the Board of Directors of each of Company A and Company B has approved 
        this Agreement and the transactions contemplated hereby;
        
        NOW, THEREFORE, in consideration of the mutual covenants and agreements contained 
        herein, and for other good and valuable consideration, the receipt and sufficiency 
        of which are hereby acknowledged, the parties agree as follows:
        
        1. THE MERGER: Subject to the terms and conditions of this Agreement, at the Effective Time, 
        Company B shall be merged with and into Company A, with Company A continuing as the 
        surviving corporation.
        
        2. CONSIDERATION: Each share of Company B common stock shall be converted into the right 
        to receive $50.00 in cash, without interest, subject to any required withholding taxes.
        
        3. REPRESENTATIONS AND WARRANTIES: Each party represents and warrants that it has full 
        corporate power and authority to execute and deliver this Agreement and to consummate 
        the transactions contemplated hereby.
        
        4. INDEMNIFICATION: Company A agrees to indemnify and hold harmless the former directors 
        and officers of Company B from and against any costs or expenses incurred in connection 
        with any claim, action, suit, proceeding or investigation.
        """
        
        test_data = {
            "document_content": complex_document,
            "document_type": "merger_agreement",
            "analysis_depth": "comprehensive",
            "user_sophistication_level": "business_executive"
        }
        
        success, response = self.run_test(
            "Document Analysis - Complex Merger Agreement",
            "POST",
            "ux/document-analysis",
            200,
            test_data
        )
        
        # Test 3: Basic Document Analysis
        simple_document = """
        RENTAL AGREEMENT
        
        This rental agreement is between Jane Doe (Tenant) and ABC Properties (Landlord) 
        for the property located at 123 Main Street, Anytown, State.
        
        Monthly Rent: $1,200
        Security Deposit: $1,200
        Lease Term: 12 months starting January 1, 2024
        
        Tenant agrees to pay rent on the 1st of each month.
        Landlord agrees to maintain the property in good condition.
        No pets allowed without written permission.
        """
        
        test_data = {
            "document_content": simple_document,
            "document_type": "rental_agreement",
            "analysis_depth": "basic",
            "user_sophistication_level": "general_consumer"
        }
        
        success, response = self.run_test(
            "Document Analysis - Basic Rental Agreement",
            "POST",
            "ux/document-analysis",
            200,
            test_data
        )

    def test_legal_workflow_templates_endpoint(self):
        """Test GET /api/ux/legal-workflow-templates - Provide professional legal templates"""
        print("\n" + "="*80)
        print("📋 TESTING LEGAL WORKFLOW TEMPLATES ENDPOINT")
        print("="*80)
        
        # Test 1: All Templates
        success, response = self.run_test(
            "Workflow Templates - All Templates",
            "GET",
            "ux/legal-workflow-templates",
            200
        )
        
        if success and response:
            required_fields = ["templates", "categories", "total_templates", "filtered_by"]
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
            else:
                print(f"✅ All required response fields present")
                print(f"   Total Templates: {response.get('total_templates')}")
                print(f"   Categories: {response.get('categories')}")
                
                # Verify template structure
                templates = response.get('templates', [])
                if templates and len(templates) > 0:
                    first_template = templates[0]
                    template_fields = ["template_id", "name", "description", "category", 
                                     "steps", "estimated_time", "skill_level_required", "tools_needed"]
                    missing_template_fields = [field for field in template_fields if field not in first_template]
                    if missing_template_fields:
                        print(f"⚠️  Missing template fields: {missing_template_fields}")
                    else:
                        print(f"✅ Template structure complete")
                        print(f"   First Template: {first_template.get('name')}")
                        print(f"   Steps: {len(first_template.get('steps', []))}")
        
        # Test 2: Filter by Category
        params = {
            "category": "contract_creation",
            "jurisdiction": "US"
        }
        
        success, response = self.run_test(
            "Workflow Templates - Contract Creation Category",
            "GET",
            "ux/legal-workflow-templates",
            200,
            params=params
        )
        
        if success and response:
            print(f"   Filtered Templates: {response.get('total_templates')}")
            print(f"   Filter Applied: {response.get('filtered_by')}")
        
        # Test 3: Filter by Skill Level
        params = {
            "skill_level": "beginner",
            "jurisdiction": "US"
        }
        
        success, response = self.run_test(
            "Workflow Templates - Beginner Skill Level",
            "GET",
            "ux/legal-workflow-templates",
            200,
            params=params
        )
        
        # Test 4: Multiple Filters
        params = {
            "category": "legal_research",
            "skill_level": "intermediate",
            "jurisdiction": "US"
        }
        
        success, response = self.run_test(
            "Workflow Templates - Legal Research + Intermediate",
            "GET",
            "ux/legal-workflow-templates",
            200,
            params=params
        )

    def run_comprehensive_ux_tests(self):
        """Run all Advanced UX API endpoint tests"""
        print("🚀 STARTING ADVANCED USER EXPERIENCE API TESTING SUITE")
        print("="*80)
        print(f"Testing against: {self.base_url}")
        print(f"API Base URL: {self.api_url}")
        
        # Test all 6 UX endpoints
        self.test_adaptive_response_endpoint()
        self.test_user_sophistication_analysis_endpoint()
        self.test_interactive_guidance_endpoint()
        self.test_personalized_recommendations_endpoint()
        self.test_document_analysis_endpoint()
        self.test_legal_workflow_templates_endpoint()
        
        # Print final results
        print("\n" + "="*80)
        print("🏁 ADVANCED UX API TESTING COMPLETE")
        print("="*80)
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        # Detailed results summary
        print("\n📊 DETAILED RESULTS BY ENDPOINT:")
        endpoint_results = {}
        for result in self.test_results:
            endpoint = result["test"].split(" - ")[0] if " - " in result["test"] else result["test"]
            if endpoint not in endpoint_results:
                endpoint_results[endpoint] = {"passed": 0, "failed": 0, "total": 0}
            endpoint_results[endpoint]["total"] += 1
            if result["status"] == "PASSED":
                endpoint_results[endpoint]["passed"] += 1
            else:
                endpoint_results[endpoint]["failed"] += 1
        
        for endpoint, stats in endpoint_results.items():
            success_rate = (stats["passed"] / stats["total"]) * 100
            status_icon = "✅" if success_rate == 100 else "⚠️" if success_rate >= 50 else "❌"
            print(f"{status_icon} {endpoint}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
        
        return self.tests_passed == self.tests_run

if __name__ == "__main__":
    tester = AdvancedUXAPITester()
    success = tester.run_comprehensive_ux_tests()
    sys.exit(0 if success else 1)