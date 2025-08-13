#!/usr/bin/env python3
"""
Legal Concept Understanding System - Comprehensive Backend Testing
Testing 6 new API endpoints for legal reasoning functionality
"""

import requests
import sys
import json
import time
from datetime import datetime

class LegalReasoningTester:
    def __init__(self, base_url="https://legal-engine-check.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

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
                    self.test_results.append({"name": name, "status": "PASSED", "response": response_data})
                    return True, response_data
                except:
                    self.test_results.append({"name": name, "status": "PASSED", "response": response.text})
                    return True, response.text
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                    self.test_results.append({"name": name, "status": "FAILED", "error": error_data})
                except:
                    print(f"   Error: {response.text}")
                    self.test_results.append({"name": name, "status": "FAILED", "error": response.text})
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timeout after {timeout} seconds")
            self.test_results.append({"name": name, "status": "TIMEOUT", "error": f"Timeout after {timeout}s"})
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Exception: {str(e)}")
            self.test_results.append({"name": name, "status": "ERROR", "error": str(e)})
            return False, {}

    def test_analyze_concepts_endpoint(self):
        """Test POST /api/legal-reasoning/analyze-concepts"""
        print("\n" + "="*80)
        print("🧠 TESTING LEGAL CONCEPT ANALYSIS ENDPOINT")
        print("="*80)
        
        # Test Case 1: Contract breach scenario
        test_data_1 = {
            "query": "I want to breach a contract by not paying the agreed amount",
            "context": {
                "contract_type": "service_agreement",
                "jurisdiction": "US"
            },
            "extract_relationships": True,
            "include_reasoning": True
        }
        
        success_1, response_1 = self.run_test(
            "Concept Analysis - Contract Breach",
            "POST",
            "legal-reasoning/analyze-concepts",
            200,
            test_data_1
        )
        
        if success_1:
            # Verify response structure
            required_fields = ["identified_concepts", "concept_relationships", "applicable_legal_standards", 
                             "confidence_scores", "reasoning_pathway", "jurisdiction_analysis"]
            missing_fields = [field for field in required_fields if field not in response_1]
            if missing_fields:
                print(f"⚠️  Missing response fields: {missing_fields}")
            else:
                print("✅ All required response fields present")
                
            # Check if legal concepts were identified
            if response_1.get("identified_concepts"):
                print(f"✅ Identified {len(response_1['identified_concepts'])} legal concepts")
            else:
                print("⚠️  No legal concepts identified")

        # Test Case 2: Constitutional law scenario
        test_data_2 = {
            "query": "Due process requires fair hearing before government action",
            "context": {
                "legal_domain": "constitutional_law",
                "jurisdiction": "US"
            },
            "extract_relationships": True,
            "include_reasoning": True
        }
        
        success_2, response_2 = self.run_test(
            "Concept Analysis - Constitutional Law",
            "POST",
            "legal-reasoning/analyze-concepts",
            200,
            test_data_2
        )

        # Test Case 3: IP law scenario
        test_data_3 = {
            "query": "Patent infringement occurs when someone uses my invention without permission",
            "context": {
                "legal_domain": "intellectual_property",
                "jurisdiction": "US"
            },
            "extract_relationships": True,
            "include_reasoning": True
        }
        
        success_3, response_3 = self.run_test(
            "Concept Analysis - IP Law",
            "POST",
            "legal-reasoning/analyze-concepts",
            200,
            test_data_3
        )

        return success_1 and success_2 and success_3

    def test_concept_relationships_endpoint(self):
        """Test GET /api/legal-reasoning/concept-relationships/{concept_id}"""
        print("\n" + "="*80)
        print("🔗 TESTING CONCEPT RELATIONSHIPS ENDPOINT")
        print("="*80)
        
        # Test common legal concepts
        test_concepts = ["offer", "acceptance", "consideration", "breach", "damages"]
        
        all_success = True
        for concept_id in test_concepts:
            success, response = self.run_test(
                f"Concept Relationships - {concept_id}",
                "GET",
                f"legal-reasoning/concept-relationships/{concept_id}?max_depth=2",
                200
            )
            
            if success:
                # Verify response structure
                required_fields = ["concept", "related_concepts", "hierarchy", "applicable_tests"]
                missing_fields = [field for field in required_fields if field not in response]
                if missing_fields:
                    print(f"⚠️  Missing response fields for {concept_id}: {missing_fields}")
                else:
                    print(f"✅ Complete relationship data for {concept_id}")
                    
                # Check concept details
                if "concept" in response and "concept_id" in response["concept"]:
                    print(f"✅ Concept ID: {response['concept']['concept_id']}")
                    print(f"✅ Domain: {response['concept'].get('domain', 'N/A')}")
                    print(f"✅ Jurisdictions: {response['concept'].get('jurisdictions', [])}")
            
            all_success = all_success and success
            
        # Test invalid concept
        success_invalid, response_invalid = self.run_test(
            "Concept Relationships - Invalid Concept",
            "GET",
            "legal-reasoning/concept-relationships/invalid_concept_xyz",
            404
        )
        
        return all_success

    def test_applicable_law_endpoint(self):
        """Test POST /api/legal-reasoning/applicable-law"""
        print("\n" + "="*80)
        print("⚖️  TESTING APPLICABLE LAW ENDPOINT")
        print("="*80)
        
        # Test Case 1: Contract formation concepts
        test_data_1 = {
            "concepts": ["offer", "acceptance", "consideration"],
            "jurisdiction": "US",
            "legal_domain": "contract_law",
            "scenario_context": "Formation of a service agreement between two parties"
        }
        
        success_1, response_1 = self.run_test(
            "Applicable Law - Contract Formation",
            "POST",
            "legal-reasoning/applicable-law",
            200,
            test_data_1
        )
        
        if success_1:
            # Verify response structure
            required_fields = ["applicable_laws", "legal_tests", "evidence_requirements", 
                             "burden_of_proof", "jurisdiction_specifics"]
            missing_fields = [field for field in required_fields if field not in response_1]
            if missing_fields:
                print(f"⚠️  Missing response fields: {missing_fields}")
            else:
                print("✅ All required response fields present")
                
            # Check applicable laws
            if response_1.get("applicable_laws"):
                print(f"✅ Found {len(response_1['applicable_laws'])} applicable laws")
            
            # Check legal tests
            if response_1.get("legal_tests"):
                print(f"✅ Found {len(response_1['legal_tests'])} legal tests")

        # Test Case 2: Different jurisdiction (UK)
        test_data_2 = {
            "concepts": ["breach", "damages"],
            "jurisdiction": "UK",
            "legal_domain": "contract_law",
            "scenario_context": "Contract breach and remedy analysis"
        }
        
        success_2, response_2 = self.run_test(
            "Applicable Law - UK Jurisdiction",
            "POST",
            "legal-reasoning/applicable-law",
            200,
            test_data_2
        )

        # Test Case 3: Constitutional concepts
        test_data_3 = {
            "concepts": ["due_process", "equal_protection"],
            "jurisdiction": "US",
            "legal_domain": "constitutional_law",
            "scenario_context": "Government action affecting individual rights"
        }
        
        success_3, response_3 = self.run_test(
            "Applicable Law - Constitutional",
            "POST",
            "legal-reasoning/applicable-law",
            200,
            test_data_3
        )

        return success_1 and success_2 and success_3

    def test_concept_hierarchy_endpoint(self):
        """Test GET /api/legal-reasoning/concept-hierarchy"""
        print("\n" + "="*80)
        print("🏗️  TESTING CONCEPT HIERARCHY ENDPOINT")
        print("="*80)
        
        # Test Case 1: Full hierarchy (no filters)
        success_1, response_1 = self.run_test(
            "Concept Hierarchy - Full",
            "GET",
            "legal-reasoning/concept-hierarchy",
            200
        )
        
        if success_1:
            # Verify response structure
            required_fields = ["ontology_statistics", "concept_hierarchy", "available_domains", 
                             "available_jurisdictions", "available_concept_types"]
            missing_fields = [field for field in required_fields if field not in response_1]
            if missing_fields:
                print(f"⚠️  Missing response fields: {missing_fields}")
            else:
                print("✅ All required response fields present")
                
            # Check statistics
            if "ontology_statistics" in response_1:
                stats = response_1["ontology_statistics"]
                print(f"✅ Ontology statistics available")
                if "concepts_by_domain" in stats:
                    print(f"✅ Concepts by domain: {len(stats['concepts_by_domain'])} domains")
                if "concepts_by_jurisdiction" in stats:
                    print(f"✅ Concepts by jurisdiction: {len(stats['concepts_by_jurisdiction'])} jurisdictions")

        # Test Case 2: Filter by domain (contract_law)
        success_2, response_2 = self.run_test(
            "Concept Hierarchy - Contract Law Domain",
            "GET",
            "legal-reasoning/concept-hierarchy?domain=contract_law",
            200
        )
        
        if success_2:
            if "concept_hierarchy" in response_2 and "contract_law" in response_2["concept_hierarchy"]:
                concepts = response_2["concept_hierarchy"]["contract_law"]
                print(f"✅ Found {len(concepts)} contract law concepts")

        # Test Case 3: Filter by jurisdiction (US)
        success_3, response_3 = self.run_test(
            "Concept Hierarchy - US Jurisdiction",
            "GET",
            "legal-reasoning/concept-hierarchy?jurisdiction=US",
            200
        )

        # Test Case 4: Invalid domain
        success_4, response_4 = self.run_test(
            "Concept Hierarchy - Invalid Domain",
            "GET",
            "legal-reasoning/concept-hierarchy?domain=invalid_domain",
            400
        )

        return success_1 and success_2 and success_3 and success_4

    def test_analyze_scenario_endpoint(self):
        """Test POST /api/legal-reasoning/analyze-scenario"""
        print("\n" + "="*80)
        print("📋 TESTING LEGAL SCENARIO ANALYSIS ENDPOINT")
        print("="*80)
        
        # Test Case 1: Contract formation scenario
        test_data_1 = {
            "facts": "Company A offered to provide web development services to Company B for $10,000. Company B accepted the offer and paid a $2,000 deposit. Company A then failed to deliver the services on time.",
            "parties": ["Company A", "Company B"],
            "jurisdiction": "US",
            "legal_domain": "contract_law",
            "issues": ["contract_formation", "breach", "damages"],
            "requested_analysis": ["formation_analysis", "breach_analysis"]
        }
        
        success_1, response_1 = self.run_test(
            "Scenario Analysis - Contract Formation",
            "POST",
            "legal-reasoning/analyze-scenario",
            200,
            test_data_1
        )
        
        if success_1:
            # Verify response structure
            required_fields = ["scenario_id", "identified_concepts", "concept_interactions", 
                             "applicable_laws", "reasoning_pathways", "legal_standards_applied",
                             "risk_assessment", "recommended_actions"]
            missing_fields = [field for field in required_fields if field not in response_1]
            if missing_fields:
                print(f"⚠️  Missing response fields: {missing_fields}")
            else:
                print("✅ All required response fields present")
                
            # Check scenario analysis results
            if response_1.get("identified_concepts"):
                print(f"✅ Identified {len(response_1['identified_concepts'])} concepts")
            
            if response_1.get("reasoning_pathways"):
                print(f"✅ Generated {len(response_1['reasoning_pathways'])} reasoning pathways")
            
            if response_1.get("recommended_actions"):
                print(f"✅ Provided {len(response_1['recommended_actions'])} recommended actions")

        # Test Case 2: Constitutional rights scenario
        test_data_2 = {
            "facts": "A government agency terminated an employee without providing a hearing or notice. The employee claims this violated their due process rights.",
            "parties": ["Government Agency", "Employee"],
            "jurisdiction": "US",
            "legal_domain": "constitutional_law",
            "issues": ["due_process", "procedural_fairness"],
            "requested_analysis": ["constitutional_analysis"]
        }
        
        success_2, response_2 = self.run_test(
            "Scenario Analysis - Constitutional Rights",
            "POST",
            "legal-reasoning/analyze-scenario",
            200,
            test_data_2
        )

        # Test Case 3: IP infringement scenario
        test_data_3 = {
            "facts": "Inventor A holds a patent for a unique software algorithm. Company B is using a similar algorithm in their product without permission from Inventor A.",
            "parties": ["Inventor A", "Company B"],
            "jurisdiction": "US",
            "legal_domain": "intellectual_property",
            "issues": ["patent_infringement", "damages"],
            "requested_analysis": ["infringement_analysis"]
        }
        
        success_3, response_3 = self.run_test(
            "Scenario Analysis - IP Infringement",
            "POST",
            "legal-reasoning/analyze-scenario",
            200,
            test_data_3
        )

        return success_1 and success_2 and success_3

    def test_search_concepts_endpoint(self):
        """Test GET /api/legal-reasoning/search-concepts"""
        print("\n" + "="*80)
        print("🔍 TESTING CONCEPT SEARCH ENDPOINT")
        print("="*80)
        
        # Test Case 1: Basic search
        success_1, response_1 = self.run_test(
            "Concept Search - Basic (contract)",
            "GET",
            "legal-reasoning/search-concepts?query=contract&limit=10",
            200
        )
        
        if success_1:
            # Verify response structure
            required_fields = ["query", "total_matches", "returned_results", "concepts", "search_filters"]
            missing_fields = [field for field in required_fields if field not in response_1]
            if missing_fields:
                print(f"⚠️  Missing response fields: {missing_fields}")
            else:
                print("✅ All required response fields present")
                
            # Check search results
            if response_1.get("concepts"):
                print(f"✅ Found {len(response_1['concepts'])} concepts")
                print(f"✅ Total matches: {response_1.get('total_matches', 0)}")
                
                # Check concept structure
                if response_1["concepts"]:
                    first_concept = response_1["concepts"][0]
                    concept_fields = ["concept_id", "name", "domain", "type", "definition", "relevance_score"]
                    missing_concept_fields = [field for field in concept_fields if field not in first_concept]
                    if missing_concept_fields:
                        print(f"⚠️  Missing concept fields: {missing_concept_fields}")
                    else:
                        print("✅ Complete concept structure")

        # Test Case 2: Domain-filtered search
        success_2, response_2 = self.run_test(
            "Concept Search - Domain Filter",
            "GET",
            "legal-reasoning/search-concepts?query=breach&domains=contract_law&limit=5",
            200
        )

        # Test Case 3: Jurisdiction-filtered search
        success_3, response_3 = self.run_test(
            "Concept Search - Jurisdiction Filter",
            "GET",
            "legal-reasoning/search-concepts?query=due process&jurisdictions=US&limit=5",
            200
        )

        # Test Case 4: Multiple filters
        success_4, response_4 = self.run_test(
            "Concept Search - Multiple Filters",
            "GET",
            "legal-reasoning/search-concepts?query=liability&domains=contract_law,tort_law&jurisdictions=US,UK&limit=15",
            200
        )

        # Test Case 5: Empty query (should still work)
        success_5, response_5 = self.run_test(
            "Concept Search - Empty Query",
            "GET",
            "legal-reasoning/search-concepts?query=&limit=5",
            200
        )

        return success_1 and success_2 and success_3 and success_4 and success_5

    def run_comprehensive_tests(self):
        """Run all legal reasoning endpoint tests"""
        print("🚀 STARTING COMPREHENSIVE LEGAL CONCEPT UNDERSTANDING SYSTEM TESTING")
        print("="*100)
        print(f"Testing against: {self.base_url}")
        print(f"API Base URL: {self.api_url}")
        print("="*100)
        
        start_time = time.time()
        
        # Test all 6 endpoints
        test_results = {
            "analyze_concepts": self.test_analyze_concepts_endpoint(),
            "concept_relationships": self.test_concept_relationships_endpoint(),
            "applicable_law": self.test_applicable_law_endpoint(),
            "concept_hierarchy": self.test_concept_hierarchy_endpoint(),
            "analyze_scenario": self.test_analyze_scenario_endpoint(),
            "search_concepts": self.test_search_concepts_endpoint()
        }
        
        end_time = time.time()
        
        # Print comprehensive summary
        print("\n" + "="*100)
        print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("="*100)
        
        print(f"⏱️  Total testing time: {end_time - start_time:.2f} seconds")
        print(f"🧪 Total tests run: {self.tests_run}")
        print(f"✅ Tests passed: {self.tests_passed}")
        print(f"❌ Tests failed: {self.tests_run - self.tests_passed}")
        print(f"📈 Success rate: {(self.tests_passed / self.tests_run * 100):.1f}%")
        
        print("\n🎯 ENDPOINT-SPECIFIC RESULTS:")
        for endpoint, success in test_results.items():
            status = "✅ WORKING" if success else "❌ FAILED"
            print(f"   {endpoint.replace('_', ' ').title()}: {status}")
        
        # Detailed test breakdown
        print("\n📋 DETAILED TEST BREAKDOWN:")
        for result in self.test_results:
            status_emoji = "✅" if result["status"] == "PASSED" else "❌"
            print(f"   {status_emoji} {result['name']}: {result['status']}")
        
        # Overall system assessment
        working_endpoints = sum(1 for success in test_results.values() if success)
        total_endpoints = len(test_results)
        
        print(f"\n🏆 LEGAL CONCEPT UNDERSTANDING SYSTEM STATUS:")
        print(f"   Working endpoints: {working_endpoints}/{total_endpoints}")
        
        if working_endpoints == total_endpoints:
            print("   🎉 ALL ENDPOINTS FULLY OPERATIONAL!")
            print("   ✅ Legal Concept Understanding System is ready for production")
        elif working_endpoints >= total_endpoints * 0.8:
            print("   ⚠️  MOSTLY OPERATIONAL - Minor issues detected")
            print("   🔧 Some endpoints may need attention")
        else:
            print("   ❌ SYSTEM ISSUES DETECTED")
            print("   🚨 Multiple endpoints require fixes")
        
        # AI Integration Assessment
        print(f"\n🤖 AI INTEGRATION ASSESSMENT:")
        print("   Expected AI Models: OpenAI GPT, Groq, Gemini")
        print("   Hybrid AI Approach: Concept extraction + Legal reasoning")
        print("   Legal Ontology: 500+ legal concepts across major domains")
        
        return working_endpoints == total_endpoints

if __name__ == "__main__":
    print("🧠 Legal Concept Understanding System - Backend Testing")
    print("Testing 6 new API endpoints for legal reasoning functionality")
    print("-" * 80)
    
    tester = LegalReasoningTester()
    success = tester.run_comprehensive_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED - Legal Concept Understanding System is fully operational!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - Check the detailed results above")
        sys.exit(1)