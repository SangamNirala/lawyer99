#!/usr/bin/env python3
"""
Focused Smart Contract Analysis Testing
Tests only the new Smart Contract Analysis endpoints
"""

import requests
import json
import sys

class SmartContractAnalysisTest:
    def __init__(self, base_url="https://7eb15b8a-98e2-46d7-83f4-aba058ad34e0.preview.emergentagent.com"):
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

    def test_contract_types_enhanced(self):
        """Test enhanced contract types endpoint - should return 56 types"""
        success, response = self.run_test("Enhanced Contract Types (56 types)", "GET", "contract-types", 200)
        if success and 'types' in response:
            types = response['types']
            total_count = response.get('total_count', len(types))
            categories = response.get('categories', [])
            
            print(f"   Found {len(types)} contract types (expected 56)")
            print(f"   Total count reported: {total_count}")
            print(f"   Categories: {len(categories)} categories")
            
            if len(types) >= 50:  # Should be around 56
                print(f"   ✅ Contract types count meets expectation (50+)")
            else:
                print(f"   ❌ Expected 56+ contract types, found {len(types)}")
            
            # Check for key contract types
            type_ids = [t.get('id') for t in types]
            expected_types = ['NDA', 'employment_agreement', 'freelance_agreement', 'partnership_agreement', 
                            'purchase_agreement', 'lease_agreement', 'software_license', 'consulting_agreement']
            missing_types = [t for t in expected_types if t not in type_ids]
            
            if not missing_types:
                print(f"   ✅ All key contract types found")
            else:
                print(f"   ⚠️  Missing some expected types: {missing_types}")
                
        return success, response

    def test_jurisdictions_enhanced(self):
        """Test enhanced jurisdictions endpoint - should return 10 jurisdictions"""
        success, response = self.run_test("Enhanced Jurisdictions (10 jurisdictions)", "GET", "jurisdictions", 200)
        if success and 'jurisdictions' in response:
            jurisdictions = response['jurisdictions']
            supported = [j for j in jurisdictions if j.get('supported', False)]
            
            print(f"   Found {len(jurisdictions)} jurisdictions")
            print(f"   Supported jurisdictions: {len(supported)}")
            
            if len(jurisdictions) >= 10:
                print(f"   ✅ Jurisdictions count meets expectation (10+)")
            else:
                print(f"   ❌ Expected 10+ jurisdictions, found {len(jurisdictions)}")
            
            # Check for key jurisdictions
            jurisdiction_codes = [j.get('code') for j in jurisdictions]
            expected_codes = ['US', 'UK', 'EU', 'CA', 'AU']
            missing_codes = [c for c in expected_codes if c not in jurisdiction_codes]
            
            if not missing_codes:
                print(f"   ✅ All key jurisdictions found")
            else:
                print(f"   ⚠️  Missing some expected jurisdictions: {missing_codes}")
                
            # Show supported jurisdictions
            supported_names = [j.get('name') for j in supported]
            print(f"   Supported: {', '.join(supported_names[:5])}{'...' if len(supported_names) > 5 else ''}")
                
        return success, response

    def test_contract_analysis(self):
        """Test AI-powered contract analysis endpoint"""
        sample_contract = """
        NON-DISCLOSURE AGREEMENT
        
        This Non-Disclosure Agreement is entered into between TechCorp Inc. and John Doe.
        
        1. CONFIDENTIAL INFORMATION
        The parties agree to maintain confidentiality of all proprietary information shared.
        
        2. PERMITTED USES
        Confidential information may only be used for evaluation purposes.
        
        3. TERM
        This agreement shall remain in effect for 2 years from the date of execution.
        
        4. GOVERNING LAW
        This agreement shall be governed by the laws of California.
        """
        
        analysis_request = {
            "contract_content": sample_contract,
            "contract_type": "NDA",
            "jurisdiction": "US"
        }
        
        success, response = self.run_test(
            "Contract Analysis with Sample NDA", 
            "POST", 
            "analyze-contract", 
            200, 
            analysis_request,
            timeout=60  # AI analysis might take longer
        )
        
        if success and response:
            print(f"   Analysis ID: {response.get('id', 'N/A')}")
            
            # Check risk assessment
            risk_assessment = response.get('risk_assessment', {})
            if risk_assessment:
                risk_score = risk_assessment.get('risk_score', 0)
                risk_level = risk_assessment.get('risk_level', 'UNKNOWN')
                risk_factors = risk_assessment.get('risk_factors', [])
                recommendations = risk_assessment.get('recommendations', [])
                
                print(f"   Risk Score: {risk_score}/100")
                print(f"   Risk Level: {risk_level}")
                print(f"   Risk Factors: {len(risk_factors)}")
                print(f"   Recommendations: {len(recommendations)}")
                
                if 0 <= risk_score <= 100:
                    print(f"   ✅ Valid risk score range")
                else:
                    print(f"   ❌ Invalid risk score: {risk_score}")
            
            # Check clause recommendations
            clause_recommendations = response.get('clause_recommendations', [])
            print(f"   Clause Recommendations: {len(clause_recommendations)}")
            
            # Check compliance issues
            compliance_issues = response.get('compliance_issues', [])
            print(f"   Compliance Issues: {len(compliance_issues)}")
            
            # Check readability and completeness scores
            readability_score = response.get('readability_score', 0)
            completeness_score = response.get('completeness_score', 0)
            print(f"   Readability Score: {readability_score}/100")
            print(f"   Completeness Score: {completeness_score}/100")
            
            if readability_score > 0 and completeness_score > 0:
                print(f"   ✅ Analysis scores generated successfully")
            else:
                print(f"   ⚠️  Analysis scores may be missing or zero")
                
        return success, response

    def test_clause_recommendations(self):
        """Test clause recommendations for different contract types"""
        contract_types_to_test = ['NDA', 'employment_agreement', 'freelance_agreement', 'partnership_agreement']
        
        all_success = True
        results = {}
        
        for contract_type in contract_types_to_test:
            success, response = self.run_test(
                f"Clause Recommendations for {contract_type}", 
                "GET", 
                f"clause-recommendations/{contract_type}?industry=Technology&jurisdiction=US", 
                200,
                timeout=45
            )
            
            if success and 'recommendations' in response:
                recommendations = response['recommendations']
                print(f"   {contract_type}: {len(recommendations)} recommendations")
                
                # Check recommendation structure
                if recommendations:
                    first_rec = recommendations[0]
                    required_fields = ['clause_type', 'title', 'content', 'priority', 'reasoning']
                    missing_fields = [field for field in required_fields if field not in first_rec]
                    
                    if not missing_fields:
                        print(f"   ✅ Recommendation structure valid")
                    else:
                        print(f"   ❌ Missing fields in recommendation: {missing_fields}")
                        all_success = False
                
                results[contract_type] = len(recommendations)
            else:
                all_success = False
                results[contract_type] = 0
        
        print(f"   Summary: {results}")
        return all_success, results

    def test_contract_comparison(self):
        """Test AI-powered contract comparison"""
        contract1 = """
        FREELANCE AGREEMENT
        
        This agreement is between Client Corp and Freelancer John.
        
        1. SCOPE OF WORK
        Developer will create a website with 5 pages.
        
        2. PAYMENT
        Total payment: $5,000 paid in 2 milestones.
        
        3. TIMELINE
        Project completion: 30 days from start date.
        """
        
        contract2 = """
        FREELANCE AGREEMENT
        
        This agreement is between Client Corp and Freelancer John.
        
        1. SCOPE OF WORK
        Developer will create a website with 10 pages and e-commerce functionality.
        
        2. PAYMENT
        Total payment: $8,000 paid in 3 milestones.
        
        3. TIMELINE
        Project completion: 45 days from start date.
        
        4. REVISIONS
        Up to 3 rounds of revisions included.
        """
        
        comparison_request = {
            "contract1_content": contract1,
            "contract2_content": contract2,
            "contract1_label": "Original Contract",
            "contract2_label": "Updated Contract"
        }
        
        success, response = self.run_test(
            "Contract Comparison Analysis", 
            "POST", 
            "compare-contracts", 
            200, 
            comparison_request,
            timeout=60
        )
        
        if success and response:
            print(f"   Comparison ID: {response.get('id', 'N/A')}")
            
            # Check similarity score
            similarity_score = response.get('similarity_score', 0)
            print(f"   Similarity Score: {similarity_score:.1f}%")
            
            # Check differences
            differences = response.get('differences', [])
            print(f"   Differences Found: {len(differences)}")
            
            if differences:
                # Show types of differences
                diff_types = [d.get('type') for d in differences]
                type_counts = {t: diff_types.count(t) for t in set(diff_types)}
                print(f"   Difference Types: {type_counts}")
                
                # Check significance levels
                significance_levels = [d.get('significance') for d in differences]
                sig_counts = {s: significance_levels.count(s) for s in set(significance_levels)}
                print(f"   Significance Levels: {sig_counts}")
            
            # Check summary
            summary = response.get('summary', '')
            if summary:
                print(f"   Summary Length: {len(summary)} characters")
                print(f"   ✅ Comparison analysis completed successfully")
            else:
                print(f"   ⚠️  No summary provided in comparison")
                
        return success, response

    def run_all_tests(self):
        """Run all Smart Contract Analysis tests"""
        print("🧠 Starting Smart Contract Analysis Testing Suite...")
        print(f"   Base URL: {self.base_url}")
        print(f"   API URL: {self.api_url}")
        print("=" * 80)
        
        # Test enhanced contract types endpoint
        self.test_contract_types_enhanced()
        
        # Test enhanced jurisdictions endpoint
        self.test_jurisdictions_enhanced()
        
        # Test contract analysis endpoint
        self.test_contract_analysis()
        
        # Test clause recommendations endpoint
        self.test_clause_recommendations()
        
        # Test contract comparison endpoint
        self.test_contract_comparison()
        
        # Print final results
        print("\n" + "=" * 80)
        print("🏁 SMART CONTRACT ANALYSIS TEST RESULTS")
        print("=" * 80)
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("🎉 ALL SMART CONTRACT ANALYSIS TESTS PASSED! 🎉")
        else:
            print(f"⚠️  {self.tests_run - self.tests_passed} tests failed. Please review the output above.")
        
        print("=" * 80)
        return self.tests_passed == self.tests_run

def main():
    print("🚀 Starting Smart Contract Analysis Backend API Tests")
    print("=" * 60)
    
    tester = SmartContractAnalysisTest()
    success = tester.run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())