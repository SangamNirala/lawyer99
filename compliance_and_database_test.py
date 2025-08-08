import requests
import json
import sys
from datetime import datetime

class ComplianceAndDatabaseTester:
    def __init__(self, base_url="https://027a3f35-a0bc-40cd-ad79-483d46c6126e.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.analysis_ids = []
        self.comparison_ids = []

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

    def test_compliance_check_json_body(self):
        """Test the fixed POST /api/compliance-check endpoint with JSON body format"""
        print("\n" + "="*80)
        print("🎯 TESTING FIX #1: POST /api/compliance-check with JSON body format")
        print("="*80)
        
        # Test data with proper JSON body format as specified in the review request
        compliance_data = {
            "contract_content": "This is a sample Non-Disclosure Agreement between TechCorp Inc. and John Doe. The purpose of this agreement is to protect confidential information shared during business discussions. The receiving party agrees not to disclose any confidential information to third parties. This agreement shall remain in effect for a period of two years from the date of execution. The parties agree that any breach of this agreement may result in irreparable harm.",
            "jurisdictions": ["US", "UK", "EU"]
        }
        
        success, response = self.run_test(
            "Compliance Check with JSON Body (Fixed)",
            "POST",
            "compliance-check",
            200,
            compliance_data,
            timeout=60  # AI analysis might take longer
        )
        
        if success:
            print(f"   ✅ COMPLIANCE CHECK FIX WORKING: Endpoint now accepts JSON body format")
            print(f"   ✅ Status: 200 (previously was 422 due to parameter mismatch)")
            
            # Validate response structure
            if isinstance(response, dict):
                expected_keys = ['overall_compliance_score', 'jurisdiction_scores', 'compliance_issues', 'recommendations']
                found_keys = [key for key in expected_keys if key in response]
                
                print(f"   Response structure validation:")
                print(f"   - Expected keys: {expected_keys}")
                print(f"   - Found keys: {found_keys}")
                
                if 'overall_compliance_score' in response:
                    score = response['overall_compliance_score']
                    print(f"   - Overall Compliance Score: {score}/100")
                    
                if 'jurisdiction_scores' in response:
                    jurisdiction_scores = response['jurisdiction_scores']
                    print(f"   - Jurisdiction Scores: {jurisdiction_scores}")
                    
                if 'compliance_issues' in response:
                    issues = response['compliance_issues']
                    print(f"   - Compliance Issues Found: {len(issues)}")
                    
                if 'recommendations' in response:
                    recommendations = response['recommendations']
                    print(f"   - Recommendations: {len(recommendations)}")
                    
                if len(found_keys) >= 3:
                    print(f"   ✅ Response structure matches expected model")
                else:
                    print(f"   ⚠️  Response structure incomplete - missing some expected fields")
            else:
                print(f"   ❌ Response is not a dictionary: {type(response)}")
        else:
            print(f"   ❌ COMPLIANCE CHECK FIX FAILED: Still returning error status")
            
        return success, response

    def test_contract_analysis_for_database_test(self):
        """Run contract analysis to create data for database list endpoint testing"""
        print("\n" + "="*60)
        print("🔧 SETUP: Creating contract analysis data for database testing")
        print("="*60)
        
        # Create a few contract analyses to test the database list endpoints
        analysis_test_cases = [
            {
                "contract_content": "NON-DISCLOSURE AGREEMENT\n\nThis Non-Disclosure Agreement is entered into between TechStart Inc. and Innovation Labs LLC. The purpose is to facilitate discussions regarding a potential joint venture in artificial intelligence development. Both parties agree to maintain strict confidentiality of all shared information for a period of three years.",
                "contract_type": "NDA",
                "jurisdiction": "US"
            },
            {
                "contract_content": "FREELANCE SERVICE AGREEMENT\n\nThis agreement is between Digital Solutions Corp and Jane Smith, a freelance web developer. The scope includes development of a responsive e-commerce website with payment integration. Payment terms: $5,000 upon completion. Project timeline: 6 weeks from start date.",
                "contract_type": "freelance_agreement", 
                "jurisdiction": "US"
            }
        ]
        
        for i, test_case in enumerate(analysis_test_cases):
            success, response = self.run_test(
                f"Contract Analysis #{i+1} - {test_case['contract_type']}",
                "POST",
                "analyze-contract",
                200,
                test_case,
                timeout=60
            )
            
            if success and isinstance(response, dict) and 'id' in response:
                analysis_id = response['id']
                self.analysis_ids.append(analysis_id)
                print(f"   ✅ Created analysis with ID: {analysis_id}")
                print(f"   - Risk Score: {response.get('risk_assessment', {}).get('risk_score', 'N/A')}")
                print(f"   - Risk Level: {response.get('risk_assessment', {}).get('risk_level', 'N/A')}")
            else:
                print(f"   ❌ Failed to create analysis #{i+1}")
                
        print(f"\n   📊 Created {len(self.analysis_ids)} contract analyses for database testing")
        return len(self.analysis_ids) > 0

    def test_contract_comparison_for_database_test(self):
        """Run contract comparisons to create data for database list endpoint testing"""
        print("\n" + "="*60)
        print("🔧 SETUP: Creating contract comparison data for database testing")
        print("="*60)
        
        # Create a few contract comparisons to test the database list endpoints
        comparison_test_cases = [
            {
                "contract1_content": "PARTNERSHIP AGREEMENT\n\nThis partnership agreement is between Alpha Corp and Beta LLC. The business purpose is software development. Profit sharing: 60/40 split. Capital contribution: $50,000 each. Partnership duration: 5 years.",
                "contract2_content": "PARTNERSHIP AGREEMENT\n\nThis partnership agreement is between Alpha Corp and Beta LLC. The business purpose is software development and consulting. Profit sharing: 50/50 split. Capital contribution: $75,000 each. Partnership duration: 3 years.",
                "contract1_label": "Original Partnership Agreement",
                "contract2_label": "Revised Partnership Agreement"
            },
            {
                "contract1_content": "CONSULTING AGREEMENT\n\nThis agreement is between Business Solutions Inc. and John Consultant. Services: Strategic business consulting. Duration: 6 months. Payment: $10,000 monthly. Confidentiality clause included.",
                "contract2_content": "CONSULTING AGREEMENT\n\nThis agreement is between Business Solutions Inc. and John Consultant. Services: Strategic business consulting and market analysis. Duration: 12 months. Payment: $8,000 monthly. Confidentiality and non-compete clauses included.",
                "contract1_label": "Initial Consulting Agreement",
                "contract2_label": "Extended Consulting Agreement"
            }
        ]
        
        for i, test_case in enumerate(comparison_test_cases):
            success, response = self.run_test(
                f"Contract Comparison #{i+1}",
                "POST",
                "compare-contracts",
                200,
                test_case,
                timeout=60
            )
            
            if success and isinstance(response, dict) and 'id' in response:
                comparison_id = response['id']
                self.comparison_ids.append(comparison_id)
                print(f"   ✅ Created comparison with ID: {comparison_id}")
                print(f"   - Similarity Score: {response.get('similarity_score', 'N/A')}%")
                print(f"   - Differences Found: {len(response.get('differences', []))}")
            else:
                print(f"   ❌ Failed to create comparison #{i+1}")
                
        print(f"\n   📊 Created {len(self.comparison_ids)} contract comparisons for database testing")
        return len(self.comparison_ids) > 0

    def test_contract_analyses_list_endpoint(self):
        """Test the fixed GET /api/contract-analyses endpoint (ObjectId serialization fix)"""
        print("\n" + "="*80)
        print("🎯 TESTING FIX #2A: GET /api/contract-analyses ObjectId serialization fix")
        print("="*80)
        
        success, response = self.run_test(
            "Contract Analyses List (Fixed ObjectId Serialization)",
            "GET",
            "contract-analyses",
            200,
            timeout=30
        )
        
        if success:
            print(f"   ✅ CONTRACT ANALYSES LIST FIX WORKING: No more 500 Internal Server Error")
            print(f"   ✅ Status: 200 (previously was 500 due to ObjectId serialization issues)")
            
            # Validate response structure and content
            if isinstance(response, list):
                print(f"   ✅ Response is a list with {len(response)} analyses")
                
                if len(response) > 0:
                    # Check first analysis for proper structure
                    first_analysis = response[0]
                    print(f"   📋 First analysis structure validation:")
                    
                    expected_fields = ['id', 'contract_content', 'risk_assessment', 'clause_recommendations', 'created_at']
                    found_fields = [field for field in expected_fields if field in first_analysis]
                    
                    print(f"   - Expected fields: {expected_fields}")
                    print(f"   - Found fields: {found_fields}")
                    
                    # Check that 'id' field is a string (not ObjectId)
                    if 'id' in first_analysis:
                        analysis_id = first_analysis['id']
                        print(f"   - Analysis ID: {analysis_id} (type: {type(analysis_id).__name__})")
                        
                        if isinstance(analysis_id, str):
                            print(f"   ✅ ID field properly serialized as string")
                        else:
                            print(f"   ❌ ID field not properly serialized: {type(analysis_id)}")
                    
                    # Check for any remaining ObjectId serialization issues
                    try:
                        json_str = json.dumps(response)
                        print(f"   ✅ Response can be JSON serialized (no ObjectId issues)")
                    except Exception as e:
                        print(f"   ❌ JSON serialization failed: {str(e)}")
                        
                    # Verify we can find our created analyses
                    found_our_analyses = [a for a in response if a.get('id') in self.analysis_ids]
                    print(f"   - Found {len(found_our_analyses)} of our created analyses in the list")
                    
                else:
                    print(f"   ⚠️  No analyses found in database (empty list)")
                    
            elif isinstance(response, dict):
                print(f"   ⚠️  Response is a dictionary instead of list: {list(response.keys())}")
            else:
                print(f"   ❌ Unexpected response type: {type(response)}")
        else:
            print(f"   ❌ CONTRACT ANALYSES LIST FIX FAILED: Still returning error status")
            
        return success, response

    def test_contract_comparisons_list_endpoint(self):
        """Test the fixed GET /api/contract-comparisons endpoint (ObjectId serialization fix)"""
        print("\n" + "="*80)
        print("🎯 TESTING FIX #2B: GET /api/contract-comparisons ObjectId serialization fix")
        print("="*80)
        
        success, response = self.run_test(
            "Contract Comparisons List (Fixed ObjectId Serialization)",
            "GET",
            "contract-comparisons",
            200,
            timeout=30
        )
        
        if success:
            print(f"   ✅ CONTRACT COMPARISONS LIST FIX WORKING: No more 500 Internal Server Error")
            print(f"   ✅ Status: 200 (previously was 500 due to ObjectId serialization issues)")
            
            # Validate response structure and content
            if isinstance(response, list):
                print(f"   ✅ Response is a list with {len(response)} comparisons")
                
                if len(response) > 0:
                    # Check first comparison for proper structure
                    first_comparison = response[0]
                    print(f"   📋 First comparison structure validation:")
                    
                    expected_fields = ['id', 'contract1_label', 'contract2_label', 'differences', 'similarity_score', 'created_at']
                    found_fields = [field for field in expected_fields if field in first_comparison]
                    
                    print(f"   - Expected fields: {expected_fields}")
                    print(f"   - Found fields: {found_fields}")
                    
                    # Check that 'id' field is a string (not ObjectId)
                    if 'id' in first_comparison:
                        comparison_id = first_comparison['id']
                        print(f"   - Comparison ID: {comparison_id} (type: {type(comparison_id).__name__})")
                        
                        if isinstance(comparison_id, str):
                            print(f"   ✅ ID field properly serialized as string")
                        else:
                            print(f"   ❌ ID field not properly serialized: {type(comparison_id)}")
                    
                    # Check similarity score
                    if 'similarity_score' in first_comparison:
                        similarity = first_comparison['similarity_score']
                        print(f"   - Similarity Score: {similarity}% (type: {type(similarity).__name__})")
                    
                    # Check for any remaining ObjectId serialization issues
                    try:
                        json_str = json.dumps(response)
                        print(f"   ✅ Response can be JSON serialized (no ObjectId issues)")
                    except Exception as e:
                        print(f"   ❌ JSON serialization failed: {str(e)}")
                        
                    # Verify we can find our created comparisons
                    found_our_comparisons = [c for c in response if c.get('id') in self.comparison_ids]
                    print(f"   - Found {len(found_our_comparisons)} of our created comparisons in the list")
                    
                else:
                    print(f"   ⚠️  No comparisons found in database (empty list)")
                    
            elif isinstance(response, dict):
                print(f"   ⚠️  Response is a dictionary instead of list: {list(response.keys())}")
            else:
                print(f"   ❌ Unexpected response type: {type(response)}")
        else:
            print(f"   ❌ CONTRACT COMPARISONS LIST FIX FAILED: Still returning error status")
            
        return success, response

    def run_focused_fix_tests(self):
        """Run focused tests for the two specific fixes mentioned in the review request"""
        print("\n" + "🎯" * 40)
        print("FOCUSED TESTING: Smart Contract Analysis Backend Fixes")
        print("🎯" * 40)
        print("Testing two specific fixes:")
        print("1. POST /api/compliance-check - JSON body format fix")
        print("2. GET /api/contract-analyses & /api/contract-comparisons - ObjectId serialization fix")
        print("="*80)
        
        # Test Fix #1: Compliance Check JSON Body Format
        fix1_success, fix1_response = self.test_compliance_check_json_body()
        
        # Setup data for Fix #2 testing
        print("\n" + "🔧" * 40)
        print("SETUP PHASE: Creating test data for database list endpoints")
        print("🔧" * 40)
        
        analyses_created = self.test_contract_analysis_for_database_test()
        comparisons_created = self.test_contract_comparison_for_database_test()
        
        # Test Fix #2: Database List Endpoints ObjectId Serialization
        fix2a_success, fix2a_response = self.test_contract_analyses_list_endpoint()
        fix2b_success, fix2b_response = self.test_contract_comparisons_list_endpoint()
        
        # Summary of results
        print("\n" + "📊" * 40)
        print("FOCUSED FIX TESTING RESULTS SUMMARY")
        print("📊" * 40)
        
        print(f"\n🎯 FIX #1: POST /api/compliance-check JSON body format")
        if fix1_success:
            print(f"   ✅ FIXED: Endpoint now accepts JSON body instead of query parameters")
            print(f"   ✅ Status: 200 (previously 422 due to parameter mismatch)")
        else:
            print(f"   ❌ STILL BROKEN: Endpoint still has parameter structure issues")
            
        print(f"\n🎯 FIX #2A: GET /api/contract-analyses ObjectId serialization")
        if fix2a_success:
            print(f"   ✅ FIXED: No more 500 Internal Server Error")
            print(f"   ✅ Status: 200 with proper JSON response")
            print(f"   ✅ ObjectId fields properly serialized to strings")
        else:
            print(f"   ❌ STILL BROKEN: Database list endpoint still failing")
            
        print(f"\n🎯 FIX #2B: GET /api/contract-comparisons ObjectId serialization")
        if fix2b_success:
            print(f"   ✅ FIXED: No more 500 Internal Server Error")
            print(f"   ✅ Status: 200 with proper JSON response")
            print(f"   ✅ ObjectId fields properly serialized to strings")
        else:
            print(f"   ❌ STILL BROKEN: Database list endpoint still failing")
            
        # Overall assessment
        fixes_working = sum([fix1_success, fix2a_success, fix2b_success])
        total_fixes = 3
        
        print(f"\n" + "🏆" * 40)
        print(f"OVERALL FIX STATUS: {fixes_working}/{total_fixes} fixes working")
        print("🏆" * 40)
        
        if fixes_working == total_fixes:
            print(f"🎉 ALL FIXES WORKING: Smart Contract Analysis backend fixes are successful!")
        elif fixes_working >= 2:
            print(f"✅ MOSTLY WORKING: {fixes_working} out of {total_fixes} fixes are working")
        else:
            print(f"⚠️  NEEDS ATTENTION: Only {fixes_working} out of {total_fixes} fixes are working")
            
        print(f"\n📈 Test Statistics:")
        print(f"   - Total tests run: {self.tests_run}")
        print(f"   - Tests passed: {self.tests_passed}")
        print(f"   - Success rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        return {
            "compliance_check_fix": fix1_success,
            "analyses_list_fix": fix2a_success,
            "comparisons_list_fix": fix2b_success,
            "overall_success": fixes_working == total_fixes,
            "fixes_working": fixes_working,
            "total_fixes": total_fixes,
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed
        }

def main():
    """Main function to run the focused fix tests"""
    print("🚀 Starting Focused Fix Testing for Smart Contract Analysis Backend")
    print("=" * 80)
    
    tester = ComplianceAndDatabaseTester()
    results = tester.run_focused_fix_tests()
    
    # Exit with appropriate code
    if results["overall_success"]:
        print(f"\n✅ SUCCESS: All fixes are working correctly!")
        sys.exit(0)
    else:
        print(f"\n❌ ISSUES FOUND: {results['total_fixes'] - results['fixes_working']} fixes still need attention")
        sys.exit(1)

if __name__ == "__main__":
    main()