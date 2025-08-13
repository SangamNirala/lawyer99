import requests
import json
import re
import time
from datetime import datetime

class CompliantContractTester:
    def __init__(self, base_url="https://legalengine.preview.emergentagent.com"):
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

    def test_compliant_contract_generation_simple(self):
        """Test POST /api/generate-contract-compliant endpoint with a simple contract request"""
        print("\n" + "="*80)
        print("🎯 TESTING COMPLIANT CONTRACT GENERATION ENDPOINT")
        print("="*80)
        
        # Simple contract request as specified in the requirements
        contract_data = {
            "contract_type": "NDA",
            "jurisdiction": "US",
            "parties": {
                "party1_name": "TechCorp Solutions Inc.",
                "party1_type": "corporation",
                "party2_name": "Jane Developer",
                "party2_type": "individual"
            },
            "terms": {
                "purpose": "Discussion of potential software development collaboration and sharing of proprietary technical information",
                "duration": "2_years"
            },
            "special_clauses": ["Non-compete clause for 6 months"]
        }
        
        success, response = self.run_test(
            "Compliant Contract Generation - Simple NDA", 
            "POST", 
            "generate-contract-compliant", 
            200, 
            contract_data,
            timeout=90  # Compliance checking might take longer
        )
        
        if success and isinstance(response, dict):
            print(f"\n📋 RESPONSE STRUCTURE ANALYSIS:")
            print(f"   Response keys: {list(response.keys())}")
            
            # Check for contract in response
            if 'contract' in response:
                contract = response['contract']
                print(f"   Contract ID: {contract.get('id', 'N/A')}")
                print(f"   Contract Type: {contract.get('contract_type', 'N/A')}")
                print(f"   Compliance Score: {contract.get('compliance_score', 'N/A')}")
                print(f"   Content Length: {len(contract.get('content', ''))} characters")
            
            # CRITICAL: Examine the "suggestions" field carefully
            if 'suggestions' in response:
                suggestions = response['suggestions']
                print(f"\n🔍 SUGGESTIONS FIELD ANALYSIS:")
                print(f"   Suggestions type: {type(suggestions)}")
                print(f"   Suggestions count: {len(suggestions) if isinstance(suggestions, list) else 'N/A'}")
                
                if isinstance(suggestions, list):
                    for i, suggestion in enumerate(suggestions):
                        print(f"   Suggestion {i+1}: {suggestion}")
                        
                        # Check if review ID is included in expected format "review (ID: {review_id})"
                        review_id_match = re.search(r'review \(ID:\s*([^)]+)\)', suggestion)
                        if review_id_match:
                            review_id = review_id_match.group(1).strip()
                            print(f"   ✅ FOUND REVIEW ID: {review_id}")
                            
                            # Test if the review ID extraction regex pattern would work
                            regex_pattern = r'ID:\s*([^)]+)'
                            regex_match = re.search(regex_pattern, suggestion)
                            if regex_match:
                                extracted_id = regex_match.group(1).strip()
                                print(f"   ✅ REGEX EXTRACTION WORKS: {extracted_id}")
                                
                                # Verify that the review is actually created and can be fetched
                                return self.verify_review_status(extracted_id)
                            else:
                                print(f"   ❌ REGEX EXTRACTION FAILED for pattern: {regex_pattern}")
                        else:
                            print(f"   ⚠️  No review ID found in suggestion: {suggestion}")
                else:
                    print(f"   Suggestions content: {suggestions}")
            else:
                print(f"   ❌ NO 'suggestions' FIELD FOUND in response")
            
            # Check other response fields
            if 'warnings' in response:
                warnings = response['warnings']
                print(f"\n⚠️  WARNINGS ({len(warnings) if isinstance(warnings, list) else 'N/A'}):")
                if isinstance(warnings, list):
                    for warning in warnings:
                        print(f"   - {warning}")
            
            return success, response
        else:
            print(f"   ❌ Invalid response format or request failed")
            return False, {}

    def verify_review_status(self, review_id):
        """Verify that the review is actually created and can be fetched via GET /api/attorney/review/status/{review_id}"""
        print(f"\n🔍 VERIFYING REVIEW STATUS FOR ID: {review_id}")
        
        success, response = self.run_test(
            f"Review Status Verification - {review_id}", 
            "GET", 
            f"attorney/review/status/{review_id}", 
            200,
            timeout=30
        )
        
        if success and isinstance(response, dict):
            print(f"   ✅ REVIEW SUCCESSFULLY FETCHED")
            print(f"   Review ID: {response.get('review_id', 'N/A')}")
            print(f"   Status: {response.get('status', 'N/A')}")
            print(f"   Progress: {response.get('progress_percentage', 'N/A')}%")
            print(f"   Created At: {response.get('created_at', 'N/A')}")
            print(f"   Assigned Attorney: {response.get('assigned_attorney', 'N/A')}")
            print(f"   Priority: {response.get('priority', 'N/A')}")
            
            return True, response
        else:
            print(f"   ❌ FAILED TO FETCH REVIEW STATUS")
            return False, {}

    def test_compliant_contract_multiple_scenarios(self):
        """Test multiple contract scenarios to understand response patterns"""
        print("\n" + "="*80)
        print("🎯 TESTING MULTIPLE COMPLIANT CONTRACT SCENARIOS")
        print("="*80)
        
        test_scenarios = [
            {
                "name": "Freelance Agreement",
                "data": {
                    "contract_type": "freelance_agreement",
                    "jurisdiction": "US",
                    "parties": {
                        "party1_name": "Digital Agency LLC",
                        "party1_type": "llc",
                        "party2_name": "John Designer",
                        "party2_type": "individual"
                    },
                    "terms": {
                        "scope": "Website design and development for e-commerce platform",
                        "payment_amount": "$5,000",
                        "payment_terms": "milestone"
                    }
                }
            },
            {
                "name": "Partnership Agreement",
                "data": {
                    "contract_type": "partnership_agreement",
                    "jurisdiction": "US",
                    "parties": {
                        "party1_name": "Alpha Ventures",
                        "party1_type": "company",
                        "party2_name": "Beta Solutions",
                        "party2_type": "company"
                    },
                    "terms": {
                        "business_purpose": "Joint venture for AI software development",
                        "profit_split": "60/40",
                        "capital_contribution": "$50,000 each"
                    }
                }
            }
        ]
        
        all_results = []
        
        for scenario in test_scenarios:
            print(f"\n--- Testing {scenario['name']} ---")
            
            success, response = self.run_test(
                f"Compliant Contract - {scenario['name']}", 
                "POST", 
                "generate-contract-compliant", 
                200, 
                scenario['data'],
                timeout=90
            )
            
            scenario_result = {
                "name": scenario['name'],
                "success": success,
                "response": response
            }
            
            if success and isinstance(response, dict):
                # Analyze suggestions field for review ID patterns
                if 'suggestions' in response:
                    suggestions = response['suggestions']
                    review_ids_found = []
                    
                    if isinstance(suggestions, list):
                        for suggestion in suggestions:
                            review_id_match = re.search(r'review \(ID:\s*([^)]+)\)', suggestion)
                            if review_id_match:
                                review_id = review_id_match.group(1).strip()
                                review_ids_found.append(review_id)
                                print(f"   ✅ Found Review ID: {review_id}")
                    
                    scenario_result['review_ids'] = review_ids_found
                    
                    if not review_ids_found:
                        print(f"   ⚠️  No review IDs found in suggestions for {scenario['name']}")
                else:
                    print(f"   ❌ No suggestions field in response for {scenario['name']}")
            
            all_results.append(scenario_result)
        
        return all_results

    def test_review_id_extraction_patterns(self):
        """Test different regex patterns that might be used for review ID extraction"""
        print("\n" + "="*80)
        print("🎯 TESTING REVIEW ID EXTRACTION PATTERNS")
        print("="*80)
        
        # Sample suggestion texts that might contain review IDs
        sample_suggestions = [
            "This contract has been submitted for attorney review (ID: abc123-def456-ghi789)",
            "Attorney review required - review (ID: 12345678-abcd-efgh-ijkl-123456789012)",
            "Document submitted for compliance review (ID: review-uuid-12345)",
            "Please wait for attorney approval. Review ID: xyz789-review-001",
            "Your contract is under review (ID: contract-review-456789)",
            "Attorney supervision required for this document review (ID: supervision-123-abc)"
        ]
        
        # Different regex patterns that might be used
        regex_patterns = [
            r'ID:\s*([^)]+)',  # The pattern mentioned in requirements
            r'review \(ID:\s*([^)]+)\)',  # Full pattern with "review" prefix
            r'ID:\s*([a-zA-Z0-9\-]+)',  # More specific pattern for UUID-like IDs
            r'\(ID:\s*([^)]+)\)',  # Pattern for anything in (ID: ...)
            r'ID:\s*([^\s,)]+)',  # Pattern that stops at whitespace, comma, or closing paren
        ]
        
        print(f"Testing {len(regex_patterns)} regex patterns against {len(sample_suggestions)} sample suggestions:")
        
        for i, pattern in enumerate(regex_patterns):
            print(f"\n   Pattern {i+1}: {pattern}")
            matches_found = 0
            
            for j, suggestion in enumerate(sample_suggestions):
                match = re.search(pattern, suggestion)
                if match:
                    extracted_id = match.group(1).strip()
                    print(f"     Sample {j+1}: ✅ Extracted '{extracted_id}'")
                    matches_found += 1
                else:
                    print(f"     Sample {j+1}: ❌ No match")
            
            print(f"   Pattern {i+1} success rate: {matches_found}/{len(sample_suggestions)} ({matches_found/len(sample_suggestions)*100:.1f}%)")
        
        return True, {"patterns_tested": len(regex_patterns), "samples_tested": len(sample_suggestions)}

    def run_comprehensive_test(self):
        """Run all compliant contract generation tests"""
        print("\n" + "🚀" + "="*78 + "🚀")
        print("🎯 COMPREHENSIVE COMPLIANT CONTRACT GENERATION TESTING")
        print("🚀" + "="*78 + "🚀")
        
        start_time = time.time()
        
        # Test 1: Simple compliant contract generation
        print(f"\n📋 TEST 1: Simple Compliant Contract Generation")
        test1_success, test1_response = self.test_compliant_contract_generation_simple()
        
        # Test 2: Multiple scenarios
        print(f"\n📋 TEST 2: Multiple Contract Scenarios")
        test2_results = self.test_compliant_contract_multiple_scenarios()
        
        # Test 3: Review ID extraction patterns
        print(f"\n📋 TEST 3: Review ID Extraction Patterns")
        test3_success, test3_response = self.test_review_id_extraction_patterns()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Summary
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("="*80)
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%" if self.tests_run > 0 else "N/A")
        print(f"Total Duration: {duration:.2f} seconds")
        
        # Detailed findings
        print(f"\n🔍 KEY FINDINGS:")
        
        if test1_success:
            print(f"✅ Compliant contract generation endpoint is working")
        else:
            print(f"❌ Compliant contract generation endpoint has issues")
        
        # Analyze multiple scenarios results
        successful_scenarios = sum(1 for result in test2_results if result['success'])
        total_scenarios = len(test2_results)
        print(f"✅ {successful_scenarios}/{total_scenarios} contract scenarios successful")
        
        # Check for review ID patterns
        review_ids_found = []
        for result in test2_results:
            if 'review_ids' in result:
                review_ids_found.extend(result['review_ids'])
        
        if review_ids_found:
            print(f"✅ Found {len(review_ids_found)} review IDs across all tests")
            print(f"   Review IDs: {review_ids_found}")
        else:
            print(f"❌ No review IDs found in any test scenarios")
            print(f"   This explains why the frontend ReviewStatus component never appears!")
        
        print(f"\n🎯 CONCLUSION:")
        if not review_ids_found:
            print(f"❌ CRITICAL ISSUE: Review IDs are not being included in the suggestions field")
            print(f"   This is why the frontend cannot extract review IDs and ReviewStatus component never appears")
            print(f"   The backend needs to include review IDs in the expected format: 'review (ID: {{review_id}})'")
        else:
            print(f"✅ Review ID extraction should work - frontend issue may be elsewhere")
        
        return {
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "success_rate": (self.tests_passed/self.tests_run*100) if self.tests_run > 0 else 0,
            "duration": duration,
            "review_ids_found": review_ids_found,
            "test1_success": test1_success,
            "test2_results": test2_results,
            "test3_success": test3_success
        }

if __name__ == "__main__":
    tester = CompliantContractTester()
    results = tester.run_comprehensive_test()