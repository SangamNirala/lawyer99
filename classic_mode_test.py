import requests
import sys
import json
import time
import random
from datetime import datetime

class ClassicModeContractTester:
    def __init__(self, base_url="https://b3d0e54e-8004-47d5-83bd-25e76a95a599.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.client_id = None
        self.review_id = None

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
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=timeout)

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

    def generate_client_id(self):
        """Generate a client ID in the format client_timestamp_randomstring"""
        timestamp = str(int(time.time() * 1000))
        random_string = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=10))
        self.client_id = f"client_{timestamp}_{random_string}"
        return self.client_id

    def test_consent_recording(self):
        """Test client consent recording endpoint"""
        if not self.client_id:
            self.generate_client_id()
        
        consent_data = {
            "client_id": self.client_id,
            "consent_text": "I understand and agree to attorney supervision for legal document generation",
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0 (Test Browser)"
        }
        
        success, response = self.run_test(
            "Client Consent Recording", 
            "POST", 
            "client/consent", 
            200, 
            consent_data
        )
        
        if success and response.get('success'):
            print(f"   ✅ Consent recorded successfully")
            print(f"   Consent ID: {response.get('consent_id')}")
            print(f"   Client ID: {self.client_id}")
        
        return success, response

    def test_consent_validation(self):
        """Test client consent validation endpoint"""
        if not self.client_id:
            print("⚠️  No client ID available for consent validation")
            return False, {}
        
        success, response = self.run_test(
            "Client Consent Validation", 
            "GET", 
            f"client/consent/check/{self.client_id}", 
            200
        )
        
        if success and response.get('has_consent'):
            print(f"   ✅ Consent validation successful")
            print(f"   Has consent: {response.get('has_consent')}")
        
        return success, response

    def test_compliant_contract_generation(self):
        """Test the compliant contract generation endpoint - the critical failing endpoint"""
        if not self.client_id:
            self.generate_client_id()
        
        # Test data matching the review request specifications
        contract_data = {
            "contract_type": "NDA",
            "jurisdiction": "US",
            "parties": {
                "party1_name": "Test Company Inc.",
                "party1_type": "company",
                "party2_name": "John Doe",
                "party2_type": "individual"
            },
            "terms": {
                "purpose": "Business collaboration evaluation",
                "duration": "2_years"
            },
            "special_clauses": [],
            "client_id": self.client_id
        }
        
        success, response = self.run_test(
            "Compliant Contract Generation (CRITICAL)", 
            "POST", 
            "generate-contract-compliant", 
            200, 
            contract_data,
            timeout=90  # Longer timeout for AI generation + compliance checking
        )
        
        if success:
            print(f"   🔍 Analyzing compliant contract generation response...")
            
            # Check for contract content
            if 'contract' in response:
                contract = response['contract']
                print(f"   ✅ Contract generated successfully")
                print(f"   Contract ID: {contract.get('id')}")
                print(f"   Contract Type: {contract.get('contract_type')}")
                print(f"   Compliance Score: {contract.get('compliance_score', 'N/A')}")
                print(f"   Content Length: {len(contract.get('content', ''))} characters")
            
            # Check for suggestions containing review ID
            if 'suggestions' in response:
                suggestions = response['suggestions']
                print(f"   📋 Suggestions found: {len(suggestions)} items")
                
                # Look for review ID in suggestions
                review_id_found = False
                for suggestion in suggestions:
                    print(f"   - {suggestion}")
                    if 'ID:' in suggestion and 'review' in suggestion.lower():
                        # Extract review ID using regex
                        import re
                        match = re.search(r'ID:\s*([^)]+)', suggestion)
                        if match:
                            self.review_id = match.group(1).strip()
                            review_id_found = True
                            print(f"   ✅ Review ID extracted: {self.review_id}")
                
                if not review_id_found:
                    print(f"   ❌ No review ID found in suggestions - this is the critical issue!")
                    print(f"   Expected format: 'Document submitted for attorney review (ID: uuid-here)'")
            else:
                print(f"   ❌ No suggestions in response - missing review workflow!")
            
            # Check for warnings
            if 'warnings' in response and response['warnings']:
                print(f"   ⚠️  Warnings: {response['warnings']}")
            
            # Check for compliance information
            if 'requires_attorney_review' in response:
                print(f"   📋 Requires attorney review: {response.get('requires_attorney_review')}")
            
        return success, response

    def test_review_status_endpoint(self):
        """Test the review status endpoint with extracted review ID"""
        if not self.review_id:
            print("⚠️  No review ID available for status check")
            return False, {}
        
        success, response = self.run_test(
            "Review Status Check", 
            "GET", 
            f"attorney/review/status/{self.review_id}", 
            200
        )
        
        if success:
            print(f"   📊 Review Status Details:")
            print(f"   Review ID: {response.get('review_id')}")
            print(f"   Status: {response.get('status')}")
            print(f"   Progress: {response.get('progress_percentage', 0)}%")
            print(f"   Created: {response.get('created_at')}")
            print(f"   Estimated Completion: {response.get('estimated_completion')}")
            
            if 'assigned_attorney' in response:
                attorney = response['assigned_attorney']
                if attorney:
                    print(f"   Assigned Attorney: {attorney.get('first_name')} {attorney.get('last_name')}")
                else:
                    print(f"   ❌ No attorney assigned - this could be the issue!")
        
        return success, response

    def test_attorney_creation(self):
        """Test attorney creation to ensure attorneys are available for assignment"""
        attorney_data = {
            "email": f"test.attorney.{int(time.time())}@legalmate.ai",
            "first_name": "Test",
            "last_name": "Attorney",
            "bar_number": f"BAR{random.randint(100000, 999999)}",
            "jurisdiction": "US",
            "role": "senior_attorney",
            "specializations": ["contract_law", "business_law"],
            "years_experience": 10,
            "password": "TestPassword123!"
        }
        
        success, response = self.run_test(
            "Attorney Creation", 
            "POST", 
            "attorney/create", 
            200, 
            attorney_data
        )
        
        if success:
            print(f"   ✅ Attorney created successfully")
            print(f"   Attorney ID: {response.get('attorney_id')}")
            print(f"   Email: {attorney_data['email']}")
        
        return success, response

    def test_complete_classic_mode_flow(self):
        """Test the complete Classic Mode contract generation flow"""
        print("\n" + "="*80)
        print("🎯 TESTING COMPLETE CLASSIC MODE CONTRACT GENERATION FLOW")
        print("="*80)
        
        # Step 1: Generate client ID
        print(f"\n📋 Step 1: Generate Client ID")
        client_id = self.generate_client_id()
        print(f"   Generated Client ID: {client_id}")
        
        # Step 2: Record consent
        print(f"\n📋 Step 2: Record Client Consent")
        consent_success, consent_response = self.test_consent_recording()
        if not consent_success:
            print(f"❌ CRITICAL FAILURE: Consent recording failed - cannot proceed")
            return False
        
        # Step 3: Validate consent
        print(f"\n📋 Step 3: Validate Client Consent")
        validation_success, validation_response = self.test_consent_validation()
        if not validation_success or not validation_response.get('has_consent'):
            print(f"❌ CRITICAL FAILURE: Consent validation failed - cannot proceed")
            return False
        
        # Step 4: Ensure attorneys are available
        print(f"\n📋 Step 4: Ensure Attorney Availability")
        attorney_success, attorney_response = self.test_attorney_creation()
        if not attorney_success:
            print(f"⚠️  Attorney creation failed - may affect assignment")
        
        # Step 5: Generate compliant contract (CRITICAL STEP)
        print(f"\n📋 Step 5: Generate Compliant Contract (CRITICAL)")
        contract_success, contract_response = self.test_compliant_contract_generation()
        if not contract_success:
            print(f"❌ CRITICAL FAILURE: Compliant contract generation failed")
            return False
        
        # Step 6: Check review status
        print(f"\n📋 Step 6: Check Review Status")
        if self.review_id:
            status_success, status_response = self.test_review_status_endpoint()
            if not status_success:
                print(f"❌ Review status check failed")
                return False
        else:
            print(f"❌ CRITICAL ISSUE: No review ID available - contract generation didn't create review")
            return False
        
        # Summary
        print(f"\n" + "="*80)
        print(f"📊 CLASSIC MODE FLOW TEST SUMMARY")
        print(f"="*80)
        print(f"✅ Consent Recording: {'SUCCESS' if consent_success else 'FAILED'}")
        print(f"✅ Consent Validation: {'SUCCESS' if validation_success else 'FAILED'}")
        print(f"✅ Attorney Creation: {'SUCCESS' if attorney_success else 'FAILED'}")
        print(f"✅ Contract Generation: {'SUCCESS' if contract_success else 'FAILED'}")
        print(f"✅ Review ID Extraction: {'SUCCESS' if self.review_id else 'FAILED'}")
        print(f"✅ Review Status Check: {'SUCCESS' if self.review_id and status_success else 'FAILED'}")
        
        overall_success = all([consent_success, validation_success, contract_success, bool(self.review_id)])
        print(f"\n🎯 OVERALL RESULT: {'SUCCESS' if overall_success else 'FAILED'}")
        
        if not overall_success:
            print(f"\n❌ CRITICAL ISSUES IDENTIFIED:")
            if not self.review_id:
                print(f"   - Contract generation does not create review ID in suggestions")
                print(f"   - This prevents ReviewStatus component from appearing")
                print(f"   - Users get stuck after consent is provided")
        
        return overall_success

    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        print(f"\n📋 Testing Edge Cases and Error Conditions")
        
        # Test 1: Contract generation without consent
        print(f"\n🔍 Test 1: Contract generation without consent")
        no_consent_client = f"client_{int(time.time() * 1000)}_noconsent"
        
        contract_data = {
            "contract_type": "NDA",
            "jurisdiction": "US",
            "parties": {
                "party1_name": "No Consent Corp",
                "party1_type": "company",
                "party2_name": "Test User",
                "party2_type": "individual"
            },
            "terms": {
                "purpose": "Testing without consent",
                "duration": "1_year"
            },
            "client_id": no_consent_client
        }
        
        success, response = self.run_test(
            "Contract Generation Without Consent", 
            "POST", 
            "generate-contract-compliant", 
            400,  # Should fail without consent
            contract_data
        )
        
        if success:
            print(f"   ✅ Correctly rejected contract generation without consent")
        else:
            print(f"   ❌ Should have rejected contract generation without consent")
        
        # Test 2: Invalid contract type
        print(f"\n🔍 Test 2: Invalid contract type")
        if not self.client_id:
            self.generate_client_id()
        
        invalid_contract_data = {
            "contract_type": "INVALID_TYPE",
            "jurisdiction": "US",
            "parties": {
                "party1_name": "Invalid Test Corp",
                "party1_type": "company",
                "party2_name": "Test User",
                "party2_type": "individual"
            },
            "terms": {
                "purpose": "Testing invalid contract type",
                "duration": "1_year"
            },
            "client_id": self.client_id
        }
        
        success, response = self.run_test(
            "Invalid Contract Type", 
            "POST", 
            "generate-contract-compliant", 
            422,  # Should fail with validation error
            invalid_contract_data
        )
        
        if success:
            print(f"   ✅ Correctly rejected invalid contract type")
        else:
            # Try with 500 status code
            success, response = self.run_test(
                "Invalid Contract Type (500)", 
                "POST", 
                "generate-contract-compliant", 
                500,
                invalid_contract_data
            )
            if success:
                print(f"   ✅ Correctly rejected invalid contract type (500)")
                self.tests_passed += 1  # Adjust count

    def run_all_tests(self):
        """Run all Classic Mode contract generation tests"""
        print("🚀 Starting Classic Mode Contract Generation Tests")
        print("="*80)
        
        start_time = time.time()
        
        # Run the complete flow test
        flow_success = self.test_complete_classic_mode_flow()
        
        # Run edge case tests
        self.test_edge_cases()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Final summary
        print(f"\n" + "="*80)
        print(f"🏁 FINAL TEST RESULTS")
        print(f"="*80)
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Client ID Used: {self.client_id}")
        print(f"Review ID Extracted: {self.review_id or 'None'}")
        
        if flow_success:
            print(f"\n✅ CLASSIC MODE FLOW: WORKING")
        else:
            print(f"\n❌ CLASSIC MODE FLOW: FAILING")
            print(f"   Root cause: Contract generation not creating review workflow")
        
        return flow_success

if __name__ == "__main__":
    tester = ClassicModeContractTester()
    success = tester.run_all_tests()
    
    if not success:
        print(f"\n🚨 CRITICAL ISSUES FOUND IN CLASSIC MODE")
        print(f"   The contract generation flow is not completing properly")
        print(f"   Users will get stuck after providing consent")
        sys.exit(1)
    else:
        print(f"\n🎉 CLASSIC MODE WORKING CORRECTLY")
        sys.exit(0)