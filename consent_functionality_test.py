#!/usr/bin/env python3
"""
Consent Functionality Fix Testing
=================================

This test specifically focuses on testing the consent functionality fix that was implemented
to resolve the infinite loop issue where clicking "Provide Consent & Continue" causes the 
consent popup to reappear repeatedly instead of proceeding with contract generation.

Test Requirements:
1. Test POST /api/client/consent - consent recording endpoint
2. Test GET /api/client/consent/check/{client_id} - consent validation endpoint  
3. Test POST /api/generate-contract-compliant - compliant contract generation
4. Test GET /api/attorney/review/status/{review_id} - review status endpoint
5. Test complete workflow end-to-end
"""

import requests
import sys
import json
import time
import random
import uuid
from datetime import datetime

class ConsentFunctionalityTester:
    def __init__(self, base_url="https://legalcore.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.client_id = None
        self.review_id = None
        
    def log(self, message, level="INFO"):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test with detailed logging"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        self.log(f"🔍 Testing {name}")
        self.log(f"   Method: {method} | URL: {url}")
        if data:
            self.log(f"   Data: {json.dumps(data, indent=2)}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=timeout)

            self.log(f"   Response Status: {response.status_code}")
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                self.log(f"✅ PASSED - {name}", "SUCCESS")
                try:
                    response_data = response.json()
                    if isinstance(response_data, dict):
                        self.log(f"   Response keys: {list(response_data.keys())}")
                        # Log important fields for debugging
                        if 'success' in response_data:
                            self.log(f"   Success: {response_data['success']}")
                        if 'consent_id' in response_data:
                            self.log(f"   Consent ID: {response_data['consent_id']}")
                        if 'has_consent' in response_data:
                            self.log(f"   Has Consent: {response_data['has_consent']}")
                    return True, response_data
                except:
                    return True, response.text
            else:
                self.log(f"❌ FAILED - {name} | Expected {expected_status}, got {response.status_code}", "ERROR")
                try:
                    error_data = response.json()
                    self.log(f"   Error Details: {error_data}", "ERROR")
                except:
                    self.log(f"   Error Text: {response.text}", "ERROR")
                return False, None
                
        except requests.exceptions.Timeout:
            self.log(f"❌ TIMEOUT - {name} (>{timeout}s)", "ERROR")
            return False, None
        except Exception as e:
            self.log(f"❌ EXCEPTION - {name}: {str(e)}", "ERROR")
            return False, None

    def generate_client_id(self):
        """Generate client_id in the format: client_[timestamp]_[randomstring]"""
        timestamp = str(int(time.time() * 1000))  # milliseconds
        random_string = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=11))
        return f"client_{timestamp}_{random_string}"

    def test_consent_recording(self):
        """Test POST /api/client/consent - consent recording endpoint"""
        self.log("=" * 60)
        self.log("TESTING CONSENT RECORDING ENDPOINT")
        self.log("=" * 60)
        
        # Generate a unique client_id for this test
        self.client_id = self.generate_client_id()
        self.log(f"Generated client_id: {self.client_id}")
        
        consent_data = {
            "client_id": self.client_id,
            "consent_text": "I hereby provide my informed consent for the creation of legal documents under attorney supervision as required by applicable legal regulations. I understand that this service operates under attorney supervision to ensure compliance with unauthorized practice of law regulations.",
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        success, response = self.run_test(
            "Consent Recording",
            "POST",
            "client/consent",
            200,
            consent_data
        )
        
        if success and response:
            # Verify response structure
            required_fields = ['success', 'consent_id', 'message']
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                self.log(f"❌ Missing required fields in response: {missing_fields}", "ERROR")
                return False
            
            if response.get('success') != True:
                self.log(f"❌ Consent recording not successful: {response.get('success')}", "ERROR")
                return False
                
            self.log(f"✅ Consent recorded successfully with ID: {response.get('consent_id')}")
            return True
        
        return False

    def test_consent_validation(self):
        """Test GET /api/client/consent/check/{client_id} - consent validation endpoint"""
        self.log("=" * 60)
        self.log("TESTING CONSENT VALIDATION ENDPOINT")
        self.log("=" * 60)
        
        if not self.client_id:
            self.log("❌ No client_id available for validation test", "ERROR")
            return False
        
        # Test with existing client_id (should return has_consent: true)
        success, response = self.run_test(
            "Consent Validation - Existing Client",
            "GET",
            f"client/consent/check/{self.client_id}",
            200
        )
        
        if success and response:
            required_fields = ['client_id', 'has_consent', 'consent_required']
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                self.log(f"❌ Missing required fields in response: {missing_fields}", "ERROR")
                return False
            
            if response.get('has_consent') != True:
                self.log(f"❌ Expected has_consent=true, got: {response.get('has_consent')}", "ERROR")
                return False
                
            self.log(f"✅ Consent validation successful - has_consent: {response.get('has_consent')}")
        else:
            return False
        
        # Test with non-existent client_id (should return has_consent: false)
        fake_client_id = self.generate_client_id()
        success, response = self.run_test(
            "Consent Validation - Non-existent Client",
            "GET",
            f"client/consent/check/{fake_client_id}",
            200
        )
        
        if success and response:
            if response.get('has_consent') != False:
                self.log(f"❌ Expected has_consent=false for non-existent client, got: {response.get('has_consent')}", "ERROR")
                return False
                
            self.log(f"✅ Non-existent client validation successful - has_consent: {response.get('has_consent')}")
            return True
        
        return False

    def test_compliant_contract_generation(self):
        """Test POST /api/generate-contract-compliant - compliant contract generation"""
        self.log("=" * 60)
        self.log("TESTING COMPLIANT CONTRACT GENERATION ENDPOINT")
        self.log("=" * 60)
        
        if not self.client_id:
            self.log("❌ No client_id available for contract generation test", "ERROR")
            return False
        
        # Use exact test data from review request
        contract_data = {
            "contract_type": "NDA",
            "parties": {
                "party1_name": "Test Company Inc.",
                "party1_type": "Company",
                "party2_name": "John Doe",
                "party2_type": "Individual"
            },
            "terms": {
                "purpose": "Business collaboration evaluation",
                "duration": "2 years",
                "governing_law": "California"
            },
            "jurisdiction": "US",
            "client_id": self.client_id
        }
        
        success, response = self.run_test(
            "Compliant Contract Generation",
            "POST",
            "generate-contract-compliant",
            200,
            contract_data
        )
        
        if success and response:
            # Verify response contains review ID in suggestions
            suggestions = response.get('suggestions', [])
            review_id_found = False
            
            for suggestion in suggestions:
                if "Document submitted for attorney review (ID:" in suggestion:
                    # Extract review ID from suggestion
                    import re
                    match = re.search(r'ID:\s*([a-f0-9-]{36})', suggestion)
                    if match:
                        self.review_id = match.group(1)
                        review_id_found = True
                        self.log(f"✅ Review ID extracted: {self.review_id}")
                        break
            
            if not review_id_found:
                self.log("❌ No review ID found in suggestions", "ERROR")
                self.log(f"   Suggestions: {suggestions}")
                return False
            
            # Verify review ID is valid UUID format
            try:
                uuid.UUID(self.review_id)
                self.log(f"✅ Review ID is valid UUID format")
            except ValueError:
                self.log(f"❌ Review ID is not valid UUID format: {self.review_id}", "ERROR")
                return False
            
            self.log(f"✅ Contract generation successful with review ID: {self.review_id}")
            return True
        
        return False

    def test_review_status(self):
        """Test GET /api/attorney/review/status/{review_id} - review status endpoint"""
        self.log("=" * 60)
        self.log("TESTING REVIEW STATUS ENDPOINT")
        self.log("=" * 60)
        
        if not self.review_id:
            self.log("❌ No review_id available for status test", "ERROR")
            return False
        
        success, response = self.run_test(
            "Review Status Check",
            "GET",
            f"attorney/review/status/{self.review_id}",
            200
        )
        
        if success and response:
            required_fields = ['review_id', 'status', 'progress_percentage', 'estimated_completion', 'attorney']
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                self.log(f"❌ Missing required fields in response: {missing_fields}", "ERROR")
                return False
            
            # Verify review_id matches
            if response.get('review_id') != self.review_id:
                self.log(f"❌ Review ID mismatch. Expected: {self.review_id}, Got: {response.get('review_id')}", "ERROR")
                return False
            
            self.log(f"✅ Review status retrieved successfully:")
            self.log(f"   Status: {response.get('status')}")
            self.log(f"   Progress: {response.get('progress_percentage')}%")
            self.log(f"   Estimated Completion: {response.get('estimated_completion')}")
            self.log(f"   Assigned Attorney: {response.get('attorney')}")
            
            return True
        
        return False

    def test_error_scenarios(self):
        """Test error scenarios for all endpoints"""
        self.log("=" * 60)
        self.log("TESTING ERROR SCENARIOS")
        self.log("=" * 60)
        
        error_tests_passed = 0
        total_error_tests = 0
        
        # Test invalid client_id format for consent recording
        total_error_tests += 1
        invalid_consent_data = {
            "client_id": "invalid_format",
            "consent_text": "Test consent",
            "ip_address": "192.168.1.100"
        }
        
        success, response = self.run_test(
            "Invalid Client ID Format - Consent Recording",
            "POST",
            "client/consent",
            422,  # Expecting validation error
            invalid_consent_data
        )
        if success:
            error_tests_passed += 1
        
        # Test missing required consent data
        total_error_tests += 1
        missing_data = {
            "client_id": self.generate_client_id()
            # Missing consent_text
        }
        
        success, response = self.run_test(
            "Missing Required Data - Consent Recording",
            "POST",
            "client/consent",
            422,  # Expecting validation error
            missing_data
        )
        if success:
            error_tests_passed += 1
        
        # Test non-existent review ID
        total_error_tests += 1
        fake_review_id = str(uuid.uuid4())
        success, response = self.run_test(
            "Non-existent Review ID - Status Check",
            "GET",
            f"attorney/review/status/{fake_review_id}",
            404  # Expecting not found
        )
        if success:
            error_tests_passed += 1
        
        self.log(f"✅ Error scenario tests: {error_tests_passed}/{total_error_tests} passed")
        return error_tests_passed == total_error_tests

    def test_complete_workflow(self):
        """Test the complete workflow end-to-end"""
        self.log("=" * 80)
        self.log("TESTING COMPLETE CONSENT WORKFLOW END-TO-END")
        self.log("=" * 80)
        
        workflow_steps = [
            ("Record Consent", self.test_consent_recording),
            ("Validate Consent", self.test_consent_validation),
            ("Generate Contract", self.test_compliant_contract_generation),
            ("Check Review Status", self.test_review_status)
        ]
        
        workflow_passed = 0
        for step_name, test_func in workflow_steps:
            self.log(f"\n🔄 Workflow Step: {step_name}")
            if test_func():
                workflow_passed += 1
                self.log(f"✅ {step_name} - PASSED")
            else:
                self.log(f"❌ {step_name} - FAILED")
                break  # Stop workflow if any step fails
        
        workflow_success = workflow_passed == len(workflow_steps)
        
        if workflow_success:
            self.log("🎉 COMPLETE WORKFLOW SUCCESSFUL!", "SUCCESS")
        else:
            self.log(f"❌ WORKFLOW FAILED at step {workflow_passed + 1}/{len(workflow_steps)}", "ERROR")
        
        return workflow_success

    def run_all_tests(self):
        """Run all consent functionality tests"""
        self.log("🚀 Starting Consent Functionality Fix Testing")
        self.log(f"Backend URL: {self.base_url}")
        self.log(f"API URL: {self.api_url}")
        
        start_time = time.time()
        
        # Run complete workflow test
        workflow_success = self.test_complete_workflow()
        
        # Run error scenario tests
        self.log("\n" + "=" * 80)
        error_tests_success = self.test_error_scenarios()
        
        # Calculate results
        end_time = time.time()
        duration = end_time - start_time
        
        self.log("\n" + "=" * 80)
        self.log("CONSENT FUNCTIONALITY TEST RESULTS")
        self.log("=" * 80)
        self.log(f"Tests Run: {self.tests_run}")
        self.log(f"Tests Passed: {self.tests_passed}")
        self.log(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        self.log(f"Duration: {duration:.2f} seconds")
        self.log(f"Complete Workflow: {'✅ PASSED' if workflow_success else '❌ FAILED'}")
        self.log(f"Error Scenarios: {'✅ PASSED' if error_tests_success else '❌ FAILED'}")
        
        overall_success = workflow_success and error_tests_success
        
        if overall_success:
            self.log("🎉 CONSENT FUNCTIONALITY FIX TESTING - OUTSTANDING SUCCESS!", "SUCCESS")
            self.log("✅ All consent endpoints are working correctly")
            self.log("✅ Complete workflow functions end-to-end")
            self.log("✅ Error handling is appropriate")
            self.log("✅ The infinite loop issue appears to be resolved from backend perspective")
        else:
            self.log("❌ CONSENT FUNCTIONALITY TESTING - ISSUES DETECTED", "ERROR")
            if not workflow_success:
                self.log("❌ Complete workflow has issues")
            if not error_tests_success:
                self.log("❌ Error handling needs improvement")
        
        return overall_success

if __name__ == "__main__":
    tester = ConsentFunctionalityTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)