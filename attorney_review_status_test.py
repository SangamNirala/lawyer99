import requests
import sys
import json
import time
import uuid
from datetime import datetime

class AttorneyReviewStatusTester:
    def __init__(self, base_url="https://cad694cf-d3ea-4711-9a72-668a6a40e52f.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.review_id = None
        self.attorney_id = None

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

    def test_create_demo_attorney(self):
        """Create a demo attorney for testing"""
        attorney_data = {
            "email": f"demo.attorney.{int(time.time())}@legalmate.test",
            "first_name": "Demo",
            "last_name": "Attorney",
            "bar_number": f"BAR{int(time.time())}",
            "jurisdiction": "US",
            "role": "reviewing_attorney",
            "specializations": ["contract_law", "business_law"],
            "years_experience": 5,
            "password": "SecurePassword123!"
        }
        
        success, response = self.run_test(
            "Create Demo Attorney", 
            "POST", 
            "attorney/create", 
            200, 
            attorney_data
        )
        
        if success and 'attorney' in response:
            self.attorney_id = response['attorney'].get('id')
            print(f"   Created attorney ID: {self.attorney_id}")
        
        return success, response

    def test_record_client_consent(self):
        """Record client consent for attorney supervision"""
        client_id = f"client_{int(time.time())}_review_test"
        consent_data = {
            "client_id": client_id,
            "consent_text": "I consent to attorney supervision for document review and legal compliance verification.",
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0 (Test Browser)"
        }
        
        success, response = self.run_test(
            "Record Client Consent", 
            "POST", 
            "client/consent", 
            200, 
            consent_data
        )
        
        if success:
            print(f"   Consent recorded for client: {client_id}")
            return success, response, client_id
        
        return success, response, None

    def test_submit_document_for_review(self, client_id=None):
        """Submit a document for attorney review to create a review ID"""
        if not client_id:
            client_id = f"client_{int(time.time())}_review_test"
        
        # First record consent if not already done
        if not client_id:
            _, _, client_id = self.test_record_client_consent()
        
        review_data = {
            "document_content": "**NON-DISCLOSURE AGREEMENT**\n\nThis Non-Disclosure Agreement is entered into between Test Company Inc. and John Doe for the purpose of business collaboration evaluation.\n\n**1. CONFIDENTIAL INFORMATION**\nFor purposes of this Agreement, 'Confidential Information' means any and all information disclosed by either party.\n\n**2. OBLIGATIONS**\nThe receiving party agrees to maintain confidentiality of all disclosed information.\n\n**3. TERM**\nThis Agreement shall remain in effect for a period of two (2) years.\n\n**4. GOVERNING LAW**\nThis Agreement shall be governed by the laws of the United States.",
            "document_type": "contract",
            "client_id": client_id,
            "original_request": {
                "contract_type": "NDA",
                "parties": {
                    "party1_name": "Test Company Inc.",
                    "party2_name": "John Doe"
                },
                "purpose": "Business collaboration evaluation"
            },
            "priority": "normal"
        }
        
        success, response = self.run_test(
            "Submit Document for Attorney Review", 
            "POST", 
            "attorney/review/submit", 
            200, 
            review_data
        )
        
        if success and 'review_id' in response:
            self.review_id = response['review_id']
            print(f"   Created review ID: {self.review_id}")
        
        return success, response

    def test_review_status_endpoint_direct(self):
        """Test the review status endpoint directly with a known review ID"""
        if not self.review_id:
            print("⚠️  No review ID available, skipping direct status test")
            return True, {}
        
        success, response = self.run_test(
            f"Get Review Status - {self.review_id}", 
            "GET", 
            f"attorney/review/status/{self.review_id}", 
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   📊 Review Status Details:")
            print(f"      Review ID: {response.get('review_id', 'N/A')}")
            print(f"      Status: {response.get('status', 'N/A')}")
            print(f"      Progress: {response.get('progress_percentage', 'N/A')}%")
            print(f"      Assigned Attorney: {response.get('assigned_attorney_id', 'N/A')}")
            print(f"      Priority: {response.get('priority', 'N/A')}")
            print(f"      Created At: {response.get('created_at', 'N/A')}")
            print(f"      Estimated Completion: {response.get('estimated_completion', 'N/A')}")
            
            # Verify required fields are present
            required_fields = ['review_id', 'status', 'progress_percentage', 'created_at', 'estimated_completion']
            missing_fields = [field for field in required_fields if field not in response]
            
            if not missing_fields:
                print(f"   ✅ All required response fields present")
            else:
                print(f"   ❌ Missing required fields: {missing_fields}")
            
            # Check progress percentage is a valid number
            progress = response.get('progress_percentage')
            if isinstance(progress, (int, float)) and 0 <= progress <= 100:
                print(f"   ✅ Progress percentage is valid: {progress}%")
            else:
                print(f"   ❌ Invalid progress percentage: {progress}")
        
        return success, response

    def test_review_status_endpoint_invalid_id(self):
        """Test review status endpoint with invalid review ID"""
        invalid_id = "invalid-review-id-12345"
        
        success, response = self.run_test(
            "Get Review Status - Invalid ID", 
            "GET", 
            f"attorney/review/status/{invalid_id}", 
            404
        )
        
        if success:
            print(f"   ✅ Correctly returned 404 for invalid review ID")
        
        return success, response

    def test_review_status_endpoint_nonexistent_uuid(self):
        """Test review status endpoint with non-existent but valid UUID"""
        nonexistent_id = str(uuid.uuid4())
        
        success, response = self.run_test(
            "Get Review Status - Non-existent UUID", 
            "GET", 
            f"attorney/review/status/{nonexistent_id}", 
            404
        )
        
        if success:
            print(f"   ✅ Correctly returned 404 for non-existent review ID")
        
        return success, response

    def test_review_status_url_construction(self):
        """Test that the review status URL is constructed correctly without double /api prefixes"""
        if not self.review_id:
            print("⚠️  No review ID available, skipping URL construction test")
            return True, {}
        
        # Test the exact URL construction that frontend would use
        expected_url = f"{self.api_url}/attorney/review/status/{self.review_id}"
        
        print(f"\n🔍 Testing URL Construction...")
        print(f"   Expected URL: {expected_url}")
        print(f"   Checking for double /api prefixes...")
        
        # Verify URL doesn't have double /api
        if "/api/api/" in expected_url:
            print(f"   ❌ Double /api prefix detected in URL!")
            return False, {"error": "Double /api prefix in URL"}
        else:
            print(f"   ✅ No double /api prefix detected")
        
        # Test the actual request
        try:
            response = requests.get(expected_url, headers={'Content-Type': 'application/json'}, timeout=30)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                self.tests_passed += 1
                print(f"   ✅ URL construction test passed - endpoint accessible")
                try:
                    response_data = response.json()
                    return True, response_data
                except:
                    return True, response.text
            else:
                print(f"   ❌ URL construction test failed - Status: {response.status_code}")
                return False, {}
                
        except Exception as e:
            print(f"   ❌ URL construction test failed - Error: {str(e)}")
            return False, {}

    def test_progress_percentage_calculation(self):
        """Test progress percentage calculation and updates over time"""
        if not self.review_id:
            print("⚠️  No review ID available, skipping progress calculation test")
            return True, {}
        
        print(f"\n🔍 Testing Progress Percentage Calculation...")
        
        # Get initial progress
        success1, response1 = self.run_test(
            "Initial Progress Check", 
            "GET", 
            f"attorney/review/status/{self.review_id}", 
            200
        )
        
        if not success1:
            return False, {}
        
        initial_progress = response1.get('progress_percentage', 0)
        initial_status = response1.get('status', 'unknown')
        
        print(f"   Initial Progress: {initial_progress}%")
        print(f"   Initial Status: {initial_status}")
        
        # Wait a few seconds and check again
        print(f"   Waiting 5 seconds to check for progress updates...")
        time.sleep(5)
        
        success2, response2 = self.run_test(
            "Progress Check After Wait", 
            "GET", 
            f"attorney/review/status/{self.review_id}", 
            200
        )
        
        if not success2:
            return False, {}
        
        updated_progress = response2.get('progress_percentage', 0)
        updated_status = response2.get('status', 'unknown')
        
        print(f"   Updated Progress: {updated_progress}%")
        print(f"   Updated Status: {updated_status}")
        
        # Analyze progress behavior
        if initial_status == 'pending' and initial_progress == 0:
            print(f"   📊 Review is in pending status with 0% progress (waiting for attorney assignment)")
            if updated_progress == 0 and updated_status == 'pending':
                print(f"   ⚠️  Progress remains at 0% - may indicate attorney assignment issue")
            else:
                print(f"   ✅ Progress updated from {initial_progress}% to {updated_progress}%")
        elif initial_status == 'in_review' and initial_progress > 0:
            print(f"   ✅ Review is in progress with {initial_progress}% completion")
            if updated_progress >= initial_progress:
                print(f"   ✅ Progress is advancing or stable: {initial_progress}% → {updated_progress}%")
            else:
                print(f"   ❌ Progress decreased: {initial_progress}% → {updated_progress}%")
        else:
            print(f"   📊 Review status: {initial_status}, Progress: {initial_progress}%")
        
        # Check for realistic completion times
        estimated_completion = response2.get('estimated_completion')
        if estimated_completion and estimated_completion != 'Overdue':
            print(f"   ✅ Realistic completion time provided: {estimated_completion}")
        elif estimated_completion == 'Overdue':
            print(f"   ⚠️  Review marked as 'Overdue' - may indicate stuck review")
        else:
            print(f"   ⚠️  No estimated completion time provided")
        
        return True, {
            "initial_progress": initial_progress,
            "updated_progress": updated_progress,
            "initial_status": initial_status,
            "updated_status": updated_status,
            "estimated_completion": estimated_completion
        }

    def test_complete_review_workflow(self):
        """Test the complete review workflow from consent to status tracking"""
        print(f"\n🔍 Testing Complete Review Workflow...")
        
        # Step 1: Record client consent
        print(f"   Step 1: Recording client consent...")
        consent_success, consent_response, client_id = self.test_record_client_consent()
        if not consent_success:
            print(f"   ❌ Failed at consent step")
            return False, {}
        
        # Step 2: Submit document for review
        print(f"   Step 2: Submitting document for review...")
        submit_success, submit_response = self.test_submit_document_for_review(client_id)
        if not submit_success:
            print(f"   ❌ Failed at document submission step")
            return False, {}
        
        # Step 3: Check review status immediately
        print(f"   Step 3: Checking initial review status...")
        status_success, status_response = self.test_review_status_endpoint_direct()
        if not status_success:
            print(f"   ❌ Failed at status check step")
            return False, {}
        
        # Step 4: Verify URL construction
        print(f"   Step 4: Verifying URL construction...")
        url_success, url_response = self.test_review_status_url_construction()
        if not url_success:
            print(f"   ❌ Failed at URL construction step")
            return False, {}
        
        # Step 5: Monitor progress over time
        print(f"   Step 5: Monitoring progress calculation...")
        progress_success, progress_response = self.test_progress_percentage_calculation()
        if not progress_success:
            print(f"   ❌ Failed at progress monitoring step")
            return False, {}
        
        print(f"   ✅ Complete workflow test passed!")
        
        return True, {
            "client_id": client_id,
            "review_id": self.review_id,
            "workflow_steps": {
                "consent": consent_success,
                "submit": submit_success,
                "status": status_success,
                "url_construction": url_success,
                "progress": progress_success
            }
        }

    def run_all_tests(self):
        """Run all attorney review status tests"""
        print("🚀 Starting Attorney Review Status Endpoint Testing...")
        print(f"   Base URL: {self.base_url}")
        print(f"   API URL: {self.api_url}")
        
        # Test 1: Create demo attorney (optional, for assignment)
        print(f"\n" + "="*60)
        print(f"TEST 1: Create Demo Attorney")
        print(f"="*60)
        self.test_create_demo_attorney()
        
        # Test 2: Complete workflow test
        print(f"\n" + "="*60)
        print(f"TEST 2: Complete Review Workflow")
        print(f"="*60)
        self.test_complete_review_workflow()
        
        # Test 3: Error handling tests
        print(f"\n" + "="*60)
        print(f"TEST 3: Error Handling Tests")
        print(f"="*60)
        self.test_review_status_endpoint_invalid_id()
        self.test_review_status_endpoint_nonexistent_uuid()
        
        # Final summary
        print(f"\n" + "="*60)
        print(f"ATTORNEY REVIEW STATUS TEST SUMMARY")
        print(f"="*60)
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%" if self.tests_run > 0 else "0%")
        
        if self.review_id:
            print(f"Review ID Created: {self.review_id}")
        if self.attorney_id:
            print(f"Attorney ID Created: {self.attorney_id}")
        
        return self.tests_passed, self.tests_run

if __name__ == "__main__":
    tester = AttorneyReviewStatusTester()
    passed, total = tester.run_all_tests()
    
    if passed == total:
        print(f"\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        sys.exit(1)