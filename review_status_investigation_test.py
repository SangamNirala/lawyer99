import requests
import sys
import json
import time
import random
from datetime import datetime

class ReviewStatusInvestigationTester:
    def __init__(self, base_url="https://legalmate-research.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.review_id = None
        self.client_id = None

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
            
            # Accept both expected status and some common alternatives
            success = response.status_code == expected_status
            if not success and expected_status == 200:
                # For 200 requests, also accept 201 (created) as success
                success = response.status_code == 201
            
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
                    # Still return the response data for analysis even if status doesn't match
                    return False, error_data
                except:
                    print(f"   Error: {response.text}")
                    return False, {"error": response.text}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timed out after {timeout} seconds")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_client_consent_recording(self):
        """Test client consent recording with exact user scenario format"""
        # Generate client ID in the exact format from user scenario
        timestamp = int(time.time() * 1000)
        random_string = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=11))
        self.client_id = f"client_{timestamp}_{random_string}"
        
        consent_data = {
            "client_id": self.client_id,
            "consent_text": "I consent to attorney supervision and review of legal documents as required by law.",
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
        
        if success:
            print(f"   ✅ Client consent recorded successfully")
            print(f"   Client ID: {self.client_id}")
            if 'consent_id' in response:
                print(f"   Consent ID: {response['consent_id']}")
        
        return success, response

    def test_client_consent_validation(self):
        """Test client consent validation"""
        if not self.client_id:
            print("⚠️  Skipping consent validation - no client ID available")
            return True, {}
        
        success, response = self.run_test(
            "Client Consent Validation",
            "GET",
            f"client/consent/check/{self.client_id}",
            200
        )
        
        if success and 'has_consent' in response:
            has_consent = response['has_consent']
            print(f"   Consent status: {has_consent}")
            if has_consent:
                print(f"   ✅ Client consent properly validated")
            else:
                print(f"   ❌ Client consent not found or invalid")
        
        return success, response

    def test_compliant_contract_generation(self):
        """Test compliant contract generation with exact user scenario"""
        if not self.client_id:
            print("⚠️  Generating client ID for contract generation")
            timestamp = int(time.time() * 1000)
            random_string = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=11))
            self.client_id = f"client_{timestamp}_{random_string}"
        
        # Exact user scenario from review request
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
            "Compliant Contract Generation (Exact User Scenario)",
            "POST",
            "generate-contract-compliant",
            200,
            contract_data,
            timeout=90  # Allow more time for compliance processing
        )
        
        if success:
            print(f"   ✅ Contract generation successful")
            
            # Extract review ID from suggestions
            suggestions = response.get('suggestions', [])
            print(f"   Suggestions received: {len(suggestions)}")
            
            for suggestion in suggestions:
                print(f"   - {suggestion}")
                # Look for review ID pattern in suggestions
                if 'ID:' in suggestion:
                    import re
                    review_id_match = re.search(r'ID:\s*([a-f0-9-]+)', suggestion)
                    if review_id_match:
                        self.review_id = review_id_match.group(1)
                        print(f"   ✅ Review ID extracted: {self.review_id}")
                        break
            
            if not self.review_id:
                print(f"   ⚠️  Could not extract review ID from suggestions")
                # Try to find it in other response fields
                if 'review_id' in response:
                    self.review_id = response['review_id']
                    print(f"   ✅ Review ID found in response: {self.review_id}")
        
        return success, response

    def test_review_status_initial(self):
        """Test initial review status - should be pending with 0% progress"""
        if not self.review_id:
            print("⚠️  Skipping initial review status test - no review ID available")
            return True, {}
        
        success, response = self.run_test(
            "Initial Review Status Check",
            "GET",
            f"attorney/review/status/{self.review_id}",
            200
        )
        
        if success:
            status = response.get('status', 'unknown')
            progress = response.get('progress_percentage', -1)
            assigned_attorney = response.get('assigned_attorney')
            
            print(f"   Status: {status}")
            print(f"   Progress: {progress}%")
            print(f"   Assigned Attorney: {assigned_attorney}")
            
            # Check if this matches the user's reported issue
            if status == 'pending' and progress == 0:
                print(f"   🚨 ISSUE CONFIRMED: Review is stuck in 'pending' status with 0% progress")
                if not assigned_attorney:
                    print(f"   🚨 ROOT CAUSE: No attorney assigned to review")
                else:
                    print(f"   ✅ Attorney is assigned: {assigned_attorney}")
            elif status == 'in_review' and progress > 0:
                print(f"   ✅ Review has progressed to 'in_review' with {progress}% progress")
            else:
                print(f"   ⚠️  Unexpected status/progress combination: {status}/{progress}%")
        
        return success, response

    def test_attorney_creation_system(self):
        """Test attorney creation system to verify it's working"""
        # Create a test attorney
        attorney_data = {
            "email": f"test.attorney.{int(time.time())}@legalmate.test",
            "first_name": "Test",
            "last_name": "Attorney",
            "bar_number": f"BAR{random.randint(100000, 999999)}",
            "jurisdiction": "US",
            "role": "reviewing_attorney",
            "specializations": ["contract_law", "business_law"],
            "years_experience": 5,
            "password": "TestPassword123!"
        }
        
        success, response = self.run_test(
            "Attorney Creation System Test",
            "POST",
            "attorney/create",
            200,
            attorney_data
        )
        
        if success:
            print(f"   ✅ Attorney creation system is working")
            attorney_id = response.get('attorney_id')
            if attorney_id:
                print(f"   Attorney ID: {attorney_id}")
        else:
            print(f"   🚨 CRITICAL: Attorney creation system is failing")
            print(f"   This could be the root cause of review assignment failures")
        
        return success, response

    def test_review_status_progression_monitoring(self):
        """Monitor review status over time to check for progression"""
        if not self.review_id:
            print("⚠️  Skipping progression monitoring - no review ID available")
            return True, {}
        
        print(f"\n🔍 Monitoring Review Status Progression for 60 seconds...")
        print(f"   Review ID: {self.review_id}")
        
        progression_data = []
        
        for i in range(6):  # Check every 10 seconds for 1 minute
            success, response = self.run_test(
                f"Review Status Check #{i+1}",
                "GET",
                f"attorney/review/status/{self.review_id}",
                200
            )
            
            if success:
                status = response.get('status', 'unknown')
                progress = response.get('progress_percentage', -1)
                assigned_attorney = response.get('assigned_attorney')
                estimated_completion = response.get('estimated_completion')
                
                progression_data.append({
                    'timestamp': datetime.now().isoformat(),
                    'status': status,
                    'progress': progress,
                    'assigned_attorney': assigned_attorney,
                    'estimated_completion': estimated_completion
                })
                
                print(f"   Check #{i+1}: Status={status}, Progress={progress}%, Attorney={assigned_attorney}")
                
                # Check for progression
                if i > 0:
                    prev_data = progression_data[i-1]
                    if (status != prev_data['status'] or 
                        progress != prev_data['progress']):
                        print(f"   ✅ PROGRESSION DETECTED: Status changed from {prev_data['status']} to {status}, Progress from {prev_data['progress']}% to {progress}%")
                
                # If we see progression to in_review, that's good
                if status == 'in_review' and progress > 0:
                    print(f"   ✅ SUCCESS: Review has progressed to 'in_review' with {progress}% progress")
                    break
                elif status == 'pending' and progress == 0:
                    print(f"   🚨 ISSUE PERSISTS: Review still stuck in 'pending' with 0% progress")
            
            if i < 5:  # Don't wait after the last check
                print(f"   Waiting 10 seconds before next check...")
                time.sleep(10)
        
        # Analyze progression data
        print(f"\n📊 Progression Analysis:")
        if len(progression_data) > 1:
            first_check = progression_data[0]
            last_check = progression_data[-1]
            
            status_changed = first_check['status'] != last_check['status']
            progress_changed = first_check['progress'] != last_check['progress']
            attorney_assigned = last_check['assigned_attorney'] is not None
            
            print(f"   Status changed: {status_changed} ({first_check['status']} → {last_check['status']})")
            print(f"   Progress changed: {progress_changed} ({first_check['progress']}% → {last_check['progress']}%)")
            print(f"   Attorney assigned: {attorney_assigned}")
            
            if not status_changed and not progress_changed:
                print(f"   🚨 CRITICAL ISSUE: No progression detected over 60 seconds")
                print(f"   This confirms the user's report of static progress")
                if not attorney_assigned:
                    print(f"   🚨 ROOT CAUSE: Attorney assignment system is not working")
                else:
                    print(f"   ⚠️  Attorney is assigned but progress is not advancing")
            else:
                print(f"   ✅ Progression detected - system is working correctly")
        
        return True, progression_data

    def test_cleanup_stuck_reviews(self):
        """Test the cleanup stuck reviews endpoint"""
        success, response = self.run_test(
            "Cleanup Stuck Reviews",
            "POST",
            "attorney/review/cleanup-stuck",
            200,
            {}
        )
        
        if success:
            fixed_count = response.get('fixed_count', 0)
            failed_count = response.get('failed_count', 0)
            details = response.get('details', [])
            
            print(f"   Fixed reviews: {fixed_count}")
            print(f"   Failed reviews: {failed_count}")
            
            if fixed_count > 0:
                print(f"   ✅ Cleanup system fixed {fixed_count} stuck reviews")
                for detail in details[:3]:  # Show first 3 details
                    print(f"   - {detail}")
            else:
                print(f"   ℹ️  No stuck reviews found to fix")
        
        return success, response

    def test_specific_review_id_from_user(self):
        """Test the specific review ID mentioned by the user"""
        user_review_id = "b57f7ca3-24c1-4769-878b-afbbcf37814f"
        
        success, response = self.run_test(
            f"User's Specific Review ID Test",
            "GET",
            f"attorney/review/status/{user_review_id}",
            200
        )
        
        if success:
            status = response.get('status', 'unknown')
            progress = response.get('progress_percentage', -1)
            assigned_attorney = response.get('assigned_attorney')
            
            print(f"   User's Review Status: {status}")
            print(f"   User's Review Progress: {progress}%")
            print(f"   User's Review Attorney: {assigned_attorney}")
            
            if status == 'pending' and progress == 0:
                print(f"   🚨 CONFIRMED: User's review is stuck in pending with 0% progress")
            else:
                print(f"   ✅ User's review has progressed beyond the reported issue")
        else:
            print(f"   ⚠️  User's review ID not found (may have been cleaned up or deleted)")
        
        return success, response

    def run_comprehensive_investigation(self):
        """Run comprehensive investigation of the review status issue"""
        print("🚨 CRITICAL ISSUE INVESTIGATION: ReviewStatus Component Progress Issue")
        print("=" * 80)
        print("User Report: ReviewStatus shows 'Pending Review' with 0% progress that never increases")
        print("Expected: Reviews should progress from 'pending' (0%) → 'in_review' (25-95%)")
        print("=" * 80)
        
        # Test 1: Client consent workflow
        print("\n📋 STEP 1: Testing Client Consent Workflow")
        consent_success, _ = self.test_client_consent_recording()
        if consent_success:
            self.test_client_consent_validation()
        
        # Test 2: Contract generation with exact user scenario
        print("\n📋 STEP 2: Testing Contract Generation (Exact User Scenario)")
        contract_success, _ = self.test_compliant_contract_generation()
        
        # Test 3: Initial review status
        print("\n📋 STEP 3: Testing Initial Review Status")
        if contract_success and self.review_id:
            self.test_review_status_initial()
        
        # Test 4: Attorney creation system
        print("\n📋 STEP 4: Testing Attorney Creation System")
        self.test_attorney_creation_system()
        
        # Test 5: Review status progression monitoring
        print("\n📋 STEP 5: Monitoring Review Status Progression")
        if self.review_id:
            self.test_review_status_progression_monitoring()
        
        # Test 6: Cleanup stuck reviews
        print("\n📋 STEP 6: Testing Cleanup Stuck Reviews")
        self.test_cleanup_stuck_reviews()
        
        # Test 7: User's specific review ID
        print("\n📋 STEP 7: Testing User's Specific Review ID")
        self.test_specific_review_id_from_user()
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎯 INVESTIGATION SUMMARY")
        print("=" * 80)
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if self.review_id:
            print(f"Generated Review ID: {self.review_id}")
        if self.client_id:
            print(f"Generated Client ID: {self.client_id}")
        
        print("\n🔍 KEY FINDINGS:")
        print("- Check the progression monitoring results above")
        print("- Look for 'ISSUE CONFIRMED' or 'ROOT CAUSE' messages")
        print("- Attorney assignment system status is critical")
        print("- Review progression over time indicates system health")
        
        return self.tests_passed, self.tests_run

if __name__ == "__main__":
    print("🚨 REVIEW STATUS INVESTIGATION - CRITICAL ISSUE TESTING")
    print("Testing the reported issue: ReviewStatus component shows 'Pending Review' with 0% progress")
    print("=" * 100)
    
    tester = ReviewStatusInvestigationTester()
    passed, total = tester.run_comprehensive_investigation()
    
    print(f"\n🏁 FINAL RESULT: {passed}/{total} tests passed ({(passed/total)*100:.1f}% success rate)")
    
    if passed == total:
        print("✅ All tests passed - System appears to be working correctly")
    else:
        print("❌ Some tests failed - Issues detected that may explain user's problem")
    
    sys.exit(0 if passed == total else 1)