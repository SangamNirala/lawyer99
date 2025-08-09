#!/usr/bin/env python3
"""
CRITICAL INVESTIGATION: Static Progress Percentage and Overdue Status Issues
Investigating the document review system for:
1. Static 50% progress issue
2. "Overdue" status problem
3. Attorney assignment and progress calculation
4. Time-based progression issues
"""

import requests
import sys
import json
import time
import uuid
from datetime import datetime

class ReviewStatusTester:
    def __init__(self, base_url="https://d7a1dca3-54ad-4826-aadd-9f393fe0628b.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.attorney_id = None
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

    def test_compliance_system_status(self):
        """Test compliance system status to ensure it's operational"""
        success, response = self.run_test(
            "Compliance System Status", 
            "GET", 
            "compliance/status", 
            200
        )
        
        if success:
            print(f"   Compliance Mode: {response.get('compliance_mode', 'Unknown')}")
            print(f"   Attorney Supervision Required: {response.get('attorney_supervision_required', 'Unknown')}")
            print(f"   System Status: {response.get('system_status', 'Unknown')}")
        
        return success, response

    def test_create_attorney(self):
        """Create an attorney for testing review assignments"""
        attorney_data = {
            "email": f"test.attorney.{int(time.time())}@legalmate.test",
            "first_name": "Test",
            "last_name": "Attorney",
            "bar_number": f"BAR{int(time.time())}",
            "jurisdiction": "US",
            "role": "senior_attorney",
            "specializations": ["contract_law", "business_law"],
            "years_experience": 10,
            "password": "TestPassword123!"
        }
        
        success, response = self.run_test(
            "Create Test Attorney", 
            "POST", 
            "attorney/create", 
            200, 
            attorney_data
        )
        
        if success and 'attorney' in response:
            self.attorney_id = response['attorney'].get('id')
            print(f"   Created Attorney ID: {self.attorney_id}")
            print(f"   Attorney Name: {response['attorney'].get('first_name')} {response['attorney'].get('last_name')}")
            print(f"   Specializations: {response['attorney'].get('specializations', [])}")
        
        return success, response

    def test_record_client_consent(self):
        """Record client consent for attorney supervision"""
        self.client_id = f"client_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        consent_data = {
            "client_id": self.client_id,
            "consent_text": "I consent to attorney supervision for document review and legal compliance.",
            "ip_address": "192.168.1.100",
            "user_agent": "ReviewStatusTester/1.0"
        }
        
        success, response = self.run_test(
            "Record Client Consent", 
            "POST", 
            "client/consent", 
            200, 
            consent_data
        )
        
        if success:
            print(f"   Client ID: {self.client_id}")
            print(f"   Consent ID: {response.get('consent_id', 'Unknown')}")
        
        return success, response

    def test_submit_document_for_review(self):
        """Submit a document for attorney review to create a review record"""
        if not self.client_id:
            print("❌ No client ID available - skipping document submission")
            return False, {}
        
        document_data = {
            "document_content": "**NON-DISCLOSURE AGREEMENT**\n\nThis Non-Disclosure Agreement is entered into for the purpose of protecting confidential business information during potential partnership discussions. The parties agree to maintain strict confidentiality regarding all shared proprietary information.",
            "document_type": "contract",
            "client_id": self.client_id,
            "original_request": {
                "contract_type": "NDA",
                "jurisdiction": "US",
                "parties": {
                    "party1_name": "ReviewTest Corp",
                    "party2_name": "Progress Tester"
                }
            },
            "priority": "normal"
        }
        
        success, response = self.run_test(
            "Submit Document for Review", 
            "POST", 
            "attorney/review/submit", 
            200, 
            document_data
        )
        
        if success and 'review_id' in response:
            self.review_id = response['review_id']
            print(f"   Review ID: {self.review_id}")
            print(f"   Status: {response.get('status', 'Unknown')}")
            print(f"   Assigned Attorney: {response.get('assigned_attorney_id', 'None')}")
            print(f"   Priority: {response.get('priority', 'Unknown')}")
        
        return success, response

    def test_review_status_endpoint(self):
        """Test the review status endpoint - MAIN FOCUS OF INVESTIGATION"""
        if not self.review_id:
            print("❌ No review ID available - skipping review status test")
            return False, {}
        
        success, response = self.run_test(
            "Review Status Endpoint", 
            "GET", 
            f"attorney/review/status/{self.review_id}", 
            200
        )
        
        if success:
            print(f"\n   📊 DETAILED REVIEW STATUS ANALYSIS:")
            print(f"   Review ID: {response.get('review_id', 'Unknown')}")
            print(f"   Status: {response.get('status', 'Unknown')}")
            print(f"   Progress Percentage: {response.get('progress_percentage', 'Unknown')}%")
            print(f"   Assigned Attorney: {response.get('assigned_attorney', 'None')}")
            print(f"   Priority: {response.get('priority', 'Unknown')}")
            print(f"   Created At: {response.get('created_at', 'Unknown')}")
            print(f"   Estimated Completion: {response.get('estimated_completion', 'Unknown')}")
            
            # CRITICAL ANALYSIS: Check for static 50% progress issue
            progress = response.get('progress_percentage')
            if progress == 50:
                print(f"   🚨 ISSUE CONFIRMED: Progress is static at 50% - this matches user report!")
            elif progress is not None:
                print(f"   ✅ Progress is {progress}% - not stuck at 50%")
            else:
                print(f"   ❌ No progress percentage returned")
            
            # Check for "Overdue" status issue
            estimated_completion = response.get('estimated_completion')
            if estimated_completion == "Overdue":
                print(f"   🚨 ISSUE CONFIRMED: Estimated completion shows 'Overdue' - matches user report!")
            elif estimated_completion:
                print(f"   ✅ Estimated completion: {estimated_completion}")
            else:
                print(f"   ❌ No estimated completion returned")
            
            # Check attorney assignment
            assigned_attorney = response.get('assigned_attorney')
            if assigned_attorney:
                print(f"   ✅ Attorney assigned: {assigned_attorney}")
            else:
                print(f"   🚨 POTENTIAL ISSUE: No attorney assigned - may cause progress calculation problems")
            
            # Check status progression
            status = response.get('status')
            if status == 'pending':
                print(f"   🚨 POTENTIAL ISSUE: Status still 'pending' - should move to 'in_review' when attorney assigned")
            elif status == 'in_review':
                print(f"   ✅ Status is 'in_review' - proper progression")
            else:
                print(f"   ℹ️  Status: {status}")
        
        return success, response

    def test_multiple_review_status_calls(self):
        """Test multiple calls to review status to check for dynamic progress"""
        if not self.review_id:
            print("❌ No review ID available - skipping multiple status calls test")
            return False, {}
        
        print(f"\n🔍 Testing Multiple Review Status Calls (checking for dynamic progress)...")
        
        results = []
        for i in range(3):
            print(f"\n   Call {i+1}/3:")
            success, response = self.run_test(
                f"Review Status Call {i+1}", 
                "GET", 
                f"attorney/review/status/{self.review_id}", 
                200
            )
            
            if success:
                progress = response.get('progress_percentage')
                estimated_completion = response.get('estimated_completion')
                status = response.get('status')
                
                results.append({
                    'call': i+1,
                    'progress': progress,
                    'estimated_completion': estimated_completion,
                    'status': status,
                    'timestamp': datetime.now().isoformat()
                })
                
                print(f"     Progress: {progress}%")
                print(f"     Estimated Completion: {estimated_completion}")
                print(f"     Status: {status}")
            
            # Wait 2 seconds between calls to see if progress changes
            if i < 2:
                time.sleep(2)
        
        # Analyze results for dynamic behavior
        print(f"\n   📈 PROGRESS ANALYSIS:")
        if len(results) >= 2:
            progress_values = [r['progress'] for r in results if r['progress'] is not None]
            if len(set(progress_values)) == 1:
                print(f"   🚨 CONFIRMED ISSUE: Progress is STATIC at {progress_values[0]}% across all calls")
                print(f"   🚨 This confirms the user-reported issue of static 50% progress")
            else:
                print(f"   ✅ Progress is DYNAMIC: {progress_values}")
            
            # Check estimated completion consistency
            completion_values = [r['estimated_completion'] for r in results]
            if all(c == "Overdue" for c in completion_values):
                print(f"   🚨 CONFIRMED ISSUE: All calls show 'Overdue' status")
            elif len(set(completion_values)) == 1:
                print(f"   ℹ️  Consistent estimated completion: {completion_values[0]}")
            else:
                print(f"   ✅ Estimated completion varies: {completion_values}")
        
        return len(results) > 0, results

    def test_specific_review_id_from_user_report(self):
        """Test the specific review ID mentioned in user report"""
        user_reported_review_id = "b57f7ca3-24c1-4769-878b-afbbcf37814f"
        
        print(f"\n🔍 Testing User-Reported Review ID: {user_reported_review_id}")
        
        success, response = self.run_test(
            "User-Reported Review ID Status", 
            "GET", 
            f"attorney/review/status/{user_reported_review_id}", 
            200
        )
        
        if success:
            print(f"   🎯 USER-REPORTED REVIEW ANALYSIS:")
            print(f"   Review ID: {response.get('review_id', 'Unknown')}")
            print(f"   Status: {response.get('status', 'Unknown')}")
            print(f"   Progress: {response.get('progress_percentage', 'Unknown')}%")
            print(f"   Estimated Completion: {response.get('estimated_completion', 'Unknown')}")
            print(f"   Assigned Attorney: {response.get('assigned_attorney', 'None')}")
            
            # Check for the specific issues reported
            if response.get('progress_percentage') == 50:
                print(f"   🚨 CONFIRMED: This review shows 50% progress (user-reported issue)")
            if response.get('estimated_completion') == "Overdue":
                print(f"   🚨 CONFIRMED: This review shows 'Overdue' status (user-reported issue)")
        else:
            # If 404, the review might have been cleaned up or doesn't exist
            if response == {} and not success:
                print(f"   ℹ️  Review not found (404) - may have been cleaned up or ID incorrect")
        
        return success, response

    def test_create_new_review_and_track_progress(self):
        """Create a fresh review and immediately track its progress"""
        print(f"\n🔍 Creating Fresh Review to Track Progress from Start...")
        
        # Create new client ID for this test
        fresh_client_id = f"fresh_client_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Record consent for new client
        consent_data = {
            "client_id": fresh_client_id,
            "consent_text": "Fresh consent for progress tracking test",
            "ip_address": "192.168.1.101"
        }
        
        consent_success, _ = self.run_test(
            "Fresh Client Consent", 
            "POST", 
            "client/consent", 
            200, 
            consent_data
        )
        
        if not consent_success:
            print("❌ Failed to record fresh client consent")
            return False, {}
        
        # Submit fresh document
        fresh_document_data = {
            "document_content": "**FRESH CONTRACT FOR PROGRESS TRACKING**\n\nThis is a fresh contract specifically created to test progress calculation and time estimation functionality.",
            "document_type": "contract",
            "client_id": fresh_client_id,
            "original_request": {
                "contract_type": "service_agreement",
                "jurisdiction": "US"
            },
            "priority": "high"
        }
        
        submit_success, submit_response = self.run_test(
            "Submit Fresh Document", 
            "POST", 
            "attorney/review/submit", 
            200, 
            fresh_document_data
        )
        
        if not submit_success or 'review_id' not in submit_response:
            print("❌ Failed to submit fresh document")
            return False, {}
        
        fresh_review_id = submit_response['review_id']
        print(f"   Fresh Review ID: {fresh_review_id}")
        
        # Track progress immediately and over time
        progress_tracking = []
        for i in range(4):
            print(f"\n   Progress Check {i+1}/4 (at T+{i*3} seconds):")
            
            status_success, status_response = self.run_test(
                f"Fresh Review Status Check {i+1}", 
                "GET", 
                f"attorney/review/status/{fresh_review_id}", 
                200
            )
            
            if status_success:
                progress = status_response.get('progress_percentage')
                estimated_completion = status_response.get('estimated_completion')
                status = status_response.get('status')
                assigned_attorney = status_response.get('assigned_attorney')
                
                progress_tracking.append({
                    'check': i+1,
                    'time_offset': i*3,
                    'progress': progress,
                    'estimated_completion': estimated_completion,
                    'status': status,
                    'assigned_attorney': assigned_attorney,
                    'timestamp': datetime.now().isoformat()
                })
                
                print(f"     Progress: {progress}%")
                print(f"     Status: {status}")
                print(f"     Estimated Completion: {estimated_completion}")
                print(f"     Assigned Attorney: {assigned_attorney}")
            
            # Wait 3 seconds between checks (except for last iteration)
            if i < 3:
                time.sleep(3)
        
        # Analyze fresh review progress
        print(f"\n   📊 FRESH REVIEW PROGRESS ANALYSIS:")
        if len(progress_tracking) >= 2:
            progress_values = [p['progress'] for p in progress_tracking if p['progress'] is not None]
            status_values = [p['status'] for p in progress_tracking]
            completion_values = [p['estimated_completion'] for p in progress_tracking]
            
            print(f"   Progress Values: {progress_values}")
            print(f"   Status Values: {status_values}")
            print(f"   Completion Values: {completion_values}")
            
            # Check for static progress issue
            if len(set(progress_values)) == 1 and progress_values[0] == 50:
                print(f"   🚨 CRITICAL ISSUE: Fresh review also stuck at 50% progress!")
                print(f"   🚨 This indicates a systemic issue with progress calculation")
            elif len(set(progress_values)) == 1:
                print(f"   🚨 ISSUE: Progress is static at {progress_values[0]}%")
            else:
                print(f"   ✅ Progress is dynamic: {progress_values}")
            
            # Check for overdue issue
            if all(c == "Overdue" for c in completion_values):
                print(f"   🚨 CRITICAL ISSUE: Fresh review immediately shows 'Overdue'!")
                print(f"   🚨 This indicates a systemic issue with time estimation")
            
            # Check attorney assignment progression
            attorney_assignments = [p['assigned_attorney'] for p in progress_tracking]
            if any(attorney_assignments) and not all(attorney_assignments):
                print(f"   ✅ Attorney assignment occurred during tracking")
            elif not any(attorney_assignments):
                print(f"   🚨 ISSUE: No attorney assigned throughout tracking period")
        
        return True, {
            'fresh_review_id': fresh_review_id,
            'progress_tracking': progress_tracking
        }

    def run_comprehensive_review_status_investigation(self):
        """Run comprehensive investigation of review status issues"""
        print("="*80)
        print("🔍 COMPREHENSIVE REVIEW STATUS INVESTIGATION")
        print("="*80)
        print("OBJECTIVES:")
        print("1. Test review status API for static 50% progress issue")
        print("2. Investigate attorney assignment and status progression")
        print("3. Test progress calculation and time estimation")
        print("4. Identify root cause of static progress and 'Overdue' status")
        print("="*80)
        
        # Test 1: System Status
        print(f"\n{'='*20} TEST 1: SYSTEM STATUS {'='*20}")
        self.test_compliance_system_status()
        
        # Test 2: Create Attorney
        print(f"\n{'='*20} TEST 2: ATTORNEY CREATION {'='*20}")
        self.test_create_attorney()
        
        # Test 3: Client Consent
        print(f"\n{'='*20} TEST 3: CLIENT CONSENT {'='*20}")
        self.test_record_client_consent()
        
        # Test 4: Document Submission
        print(f"\n{'='*20} TEST 4: DOCUMENT SUBMISSION {'='*20}")
        self.test_submit_document_for_review()
        
        # Test 5: Review Status (Main Investigation)
        print(f"\n{'='*20} TEST 5: REVIEW STATUS INVESTIGATION {'='*20}")
        self.test_review_status_endpoint()
        
        # Test 6: Multiple Status Calls
        print(f"\n{'='*20} TEST 6: DYNAMIC PROGRESS CHECK {'='*20}")
        self.test_multiple_review_status_calls()
        
        # Test 7: User-Reported Review ID
        print(f"\n{'='*20} TEST 7: USER-REPORTED REVIEW ID {'='*20}")
        self.test_specific_review_id_from_user_report()
        
        # Test 8: Fresh Review Tracking
        print(f"\n{'='*20} TEST 8: FRESH REVIEW PROGRESS TRACKING {'='*20}")
        self.test_create_new_review_and_track_progress()
        
        # Final Summary
        print(f"\n{'='*20} INVESTIGATION SUMMARY {'='*20}")
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        print(f"\n🎯 ROOT CAUSE ANALYSIS FINDINGS:")
        print(f"- Review Status API endpoint accessibility: {'✅ Working' if self.tests_passed > 0 else '❌ Issues'}")
        print(f"- Attorney assignment system: {'✅ Functional' if self.attorney_id else '❌ Issues'}")
        print(f"- Document submission workflow: {'✅ Working' if self.review_id else '❌ Issues'}")
        
        print(f"\n📋 RECOMMENDATIONS:")
        print(f"1. Investigate _calculate_progress_percentage method implementation")
        print(f"2. Check _calculate_estimated_completion time calculation logic")
        print(f"3. Verify attorney assignment triggers proper status transitions")
        print(f"4. Review time-based progression from 25% to 95%")
        print(f"5. Check if review creation timestamps are being set correctly")

if __name__ == "__main__":
    tester = ReviewStatusTester()
    tester.run_comprehensive_review_status_investigation()