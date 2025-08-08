#!/usr/bin/env python3
"""
ROOT CAUSE ANALYSIS: Static Progress and Overdue Status Investigation
Based on initial findings, investigating specific issues:
1. Attorney role validation issue (senior_attorney not valid)
2. Progress calculation stuck at 0% due to no attorney assignment
3. Time estimation showing future dates instead of "Overdue"
"""

import requests
import sys
import json
import time
import uuid
from datetime import datetime

class RootCauseAnalyzer:
    def __init__(self, base_url="https://027a3f35-a0bc-40cd-ad79-483d46c6126e.preview.emergentagent.com"):
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
                    return False, error_data
                except:
                    print(f"   Error: {response.text}")
                    return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_valid_attorney_roles(self):
        """Test creating attorneys with valid roles based on the enum"""
        valid_roles = [
            "supervising_attorney",
            "reviewing_attorney", 
            "senior_partner",
            "compliance_officer"
        ]
        
        print(f"\n🔍 Testing Valid Attorney Roles...")
        
        for role in valid_roles:
            attorney_data = {
                "email": f"test.{role}.{int(time.time())}@legalmate.test",
                "first_name": "Test",
                "last_name": f"{role.replace('_', ' ').title()}",
                "bar_number": f"BAR{int(time.time())}{role[:3].upper()}",
                "jurisdiction": "US",
                "role": role,
                "specializations": ["contract_law", "business_law"],
                "years_experience": 10,
                "password": "TestPassword123!"
            }
            
            success, response = self.run_test(
                f"Create Attorney with {role} role", 
                "POST", 
                "attorney/create", 
                200, 
                attorney_data
            )
            
            if success and 'attorney' in response:
                if not self.attorney_id:  # Store first successful attorney
                    self.attorney_id = response['attorney'].get('id')
                    print(f"   ✅ Stored Attorney ID: {self.attorney_id} (Role: {role})")
                print(f"   ✅ {role} role accepted")
            else:
                print(f"   ❌ {role} role rejected")
        
        return self.attorney_id is not None

    def test_attorney_assignment_workflow(self):
        """Test the complete attorney assignment workflow"""
        if not self.attorney_id:
            print("❌ No attorney available - skipping assignment workflow test")
            return False, {}
        
        # Step 1: Record client consent
        self.client_id = f"client_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        consent_data = {
            "client_id": self.client_id,
            "consent_text": "I consent to attorney supervision for document review and legal compliance.",
            "ip_address": "192.168.1.100"
        }
        
        consent_success, _ = self.run_test(
            "Record Client Consent for Assignment Test", 
            "POST", 
            "client/consent", 
            200, 
            consent_data
        )
        
        if not consent_success:
            return False, {}
        
        # Step 2: Submit document for review
        document_data = {
            "document_content": "**ATTORNEY ASSIGNMENT TEST CONTRACT**\n\nThis contract is specifically designed to test attorney assignment and progress calculation functionality in the document review system.",
            "document_type": "contract",
            "client_id": self.client_id,
            "original_request": {
                "contract_type": "service_agreement",
                "jurisdiction": "US",
                "parties": {
                    "party1_name": "Assignment Test Corp",
                    "party2_name": "Progress Calculation LLC"
                }
            },
            "priority": "high"  # High priority to trigger assignment
        }
        
        submit_success, submit_response = self.run_test(
            "Submit Document for Assignment Test", 
            "POST", 
            "attorney/review/submit", 
            200, 
            document_data
        )
        
        if not submit_success or 'review_id' not in submit_response:
            return False, {}
        
        self.review_id = submit_response['review_id']
        print(f"   Assignment Test Review ID: {self.review_id}")
        
        # Step 3: Check initial status
        initial_success, initial_response = self.run_test(
            "Check Initial Review Status", 
            "GET", 
            f"attorney/review/status/{self.review_id}", 
            200
        )
        
        if initial_success:
            print(f"   Initial Status: {initial_response.get('status')}")
            print(f"   Initial Progress: {initial_response.get('progress_percentage')}%")
            print(f"   Initial Attorney: {initial_response.get('attorney')}")
            print(f"   Initial Estimated Completion: {initial_response.get('estimated_completion')}")
        
        # Step 4: Wait and check if assignment happens automatically
        print(f"\n   Waiting 5 seconds for potential auto-assignment...")
        time.sleep(5)
        
        auto_success, auto_response = self.run_test(
            "Check Auto-Assignment Status", 
            "GET", 
            f"attorney/review/status/{self.review_id}", 
            200
        )
        
        if auto_success:
            print(f"   After 5s Status: {auto_response.get('status')}")
            print(f"   After 5s Progress: {auto_response.get('progress_percentage')}%")
            print(f"   After 5s Attorney: {auto_response.get('attorney')}")
            print(f"   After 5s Estimated Completion: {auto_response.get('estimated_completion')}")
            
            # Check if assignment occurred
            if auto_response.get('attorney') and auto_response.get('status') == 'in_review':
                print(f"   ✅ Auto-assignment successful!")
                return True, auto_response
            else:
                print(f"   ❌ Auto-assignment did not occur")
        
        return False, {}

    def test_cleanup_and_manual_assignment(self):
        """Test cleanup stuck reviews and manual assignment"""
        if not self.review_id:
            print("❌ No review ID available - skipping cleanup test")
            return False, {}
        
        # Test cleanup stuck reviews
        cleanup_success, cleanup_response = self.run_test(
            "Cleanup Stuck Reviews", 
            "POST", 
            "attorney/review/cleanup-stuck", 
            200,
            {}
        )
        
        if cleanup_success:
            print(f"   Cleanup Fixed Count: {cleanup_response.get('fixed_count', 0)}")
            print(f"   Cleanup Message: {cleanup_response.get('message', 'No message')}")
            
            # Check status after cleanup
            time.sleep(2)
            
            post_cleanup_success, post_cleanup_response = self.run_test(
                "Check Status After Cleanup", 
                "GET", 
                f"attorney/review/status/{self.review_id}", 
                200
            )
            
            if post_cleanup_success:
                print(f"   Post-Cleanup Status: {post_cleanup_response.get('status')}")
                print(f"   Post-Cleanup Progress: {post_cleanup_response.get('progress_percentage')}%")
                print(f"   Post-Cleanup Attorney: {post_cleanup_response.get('attorney')}")
                print(f"   Post-Cleanup Estimated Completion: {post_cleanup_response.get('estimated_completion')}")
                
                # Check if cleanup fixed the assignment
                if post_cleanup_response.get('attorney') and post_cleanup_response.get('status') == 'in_review':
                    print(f"   ✅ Cleanup successfully assigned attorney!")
                    return True, post_cleanup_response
                elif post_cleanup_response.get('progress_percentage', 0) > 0:
                    print(f"   ✅ Cleanup improved progress calculation!")
                    return True, post_cleanup_response
        
        return False, {}

    def test_progress_calculation_over_time(self):
        """Test progress calculation changes over time for in_review status"""
        if not self.review_id:
            print("❌ No review ID available - skipping progress calculation test")
            return False, {}
        
        print(f"\n🔍 Testing Progress Calculation Over Time...")
        
        # Get current status
        current_success, current_response = self.run_test(
            "Current Review Status for Progress Test", 
            "GET", 
            f"attorney/review/status/{self.review_id}", 
            200
        )
        
        if not current_success:
            return False, {}
        
        current_status = current_response.get('status')
        current_progress = current_response.get('progress_percentage', 0)
        
        print(f"   Current Status: {current_status}")
        print(f"   Current Progress: {current_progress}%")
        
        if current_status != 'in_review':
            print(f"   ⚠️  Review is not in 'in_review' status - progress calculation may not be dynamic")
            return False, {}
        
        # Track progress over multiple calls
        progress_tracking = []
        for i in range(5):
            time.sleep(2)  # Wait 2 seconds between calls
            
            track_success, track_response = self.run_test(
                f"Progress Tracking Call {i+1}", 
                "GET", 
                f"attorney/review/status/{self.review_id}", 
                200
            )
            
            if track_success:
                progress = track_response.get('progress_percentage', 0)
                estimated_completion = track_response.get('estimated_completion')
                
                progress_tracking.append({
                    'call': i+1,
                    'progress': progress,
                    'estimated_completion': estimated_completion,
                    'timestamp': datetime.now().isoformat()
                })
                
                print(f"   Call {i+1}: Progress {progress}%, Completion: {estimated_completion}")
        
        # Analyze progress changes
        if len(progress_tracking) >= 2:
            progress_values = [p['progress'] for p in progress_tracking]
            unique_progress = set(progress_values)
            
            print(f"\n   📊 PROGRESS ANALYSIS:")
            print(f"   Progress Values: {progress_values}")
            
            if len(unique_progress) == 1:
                print(f"   🚨 ISSUE: Progress is STATIC at {progress_values[0]}%")
                
                # Check if it's the expected static value for the status
                if current_status == 'pending' and progress_values[0] == 0:
                    print(f"   ℹ️  Static 0% is expected for 'pending' status")
                elif current_status == 'in_review' and progress_values[0] >= 25:
                    print(f"   ℹ️  Static progress in 'in_review' may indicate time-based calculation issue")
                else:
                    print(f"   🚨 Unexpected static progress for status '{current_status}'")
            else:
                print(f"   ✅ Progress is DYNAMIC: {list(unique_progress)}")
                
                # Check if progress is increasing
                if progress_values[-1] > progress_values[0]:
                    print(f"   ✅ Progress is INCREASING over time")
                else:
                    print(f"   ⚠️  Progress is not consistently increasing")
        
        return True, progress_tracking

    def test_overdue_status_investigation(self):
        """Investigate why estimated completion shows future dates instead of 'Overdue'"""
        if not self.review_id:
            print("❌ No review ID available - skipping overdue investigation")
            return False, {}
        
        print(f"\n🔍 Investigating 'Overdue' Status Issue...")
        
        # Get current review status
        status_success, status_response = self.run_test(
            "Review Status for Overdue Investigation", 
            "GET", 
            f"attorney/review/status/{self.review_id}", 
            200
        )
        
        if not status_success:
            return False, {}
        
        estimated_completion = status_response.get('estimated_completion')
        created_at = status_response.get('created_at')
        status = status_response.get('status')
        
        print(f"   Review Status: {status}")
        print(f"   Created At: {created_at}")
        print(f"   Estimated Completion: {estimated_completion}")
        
        # Parse dates to check if overdue
        try:
            if estimated_completion and estimated_completion != "Overdue":
                from datetime import datetime
                completion_time = datetime.fromisoformat(estimated_completion.replace('Z', '+00:00'))
                current_time = datetime.now(completion_time.tzinfo)
                
                print(f"   Current Time: {current_time.isoformat()}")
                print(f"   Completion Time: {completion_time.isoformat()}")
                
                if current_time > completion_time:
                    print(f"   🚨 ISSUE FOUND: Review IS overdue but shows future completion time!")
                    print(f"   🚨 Expected: 'Overdue', Got: {estimated_completion}")
                    
                    # Calculate how overdue it is
                    overdue_duration = current_time - completion_time
                    print(f"   🚨 Overdue by: {overdue_duration}")
                    
                    return False, {
                        'issue': 'overdue_not_detected',
                        'expected': 'Overdue',
                        'actual': estimated_completion,
                        'overdue_by': str(overdue_duration)
                    }
                else:
                    print(f"   ✅ Review is not yet overdue")
                    time_remaining = completion_time - current_time
                    print(f"   ✅ Time remaining: {time_remaining}")
            elif estimated_completion == "Overdue":
                print(f"   ✅ Correctly showing 'Overdue' status")
            else:
                print(f"   ❌ No estimated completion time provided")
                
        except Exception as e:
            print(f"   ❌ Error parsing completion time: {e}")
            return False, {'error': str(e)}
        
        return True, status_response

    def run_root_cause_analysis(self):
        """Run comprehensive root cause analysis"""
        print("="*80)
        print("🔍 ROOT CAUSE ANALYSIS: Static Progress & Overdue Status Issues")
        print("="*80)
        print("INVESTIGATION PLAN:")
        print("1. Fix attorney role validation issue")
        print("2. Test attorney assignment workflow")
        print("3. Test progress calculation over time")
        print("4. Investigate 'Overdue' status detection")
        print("="*80)
        
        # Test 1: Fix Attorney Role Issue
        print(f"\n{'='*20} TEST 1: ATTORNEY ROLE VALIDATION {'='*20}")
        attorney_success = self.test_valid_attorney_roles()
        
        if not attorney_success:
            print(f"\n❌ CRITICAL: Cannot proceed without valid attorney")
            return False
        
        # Test 2: Attorney Assignment Workflow
        print(f"\n{'='*20} TEST 2: ATTORNEY ASSIGNMENT WORKFLOW {'='*20}")
        assignment_success, assignment_response = self.test_attorney_assignment_workflow()
        
        # Test 3: Cleanup if assignment failed
        if not assignment_success:
            print(f"\n{'='*20} TEST 3: CLEANUP AND MANUAL ASSIGNMENT {'='*20}")
            cleanup_success, cleanup_response = self.test_cleanup_and_manual_assignment()
        
        # Test 4: Progress Calculation Over Time
        print(f"\n{'='*20} TEST 4: PROGRESS CALCULATION ANALYSIS {'='*20}")
        progress_success, progress_data = self.test_progress_calculation_over_time()
        
        # Test 5: Overdue Status Investigation
        print(f"\n{'='*20} TEST 5: OVERDUE STATUS INVESTIGATION {'='*20}")
        overdue_success, overdue_data = self.test_overdue_status_investigation()
        
        # Final Analysis
        print(f"\n{'='*20} ROOT CAUSE ANALYSIS RESULTS {'='*20}")
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        print(f"\n🎯 KEY FINDINGS:")
        
        # Finding 1: Attorney Role Issue
        if attorney_success:
            print(f"✅ FIXED: Attorney role validation - use 'reviewing_attorney' instead of 'senior_attorney'")
        else:
            print(f"❌ ISSUE: Attorney role validation still failing")
        
        # Finding 2: Assignment Issue
        if assignment_success or (not assignment_success and cleanup_success):
            print(f"✅ WORKING: Attorney assignment system functional")
        else:
            print(f"❌ ISSUE: Attorney assignment system not working properly")
        
        # Finding 3: Progress Calculation
        if progress_success and progress_data:
            progress_values = [p['progress'] for p in progress_data]
            if len(set(progress_values)) > 1:
                print(f"✅ WORKING: Progress calculation is dynamic")
            else:
                print(f"❌ ISSUE: Progress calculation is static at {progress_values[0] if progress_values else 'unknown'}%")
        
        # Finding 4: Overdue Detection
        if overdue_success and isinstance(overdue_data, dict) and overdue_data.get('issue') == 'overdue_not_detected':
            print(f"❌ ISSUE: 'Overdue' status not being detected properly")
        elif overdue_success:
            print(f"✅ WORKING: Time estimation working correctly")
        
        print(f"\n📋 RECOMMENDATIONS FOR MAIN AGENT:")
        print(f"1. Update attorney creation to use valid roles: 'reviewing_attorney', 'supervising_attorney', 'senior_partner', 'compliance_officer'")
        print(f"2. Ensure attorney assignment triggers status change from 'pending' to 'in_review'")
        print(f"3. Verify progress calculation uses assignment_date for time-based progression")
        print(f"4. Check if 'Overdue' detection logic compares current time with estimated completion")
        print(f"5. Run cleanup-stuck-reviews endpoint to fix existing unassigned reviews")

if __name__ == "__main__":
    analyzer = RootCauseAnalyzer()
    analyzer.run_root_cause_analysis()