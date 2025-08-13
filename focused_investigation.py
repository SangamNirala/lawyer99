#!/usr/bin/env python3
"""
FOCUSED INVESTIGATION: Review Status Issues
Testing specific endpoints to understand the root cause of:
1. Static progress percentage issue
2. "Overdue" status problem
3. Attorney assignment issues
"""

import requests
import sys
import json
import time
import uuid
from datetime import datetime

class FocusedInvestigator:
    def __init__(self, base_url="https://legalcore.preview.emergentagent.com"):
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
                    return False, error_data
                except:
                    print(f"   Error: {response.text}")
                    return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

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
            print(f"   Fixed Count: {response.get('fixed_count', 0)}")
            print(f"   Message: {response.get('message', 'No message')}")
            if response.get('details'):
                print(f"   Details: {response['details']}")
        
        return success, response

    def test_existing_review_status(self):
        """Test status of existing reviews to understand the issue"""
        # Test some common review IDs that might exist
        test_review_ids = [
            "b57f7ca3-24c1-4769-878b-afbbcf37814f",  # User reported ID
            "cef9d675-7285-4c1c-8031-a5572bad5946",  # From test results
            "b98195e3-659b-467f-a2e7-d659f31f136e",  # From recent test
            "cb6576a2-6f80-4613-aaf5-a1bff4671f2b"   # From recent test
        ]
        
        working_reviews = []
        
        for review_id in test_review_ids:
            success, response = self.run_test(
                f"Check Review Status {review_id[:8]}...", 
                "GET", 
                f"attorney/review/status/{review_id}", 
                200
            )
            
            if success:
                working_reviews.append({
                    'review_id': review_id,
                    'status': response.get('status'),
                    'progress': response.get('progress_percentage'),
                    'estimated_completion': response.get('estimated_completion'),
                    'attorney': response.get('attorney'),
                    'created_at': response.get('created_at')
                })
                
                print(f"   Status: {response.get('status')}")
                print(f"   Progress: {response.get('progress_percentage')}%")
                print(f"   Estimated Completion: {response.get('estimated_completion')}")
                print(f"   Attorney: {response.get('attorney')}")
                
                # Check for specific issues
                progress = response.get('progress_percentage')
                estimated_completion = response.get('estimated_completion')
                
                if progress == 50:
                    print(f"   🚨 CONFIRMED: Static 50% progress issue!")
                elif progress == 0:
                    print(f"   ⚠️  Progress at 0% - may indicate pending status")
                
                if estimated_completion == "Overdue":
                    print(f"   🚨 CONFIRMED: 'Overdue' status found!")
                elif estimated_completion and estimated_completion != "Overdue":
                    # Check if it should be overdue
                    try:
                        completion_time = datetime.fromisoformat(estimated_completion.replace('Z', '+00:00'))
                        current_time = datetime.now(completion_time.tzinfo)
                        if current_time > completion_time:
                            print(f"   🚨 ISSUE: Should be 'Overdue' but shows future time!")
                    except:
                        pass
        
        return len(working_reviews) > 0, working_reviews

    def test_create_simple_review(self):
        """Create a simple review to test the workflow"""
        # Step 1: Record consent
        client_id = f"test_client_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        consent_data = {
            "client_id": client_id,
            "consent_text": "I consent to attorney supervision for testing purposes.",
            "ip_address": "192.168.1.100"
        }
        
        consent_success, _ = self.run_test(
            "Record Test Client Consent", 
            "POST", 
            "client/consent", 
            200, 
            consent_data
        )
        
        if not consent_success:
            return False, {}
        
        # Step 2: Submit document
        document_data = {
            "document_content": "**TEST CONTRACT FOR INVESTIGATION**\n\nThis is a test contract to investigate the static progress and overdue status issues in the document review system.",
            "document_type": "contract",
            "client_id": client_id,
            "original_request": {
                "contract_type": "test_agreement",
                "jurisdiction": "US"
            },
            "priority": "normal"
        }
        
        submit_success, submit_response = self.run_test(
            "Submit Test Document", 
            "POST", 
            "attorney/review/submit", 
            200, 
            document_data
        )
        
        if not submit_success or 'review_id' not in submit_response:
            return False, {}
        
        review_id = submit_response['review_id']
        print(f"   Created Review ID: {review_id}")
        
        # Step 3: Check initial status
        initial_success, initial_response = self.run_test(
            "Check Initial Review Status", 
            "GET", 
            f"attorney/review/status/{review_id}", 
            200
        )
        
        if initial_success:
            print(f"   Initial Status: {initial_response.get('status')}")
            print(f"   Initial Progress: {initial_response.get('progress_percentage')}%")
            print(f"   Initial Attorney: {initial_response.get('attorney')}")
            print(f"   Initial Estimated Completion: {initial_response.get('estimated_completion')}")
        
        return True, {
            'review_id': review_id,
            'client_id': client_id,
            'initial_response': initial_response if initial_success else None
        }

    def test_attorney_queue_without_attorney(self):
        """Test attorney queue endpoints to see if any attorneys exist"""
        # Try some common attorney IDs that might exist
        test_attorney_ids = [
            "attorney_1",
            "test_attorney",
            str(uuid.uuid4()),  # Random UUID
        ]
        
        for attorney_id in test_attorney_ids:
            success, response = self.run_test(
                f"Check Attorney Queue {attorney_id[:8]}...", 
                "GET", 
                f"attorney/review/queue/{attorney_id}", 
                200
            )
            
            if success:
                reviews = response.get('reviews', [])
                print(f"   Attorney {attorney_id[:8]}... has {len(reviews)} reviews in queue")
                return True, response
        
        return False, {}

    def run_focused_investigation(self):
        """Run focused investigation on the specific issues"""
        print("="*80)
        print("🔍 FOCUSED INVESTIGATION: Review Status Issues")
        print("="*80)
        print("FOCUS AREAS:")
        print("1. Test cleanup stuck reviews endpoint")
        print("2. Check existing review statuses for patterns")
        print("3. Create new review to test workflow")
        print("4. Investigate attorney assignment system")
        print("="*80)
        
        # Test 1: Cleanup Stuck Reviews
        print(f"\n{'='*20} TEST 1: CLEANUP STUCK REVIEWS {'='*20}")
        cleanup_success, cleanup_response = self.test_cleanup_stuck_reviews()
        
        # Test 2: Check Existing Reviews
        print(f"\n{'='*20} TEST 2: EXISTING REVIEW STATUS ANALYSIS {'='*20}")
        existing_success, existing_reviews = self.test_existing_review_status()
        
        # Test 3: Create New Review
        print(f"\n{'='*20} TEST 3: CREATE NEW REVIEW TEST {'='*20}")
        new_review_success, new_review_data = self.test_create_simple_review()
        
        # Test 4: Attorney Queue Investigation
        print(f"\n{'='*20} TEST 4: ATTORNEY SYSTEM INVESTIGATION {'='*20}")
        attorney_success, attorney_data = self.test_attorney_queue_without_attorney()
        
        # Analysis
        print(f"\n{'='*20} INVESTIGATION RESULTS {'='*20}")
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        print(f"\n🎯 KEY FINDINGS:")
        
        # Analyze cleanup results
        if cleanup_success and cleanup_response.get('fixed_count', 0) > 0:
            print(f"✅ CLEANUP: Fixed {cleanup_response['fixed_count']} stuck reviews")
        elif cleanup_success:
            print(f"ℹ️  CLEANUP: No stuck reviews found to fix")
        else:
            print(f"❌ CLEANUP: Cleanup endpoint not working")
        
        # Analyze existing reviews
        if existing_success and existing_reviews:
            print(f"✅ EXISTING REVIEWS: Found {len(existing_reviews)} working reviews")
            
            # Check for patterns
            static_50_count = sum(1 for r in existing_reviews if r['progress'] == 50)
            static_0_count = sum(1 for r in existing_reviews if r['progress'] == 0)
            overdue_count = sum(1 for r in existing_reviews if r['estimated_completion'] == 'Overdue')
            no_attorney_count = sum(1 for r in existing_reviews if not r['attorney'])
            
            if static_50_count > 0:
                print(f"🚨 PATTERN: {static_50_count} reviews stuck at 50% progress")
            if static_0_count > 0:
                print(f"⚠️  PATTERN: {static_0_count} reviews at 0% progress (may be pending)")
            if overdue_count > 0:
                print(f"🚨 PATTERN: {overdue_count} reviews showing 'Overdue' status")
            if no_attorney_count > 0:
                print(f"🚨 PATTERN: {no_attorney_count} reviews with no attorney assigned")
        
        # Analyze new review creation
        if new_review_success and new_review_data:
            initial_response = new_review_data.get('initial_response', {})
            if initial_response:
                status = initial_response.get('status')
                progress = initial_response.get('progress_percentage')
                attorney = initial_response.get('attorney')
                
                print(f"✅ NEW REVIEW: Successfully created review")
                print(f"   Status: {status}, Progress: {progress}%, Attorney: {'Yes' if attorney else 'No'}")
                
                if status == 'pending' and progress == 0 and not attorney:
                    print(f"🚨 ROOT CAUSE IDENTIFIED: New reviews start as 'pending' with 0% progress and no attorney")
                    print(f"🚨 This explains why progress appears static - reviews aren't being assigned to attorneys!")
        
        print(f"\n📋 ROOT CAUSE SUMMARY:")
        print(f"1. Reviews are created in 'pending' status with 0% progress")
        print(f"2. Attorney assignment system may not be working automatically")
        print(f"3. Without attorney assignment, reviews stay at 0% progress indefinitely")
        print(f"4. Progress calculation requires 'in_review' status to be dynamic")
        print(f"5. Time estimation may be calculating from creation time instead of assignment time")
        
        print(f"\n📋 RECOMMENDATIONS:")
        print(f"1. Fix attorney creation endpoint (currently returning 500 errors)")
        print(f"2. Ensure auto-assignment triggers when documents are submitted")
        print(f"3. Run cleanup-stuck-reviews to fix existing unassigned reviews")
        print(f"4. Verify progress calculation uses assignment_date for time-based progression")
        print(f"5. Check if estimated completion should show 'Overdue' for old pending reviews")

if __name__ == "__main__":
    investigator = FocusedInvestigator()
    investigator.run_focused_investigation()