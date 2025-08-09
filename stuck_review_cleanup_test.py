import requests
import sys
import json
import time
import uuid
from datetime import datetime

class StuckReviewCleanupTester:
    def __init__(self, base_url="https://b2e17d7f-d715-45b1-acd0-e51dd70d439b.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.created_attorney_id = None
        self.created_review_id = None

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
            "Compliance System Status Check", 
            "GET", 
            "compliance/status", 
            200
        )
        
        if success:
            print(f"   Compliance Mode: {response.get('compliance_mode', 'Unknown')}")
            print(f"   Attorney Supervision Required: {response.get('attorney_supervision_required', 'Unknown')}")
            print(f"   System Status: {response.get('system_status', 'Unknown')}")
            
        return success, response

    def test_create_attorney_for_testing(self):
        """Create an attorney account for testing review assignment"""
        attorney_data = {
            "email": f"test.attorney.{int(time.time())}@legalmate.test",
            "first_name": "Test",
            "last_name": "Attorney",
            "bar_number": f"BAR{int(time.time())}",
            "jurisdiction": "US",
            "role": "reviewing_attorney",
            "specializations": ["contract_law", "business_law"],
            "years_experience": 5,
            "password": "TestPassword123!"
        }
        
        success, response = self.run_test(
            "Create Attorney for Testing", 
            "POST", 
            "attorney/create", 
            200, 
            attorney_data
        )
        
        if success and 'attorney_id' in response:
            self.created_attorney_id = response['attorney_id']
            print(f"   Created Attorney ID: {self.created_attorney_id}")
        
        return success, response

    def test_submit_document_for_review(self):
        """Submit a document for attorney review to create a review record"""
        document_data = {
            "document_content": "This is a test contract document that needs attorney review for compliance verification.",
            "document_type": "contract",
            "client_id": f"client_test_{int(time.time())}",
            "original_request": {
                "contract_type": "NDA",
                "jurisdiction": "US",
                "purpose": "Testing stuck review cleanup functionality"
            },
            "priority": "normal"
        }
        
        success, response = self.run_test(
            "Submit Document for Attorney Review", 
            "POST", 
            "attorney/review/submit", 
            200, 
            document_data
        )
        
        if success and 'review_id' in response:
            self.created_review_id = response['review_id']
            print(f"   Created Review ID: {self.created_review_id}")
            print(f"   Estimated Review Time: {response.get('estimated_review_time', 'Unknown')}")
            
        return success, response

    def test_review_status_endpoint(self):
        """Test the review status endpoint with created review ID"""
        if not self.created_review_id:
            print("⚠️  Skipping review status test - no review ID available")
            return True, {}
        
        success, response = self.run_test(
            f"Review Status Check - {self.created_review_id}", 
            "GET", 
            f"attorney/review/status/{self.created_review_id}", 
            200
        )
        
        if success:
            print(f"   Review ID: {response.get('review_id', 'Unknown')}")
            print(f"   Status: {response.get('status', 'Unknown')}")
            print(f"   Progress: {response.get('progress_percentage', 'Unknown')}%")
            print(f"   Created At: {response.get('created_at', 'Unknown')}")
            print(f"   Assigned Attorney: {response.get('assigned_attorney', 'None')}")
            print(f"   Priority: {response.get('priority', 'Unknown')}")
            
            # Check if this review might be stuck (0% progress, no attorney)
            progress = response.get('progress_percentage', 0)
            assigned_attorney = response.get('assigned_attorney')
            
            if progress == 0 and not assigned_attorney:
                print(f"   ⚠️  This review appears to be stuck (0% progress, no attorney assigned)")
            elif progress > 0:
                print(f"   ✅ Review has progress > 0% - not stuck")
            elif assigned_attorney:
                print(f"   ✅ Review has assigned attorney - not stuck")
                
        return success, response

    def test_specific_review_status(self):
        """Test the review status endpoint with the specific review ID from user's screenshot"""
        specific_review_id = "b57f7ca3-24c1-4769-878b-afbbcf37814f"
        
        success, response = self.run_test(
            f"Specific Review Status Check - {specific_review_id}", 
            "GET", 
            f"attorney/review/status/{specific_review_id}", 
            200
        )
        
        if success:
            print(f"   Review ID: {response.get('review_id', 'Unknown')}")
            print(f"   Status: {response.get('status', 'Unknown')}")
            print(f"   Progress: {response.get('progress_percentage', 'Unknown')}%")
            print(f"   Created At: {response.get('created_at', 'Unknown')}")
            print(f"   Assigned Attorney: {response.get('assigned_attorney', 'None')}")
            print(f"   Priority: {response.get('priority', 'Unknown')}")
            print(f"   Estimated Completion: {response.get('estimated_completion', 'Unknown')}")
            
            # Check if this review is stuck
            progress = response.get('progress_percentage', 0)
            assigned_attorney = response.get('assigned_attorney')
            status = response.get('status', '').lower()
            
            if progress == 0 and not assigned_attorney and status == 'pending':
                print(f"   🚨 CONFIRMED: This review is stuck (0% progress, no attorney, pending status)")
                print(f"   📋 This review is a candidate for cleanup script")
            elif progress > 0:
                print(f"   ✅ Review has progress > 0% - not stuck")
            elif assigned_attorney:
                print(f"   ✅ Review has assigned attorney - not stuck")
        else:
            # If 404, the review might not exist
            if response == {} and not success:
                print(f"   ⚠️  Review {specific_review_id} not found - may have been deleted or ID incorrect")
                
        return success, response

    def test_stuck_review_cleanup_endpoint(self):
        """Test the stuck review cleanup endpoint"""
        success, response = self.run_test(
            "Stuck Review Cleanup Script", 
            "POST", 
            "attorney/review/cleanup-stuck", 
            200
        )
        
        if success:
            print(f"   📊 CLEANUP RESULTS:")
            print(f"   Total Reviews Processed: {response.get('total_reviews_processed', 'Unknown')}")
            print(f"   Stuck Reviews Found: {response.get('stuck_reviews_found', 'Unknown')}")
            print(f"   Reviews Fixed: {response.get('reviews_fixed', 'Unknown')}")
            print(f"   Reviews Failed to Fix: {response.get('reviews_failed', 'Unknown')}")
            
            if 'fixed_reviews' in response and response['fixed_reviews']:
                print(f"   ✅ FIXED REVIEWS:")
                for fixed_review in response['fixed_reviews']:
                    print(f"     - Review ID: {fixed_review.get('review_id', 'Unknown')}")
                    print(f"       Assigned Attorney: {fixed_review.get('assigned_attorney_id', 'Unknown')}")
                    print(f"       New Status: {fixed_review.get('new_status', 'Unknown')}")
                    print(f"       New Progress: {fixed_review.get('new_progress', 'Unknown')}%")
            
            if 'failed_reviews' in response and response['failed_reviews']:
                print(f"   ❌ FAILED TO FIX:")
                for failed_review in response['failed_reviews']:
                    print(f"     - Review ID: {failed_review.get('review_id', 'Unknown')}")
                    print(f"       Reason: {failed_review.get('reason', 'Unknown')}")
            
            # Check if any reviews were actually fixed
            reviews_fixed = response.get('reviews_fixed', 0)
            if reviews_fixed > 0:
                print(f"   🎉 SUCCESS: {reviews_fixed} stuck reviews were fixed!")
            else:
                print(f"   ℹ️  No stuck reviews found to fix (this is good if system is working properly)")
                
        return success, response

    def test_review_status_after_cleanup(self):
        """Test review status after cleanup to verify improvements"""
        if not self.created_review_id:
            print("⚠️  Skipping post-cleanup review status test - no review ID available")
            return True, {}
        
        # Wait a moment for cleanup to take effect
        time.sleep(2)
        
        success, response = self.run_test(
            f"Review Status After Cleanup - {self.created_review_id}", 
            "GET", 
            f"attorney/review/status/{self.created_review_id}", 
            200
        )
        
        if success:
            print(f"   📈 POST-CLEANUP STATUS:")
            print(f"   Review ID: {response.get('review_id', 'Unknown')}")
            print(f"   Status: {response.get('status', 'Unknown')}")
            print(f"   Progress: {response.get('progress_percentage', 'Unknown')}%")
            print(f"   Assigned Attorney: {response.get('assigned_attorney', 'None')}")
            
            # Check if the review was improved by cleanup
            progress = response.get('progress_percentage', 0)
            assigned_attorney = response.get('assigned_attorney')
            status = response.get('status', '').lower()
            
            if progress > 0 and assigned_attorney and status != 'pending':
                print(f"   🎉 SUCCESS: Review was fixed by cleanup script!")
                print(f"   ✅ Progress increased from 0% to {progress}%")
                print(f"   ✅ Attorney assigned: {assigned_attorney}")
                print(f"   ✅ Status changed from 'pending' to '{status}'")
            elif progress == 0 and not assigned_attorney:
                print(f"   ⚠️  Review still appears stuck after cleanup")
            else:
                print(f"   ℹ️  Review status partially improved")
                
        return success, response

    def test_specific_review_status_after_cleanup(self):
        """Test the specific review ID after cleanup"""
        specific_review_id = "b57f7ca3-24c1-4769-878b-afbbcf37814f"
        
        # Wait a moment for cleanup to take effect
        time.sleep(2)
        
        success, response = self.run_test(
            f"Specific Review Status After Cleanup - {specific_review_id}", 
            "GET", 
            f"attorney/review/status/{specific_review_id}", 
            200
        )
        
        if success:
            print(f"   📈 POST-CLEANUP STATUS FOR USER'S REVIEW:")
            print(f"   Review ID: {response.get('review_id', 'Unknown')}")
            print(f"   Status: {response.get('status', 'Unknown')}")
            print(f"   Progress: {response.get('progress_percentage', 'Unknown')}%")
            print(f"   Assigned Attorney: {response.get('assigned_attorney', 'None')}")
            print(f"   Estimated Completion: {response.get('estimated_completion', 'Unknown')}")
            
            # Check if the user's specific review was fixed
            progress = response.get('progress_percentage', 0)
            assigned_attorney = response.get('assigned_attorney')
            status = response.get('status', '').lower()
            
            if progress > 0 and assigned_attorney and status != 'pending':
                print(f"   🎉 USER'S REVIEW FIXED: The stuck review was successfully resolved!")
                print(f"   ✅ Progress: 0% → {progress}%")
                print(f"   ✅ Attorney assigned: {assigned_attorney}")
                print(f"   ✅ Status: 'pending' → '{status}'")
                print(f"   ✅ Should no longer show 'Overdue' in frontend")
            elif progress == 0 and not assigned_attorney:
                print(f"   ❌ USER'S REVIEW STILL STUCK: Cleanup script did not fix this review")
            else:
                print(f"   ⚠️  USER'S REVIEW PARTIALLY IMPROVED: Some progress made")
        else:
            print(f"   ⚠️  Could not retrieve status for user's specific review after cleanup")
                
        return success, response

    def test_invalid_review_status(self):
        """Test review status endpoint with invalid review ID"""
        invalid_review_id = "invalid-review-id-12345"
        
        success, response = self.run_test(
            f"Invalid Review Status Check - {invalid_review_id}", 
            "GET", 
            f"attorney/review/status/{invalid_review_id}", 
            404  # Should return 404 for invalid ID
        )
        
        if success:
            print(f"   ✅ Correctly returned 404 for invalid review ID")
            
        return success, response

    def test_nonexistent_review_status(self):
        """Test review status endpoint with non-existent but valid UUID format"""
        nonexistent_review_id = str(uuid.uuid4())
        
        success, response = self.run_test(
            f"Non-existent Review Status Check - {nonexistent_review_id}", 
            "GET", 
            f"attorney/review/status/{nonexistent_review_id}", 
            404  # Should return 404 for non-existent review
        )
        
        if success:
            print(f"   ✅ Correctly returned 404 for non-existent review ID")
            
        return success, response

    def run_all_tests(self):
        """Run all stuck review cleanup and status tests"""
        print("🚀 Starting Stuck Review Cleanup and Status Testing...")
        print(f"🌐 Base URL: {self.base_url}")
        print(f"🔗 API URL: {self.api_url}")
        print("=" * 80)

        # Test sequence following the user's requirements
        test_results = []
        
        # 1. Check compliance system is operational
        print("\n📋 STEP 1: VERIFY COMPLIANCE SYSTEM STATUS")
        result = self.test_compliance_system_status()
        test_results.append(("Compliance System Status", result[0]))
        
        # 2. Create attorney for testing (if needed)
        print("\n👨‍⚖️ STEP 2: CREATE ATTORNEY FOR TESTING")
        result = self.test_create_attorney_for_testing()
        test_results.append(("Create Test Attorney", result[0]))
        
        # 3. Submit document for review to create test data
        print("\n📄 STEP 3: SUBMIT DOCUMENT FOR REVIEW")
        result = self.test_submit_document_for_review()
        test_results.append(("Submit Document for Review", result[0]))
        
        # 4. Test review status endpoint with created review
        print("\n📊 STEP 4: TEST REVIEW STATUS ENDPOINT")
        result = self.test_review_status_endpoint()
        test_results.append(("Review Status Endpoint", result[0]))
        
        # 5. Test specific review ID from user's screenshot
        print("\n🎯 STEP 5: TEST SPECIFIC REVIEW ID FROM USER")
        result = self.test_specific_review_status()
        test_results.append(("Specific Review Status", result[0]))
        
        # 6. **CRITICAL TEST**: Run stuck review cleanup script
        print("\n🔧 STEP 6: RUN STUCK REVIEW CLEANUP SCRIPT")
        result = self.test_stuck_review_cleanup_endpoint()
        test_results.append(("Stuck Review Cleanup", result[0]))
        
        # 7. Verify review status after cleanup
        print("\n📈 STEP 7: VERIFY REVIEW STATUS AFTER CLEANUP")
        result = self.test_review_status_after_cleanup()
        test_results.append(("Review Status After Cleanup", result[0]))
        
        # 8. Verify specific review status after cleanup
        print("\n🎯 STEP 8: VERIFY SPECIFIC REVIEW AFTER CLEANUP")
        result = self.test_specific_review_status_after_cleanup()
        test_results.append(("Specific Review After Cleanup", result[0]))
        
        # 9. Test error handling
        print("\n❌ STEP 9: TEST ERROR HANDLING")
        result = self.test_invalid_review_status()
        test_results.append(("Invalid Review Status", result[0]))
        
        result = self.test_nonexistent_review_status()
        test_results.append(("Non-existent Review Status", result[0]))

        # Print final results
        print("\n" + "=" * 80)
        print("📊 FINAL TEST RESULTS")
        print("=" * 80)
        
        for test_name, passed in test_results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status} - {test_name}")
        
        success_rate = (self.tests_passed / self.tests_run) * 100 if self.tests_run > 0 else 0
        
        print(f"\n📈 OVERALL RESULTS:")
        print(f"   Tests Run: {self.tests_run}")
        print(f"   Tests Passed: {self.tests_passed}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print(f"🎉 EXCELLENT: Stuck review cleanup functionality is working well!")
        elif success_rate >= 60:
            print(f"⚠️  GOOD: Most functionality working, some issues to address")
        else:
            print(f"❌ NEEDS ATTENTION: Significant issues with stuck review cleanup")
        
        return success_rate

if __name__ == "__main__":
    tester = StuckReviewCleanupTester()
    success_rate = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success_rate >= 70 else 1)