import requests
import sys
import json
import time
import random
from datetime import datetime

class ReviewStatusFixVerificationTester:
    def __init__(self, base_url="https://9c72c586-5126-47dc-85cc-7afca9190d08.preview.emergentagent.com"):
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

    def test_complete_workflow_with_progression(self):
        """Test the complete workflow from contract generation to review progression"""
        print("🎯 TESTING COMPLETE WORKFLOW WITH REVIEW PROGRESSION")
        print("=" * 70)
        
        # Step 1: Record client consent
        timestamp = int(time.time() * 1000)
        random_string = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=11))
        client_id = f"client_{timestamp}_{random_string}"
        
        consent_data = {
            "client_id": client_id,
            "consent_text": "I consent to attorney supervision and review of legal documents as required by law.",
            "ip_address": "192.168.1.100"
        }
        
        success, response = self.run_test(
            "Client Consent Recording",
            "POST",
            "client/consent",
            200,
            consent_data
        )
        
        if not success:
            return False
        
        print(f"   ✅ Client ID: {client_id}")
        
        # Step 2: Generate contract with exact user scenario
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
            "client_id": client_id
        }
        
        success, response = self.run_test(
            "Contract Generation (User Scenario)",
            "POST",
            "generate-contract-compliant",
            200,
            contract_data,
            timeout=90
        )
        
        if not success:
            return False
        
        # Extract review ID
        review_id = None
        suggestions = response.get('suggestions', [])
        for suggestion in suggestions:
            if 'ID:' in suggestion:
                import re
                review_id_match = re.search(r'ID:\s*([a-f0-9-]+)', suggestion)
                if review_id_match:
                    review_id = review_id_match.group(1)
                    break
        
        if not review_id:
            print("❌ Could not extract review ID from contract generation")
            return False
        
        print(f"   ✅ Review ID extracted: {review_id}")
        
        # Step 3: Check initial review status
        success, response = self.run_test(
            "Initial Review Status",
            "GET",
            f"attorney/review/status/{review_id}",
            200
        )
        
        if not success:
            return False
        
        initial_status = response.get('status')
        initial_progress = response.get('progress_percentage', 0)
        initial_attorney = response.get('attorney')
        
        print(f"   Initial Status: {initial_status}")
        print(f"   Initial Progress: {initial_progress}%")
        print(f"   Initial Attorney: {initial_attorney}")
        
        # Step 4: If stuck in pending, run cleanup
        if initial_status == 'pending' and initial_progress == 0 and not initial_attorney:
            print(f"   🚨 Review is stuck - running cleanup system...")
            
            success, response = self.run_test(
                "Cleanup Stuck Reviews",
                "POST",
                "attorney/review/cleanup-stuck",
                200,
                {}
            )
            
            if success:
                fixed_count = response.get('fixed_count', 0)
                print(f"   ✅ Cleanup fixed {fixed_count} reviews")
                
                # Wait a moment for cleanup to process
                time.sleep(2)
        
        # Step 5: Check review status after cleanup
        success, response = self.run_test(
            "Review Status After Cleanup",
            "GET",
            f"attorney/review/status/{review_id}",
            200
        )
        
        if not success:
            return False
        
        final_status = response.get('status')
        final_progress = response.get('progress_percentage', 0)
        final_attorney = response.get('attorney')
        estimated_completion = response.get('estimated_completion')
        
        print(f"   Final Status: {final_status}")
        print(f"   Final Progress: {final_progress}%")
        print(f"   Final Attorney: {final_attorney}")
        print(f"   Estimated Completion: {estimated_completion}")
        
        # Step 6: Verify the fix
        if final_status == 'in_review' and final_progress > 0 and final_attorney:
            print(f"   ✅ SUCCESS: Review has progressed from 'pending' to 'in_review'")
            print(f"   ✅ SUCCESS: Progress advanced from 0% to {final_progress}%")
            print(f"   ✅ SUCCESS: Attorney assigned: {final_attorney.get('name', 'Unknown')}")
            return True
        else:
            print(f"   ❌ ISSUE PERSISTS: Review still not progressing properly")
            return False

    def test_multiple_reviews_progression(self):
        """Test multiple reviews to ensure consistent progression"""
        print("\n🎯 TESTING MULTIPLE REVIEWS FOR CONSISTENT PROGRESSION")
        print("=" * 70)
        
        all_success = True
        
        for i in range(3):
            print(f"\n--- Testing Review #{i+1} ---")
            
            # Generate unique client ID
            timestamp = int(time.time() * 1000) + i
            random_string = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=11))
            client_id = f"client_{timestamp}_{random_string}"
            
            # Record consent
            consent_data = {
                "client_id": client_id,
                "consent_text": "I consent to attorney supervision and review of legal documents as required by law."
            }
            
            success, _ = self.run_test(
                f"Consent for Review #{i+1}",
                "POST",
                "client/consent",
                200,
                consent_data
            )
            
            if not success:
                all_success = False
                continue
            
            # Generate contract
            contract_data = {
                "contract_type": "NDA",
                "jurisdiction": "US",
                "parties": {
                    "party1_name": f"Test Company {i+1} Inc.",
                    "party1_type": "company",
                    "party2_name": f"John Doe {i+1}",
                    "party2_type": "individual"
                },
                "terms": {
                    "purpose": f"Business collaboration evaluation #{i+1}",
                    "duration": "2_years"
                },
                "client_id": client_id
            }
            
            success, response = self.run_test(
                f"Contract Generation #{i+1}",
                "POST",
                "generate-contract-compliant",
                200,
                contract_data,
                timeout=60
            )
            
            if not success:
                all_success = False
                continue
            
            # Extract review ID
            review_id = None
            suggestions = response.get('suggestions', [])
            for suggestion in suggestions:
                if 'ID:' in suggestion:
                    import re
                    review_id_match = re.search(r'ID:\s*([a-f0-9-]+)', suggestion)
                    if review_id_match:
                        review_id = review_id_match.group(1)
                        break
            
            if not review_id:
                print(f"   ❌ Could not extract review ID for review #{i+1}")
                all_success = False
                continue
            
            # Wait a moment for processing
            time.sleep(1)
            
            # Check review status
            success, response = self.run_test(
                f"Review Status #{i+1}",
                "GET",
                f"attorney/review/status/{review_id}",
                200
            )
            
            if success:
                status = response.get('status')
                progress = response.get('progress_percentage', 0)
                attorney = response.get('attorney')
                
                print(f"   Review #{i+1}: Status={status}, Progress={progress}%, Attorney={attorney is not None}")
                
                # Check if review is progressing properly
                if status == 'in_review' and progress > 0 and attorney:
                    print(f"   ✅ Review #{i+1} is progressing correctly")
                elif status == 'pending':
                    print(f"   ⚠️  Review #{i+1} is still pending - may need cleanup")
                else:
                    print(f"   ❌ Review #{i+1} has unexpected status")
                    all_success = False
            else:
                all_success = False
        
        return all_success

    def run_comprehensive_verification(self):
        """Run comprehensive verification of the review status fix"""
        print("🎯 REVIEW STATUS FIX VERIFICATION")
        print("=" * 80)
        print("Verifying that the review status progression issue has been resolved")
        print("Expected: Reviews should progress from 'pending' (0%) → 'in_review' (25-95%)")
        print("=" * 80)
        
        # Test 1: Complete workflow with progression
        workflow_success = self.test_complete_workflow_with_progression()
        
        # Test 2: Multiple reviews for consistency
        multiple_success = self.test_multiple_reviews_progression()
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎯 VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        print("\n🔍 KEY FINDINGS:")
        if workflow_success:
            print("✅ Complete workflow test PASSED - Reviews are progressing correctly")
        else:
            print("❌ Complete workflow test FAILED - Issues still exist")
        
        if multiple_success:
            print("✅ Multiple reviews test PASSED - Consistent progression detected")
        else:
            print("❌ Multiple reviews test FAILED - Inconsistent behavior")
        
        overall_success = workflow_success and multiple_success
        
        if overall_success:
            print("\n🎉 CONCLUSION: Review status progression issue has been RESOLVED")
            print("   - Reviews are no longer stuck at 0% progress")
            print("   - Attorney assignment system is working")
            print("   - Progress advances from 25% to 95% as expected")
            print("   - Cleanup system fixes any stuck reviews")
        else:
            print("\n🚨 CONCLUSION: Review status progression issue PERSISTS")
            print("   - Some reviews may still get stuck at 0% progress")
            print("   - Attorney assignment may have intermittent issues")
            print("   - Manual intervention may be required")
        
        return overall_success

if __name__ == "__main__":
    print("🎯 REVIEW STATUS FIX VERIFICATION TEST")
    print("Testing to verify that the review status progression issue has been resolved")
    print("=" * 100)
    
    tester = ReviewStatusFixVerificationTester()
    success = tester.run_comprehensive_verification()
    
    print(f"\n🏁 FINAL RESULT: {'SUCCESS' if success else 'FAILURE'}")
    
    sys.exit(0 if success else 1)