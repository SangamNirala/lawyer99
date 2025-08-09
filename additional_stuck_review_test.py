import requests
import sys
import json
import time
from datetime import datetime

class AdditionalStuckReviewTester:
    def __init__(self, base_url="https://cad694cf-d3ea-4711-9a72-668a6a40e52f.preview.emergentagent.com"):
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

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_multiple_review_ids(self):
        """Test multiple review IDs that might exist in the system"""
        # These are potential review IDs that might exist based on the user's report
        potential_review_ids = [
            "b57f7ca3-24c1-4769-878b-afbbcf37814f",  # User's specific ID
            "cef9d675-7285-4c1c-8031-a5572bad5946",  # From testing agent's previous report
            "b5f7f23-24c1-4780-878b-afb6cf3814f",    # Variation mentioned in testing
        ]
        
        found_reviews = []
        
        for review_id in potential_review_ids:
            success, response = self.run_test(
                f"Check Review ID - {review_id}", 
                "GET", 
                f"attorney/review/status/{review_id}", 
                200
            )
            
            if success:
                found_reviews.append((review_id, response))
                print(f"   ✅ FOUND REVIEW: {review_id}")
                print(f"   Status: {response.get('status', 'Unknown')}")
                print(f"   Progress: {response.get('progress_percentage', 'Unknown')}%")
                print(f"   Assigned Attorney: {response.get('assigned_attorney', 'None')}")
                
                # Check if this is a stuck review
                progress = response.get('progress_percentage', 0)
                assigned_attorney = response.get('assigned_attorney')
                status = response.get('status', '').lower()
                
                if progress == 0 and not assigned_attorney and status == 'pending':
                    print(f"   🚨 STUCK REVIEW DETECTED!")
            else:
                print(f"   ❌ Review {review_id} not found (404)")
        
        return len(found_reviews) > 0, found_reviews

    def test_create_stuck_review_scenario(self):
        """Create a scenario that might result in a stuck review"""
        # First create an attorney
        attorney_data = {
            "email": f"stuck.test.attorney.{int(time.time())}@legalmate.test",
            "first_name": "Stuck",
            "last_name": "TestAttorney",
            "bar_number": f"STUCK{int(time.time())}",
            "jurisdiction": "US",
            "role": "reviewing_attorney",
            "specializations": ["contract_law"],
            "years_experience": 3,
            "password": "StuckTest123!"
        }
        
        success, response = self.run_test(
            "Create Attorney for Stuck Review Test", 
            "POST", 
            "attorney/create", 
            200, 
            attorney_data
        )
        
        if not success:
            return False, {}
        
        attorney_id = response.get('attorney_id')
        
        # Submit multiple documents to potentially create stuck reviews
        stuck_review_ids = []
        
        for i in range(3):
            document_data = {
                "document_content": f"Test document {i+1} for stuck review scenario testing. This document should be processed by the attorney supervision system.",
                "document_type": "contract",
                "client_id": f"stuck_test_client_{int(time.time())}_{i}",
                "original_request": {
                    "contract_type": "NDA",
                    "jurisdiction": "US",
                    "purpose": f"Stuck review test scenario {i+1}"
                },
                "priority": "normal"
            }
            
            success, response = self.run_test(
                f"Submit Document {i+1} for Stuck Review Test", 
                "POST", 
                "attorney/review/submit", 
                200, 
                document_data
            )
            
            if success and 'review_id' in response:
                review_id = response['review_id']
                stuck_review_ids.append(review_id)
                print(f"   Created Review ID: {review_id}")
        
        return len(stuck_review_ids) > 0, stuck_review_ids

    def test_cleanup_with_detailed_analysis(self):
        """Run cleanup and analyze results in detail"""
        print(f"\n🔧 RUNNING DETAILED CLEANUP ANALYSIS...")
        
        success, response = self.run_test(
            "Detailed Stuck Review Cleanup Analysis", 
            "POST", 
            "attorney/review/cleanup-stuck", 
            200
        )
        
        if success:
            print(f"\n📊 DETAILED CLEANUP RESULTS:")
            print(f"   Success: {response.get('success', 'Unknown')}")
            print(f"   Message: {response.get('message', 'No message')}")
            print(f"   Fixed Count: {response.get('fixed_count', 'Unknown')}")
            
            if 'details' in response:
                details = response['details']
                print(f"   Details: {details}")
                
                # Look for specific information about what was processed
                if isinstance(details, dict):
                    for key, value in details.items():
                        print(f"     {key}: {value}")
                elif isinstance(details, list):
                    for i, detail in enumerate(details):
                        print(f"     Detail {i+1}: {detail}")
            
            # Check if any reviews were actually fixed
            fixed_count = response.get('fixed_count', 0)
            if fixed_count > 0:
                print(f"   🎉 SUCCESS: {fixed_count} reviews were fixed by cleanup!")
            else:
                print(f"   ℹ️  No stuck reviews found to fix")
                print(f"   💡 This could mean:")
                print(f"      - The system is working correctly (no stuck reviews)")
                print(f"      - All reviews are being properly assigned")
                print(f"      - Previous cleanup runs have already fixed issues")
        
        return success, response

    def test_attorney_queue_status(self):
        """Check attorney queue to see if there are any pending reviews"""
        # We need an attorney ID - let's create one for testing
        attorney_data = {
            "email": f"queue.test.attorney.{int(time.time())}@legalmate.test",
            "first_name": "Queue",
            "last_name": "TestAttorney",
            "bar_number": f"QUEUE{int(time.time())}",
            "jurisdiction": "US",
            "role": "reviewing_attorney",
            "specializations": ["contract_law", "business_law"],
            "years_experience": 5,
            "password": "QueueTest123!"
        }
        
        success, response = self.run_test(
            "Create Attorney for Queue Test", 
            "POST", 
            "attorney/create", 
            200, 
            attorney_data
        )
        
        if not success:
            return False, {}
        
        attorney_id = response.get('attorney_id')
        
        # Check the attorney's queue
        success, response = self.run_test(
            f"Check Attorney Queue - {attorney_id}", 
            "GET", 
            f"attorney/review/queue/{attorney_id}", 
            200
        )
        
        if success:
            print(f"   📋 ATTORNEY QUEUE STATUS:")
            print(f"   Attorney ID: {attorney_id}")
            
            if 'reviews' in response:
                reviews = response['reviews']
                print(f"   Total Reviews in Queue: {len(reviews)}")
                
                if reviews:
                    for i, review in enumerate(reviews[:5]):  # Show first 5 reviews
                        print(f"     Review {i+1}:")
                        print(f"       ID: {review.get('review_id', 'Unknown')}")
                        print(f"       Status: {review.get('status', 'Unknown')}")
                        print(f"       Priority: {review.get('priority', 'Unknown')}")
                        print(f"       Created: {review.get('created_at', 'Unknown')}")
                else:
                    print(f"   ✅ No reviews in queue (attorney available)")
            else:
                print(f"   ⚠️  Queue response format unexpected")
        
        return success, response

    def run_comprehensive_test(self):
        """Run comprehensive stuck review testing"""
        print("🚀 Starting Comprehensive Stuck Review Testing...")
        print(f"🌐 Base URL: {self.base_url}")
        print("=" * 80)

        test_results = []
        
        # 1. Test multiple potential review IDs
        print("\n🎯 STEP 1: CHECK MULTIPLE POTENTIAL REVIEW IDs")
        result = self.test_multiple_review_ids()
        test_results.append(("Multiple Review ID Check", result[0]))
        
        # 2. Create stuck review scenario
        print("\n📄 STEP 2: CREATE STUCK REVIEW SCENARIO")
        result = self.test_create_stuck_review_scenario()
        test_results.append(("Create Stuck Review Scenario", result[0]))
        
        # 3. Check attorney queue status
        print("\n👨‍⚖️ STEP 3: CHECK ATTORNEY QUEUE STATUS")
        result = self.test_attorney_queue_status()
        test_results.append(("Attorney Queue Status", result[0]))
        
        # 4. Run detailed cleanup analysis
        print("\n🔧 STEP 4: DETAILED CLEANUP ANALYSIS")
        result = self.test_cleanup_with_detailed_analysis()
        test_results.append(("Detailed Cleanup Analysis", result[0]))

        # Print final results
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST RESULTS")
        print("=" * 80)
        
        for test_name, passed in test_results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status} - {test_name}")
        
        success_rate = (self.tests_passed / self.tests_run) * 100 if self.tests_run > 0 else 0
        
        print(f"\n📈 OVERALL RESULTS:")
        print(f"   Tests Run: {self.tests_run}")
        print(f"   Tests Passed: {self.tests_passed}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        return success_rate

if __name__ == "__main__":
    tester = AdditionalStuckReviewTester()
    success_rate = tester.run_comprehensive_test()
    
    sys.exit(0 if success_rate >= 70 else 1)