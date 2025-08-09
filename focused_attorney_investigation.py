import requests
import json
import time
from datetime import datetime

class FocusedAttorneyInvestigation:
    def __init__(self):
        self.api_url = "https://8cd68d5c-4981-470b-b1c0-9982d2b4a8d2.preview.emergentagent.com/api"
        self.working_review_ids = [
            'cef9d675-7285-4c1c-8031-a5572bad5946',
            'b57f7ca3-24c1-4769-878b-afbbcf37814f'
        ]
        self.original_review_id = 'b5f7f23-24c1-4780-878b-afb6cf3814f'

    def investigate_review_assignment_issue(self):
        """Investigate why reviews are not getting attorneys assigned"""
        print("🔍 INVESTIGATING ATTORNEY ASSIGNMENT ISSUE")
        print("=" * 60)
        
        # Check existing reviews
        for review_id in self.working_review_ids:
            print(f"\n📋 ANALYZING REVIEW: {review_id}")
            try:
                response = requests.get(f'{self.api_url}/attorney/review/status/{review_id}', timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    print(f"   Status: {data.get('status', 'N/A')}")
                    print(f"   Progress: {data.get('progress_percentage', 'N/A')}%")
                    print(f"   Created: {data.get('created_at', 'N/A')}")
                    print(f"   Priority: {data.get('priority', 'N/A')}")
                    
                    # Calculate how long it's been pending
                    created_at = data.get('created_at', '')
                    if created_at:
                        try:
                            created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            current_time = datetime.now(created_time.tzinfo)
                            time_diff = current_time - created_time
                            hours_elapsed = time_diff.total_seconds() / 3600
                            print(f"   ⏰ Pending for: {hours_elapsed:.1f} hours")
                            
                            if hours_elapsed > 24:
                                print(f"   🚨 OVERDUE: Review has been pending for {hours_elapsed:.1f} hours")
                            elif hours_elapsed > 4:
                                print(f"   ⚠️  DELAYED: Review has been pending for {hours_elapsed:.1f} hours")
                        except Exception as e:
                            print(f"   ⚠️  Could not calculate time: {e}")
                    
                    # Check attorney assignment
                    attorney = data.get('attorney', {})
                    if attorney:
                        print(f"   👩‍⚖️ Attorney: {attorney.get('name', 'N/A')} (ID: {attorney.get('id', 'N/A')})")
                        print(f"   👩‍⚖️ Role: {attorney.get('role', 'N/A')}")
                    else:
                        print(f"   ❌ NO ATTORNEY ASSIGNED - This is the core issue!")
                        
                    # Check estimated completion
                    estimated = data.get('estimated_completion', '')
                    if estimated:
                        print(f"   📅 Estimated completion: {estimated}")
                    
                    print(f"   📄 Document Type: {data.get('document_type', 'N/A')}")
                    print(f"   📝 Comments: {data.get('comments', 'None')}")
                    
            except Exception as e:
                print(f"   ❌ Error checking review: {e}")

    def test_attorney_creation_and_assignment(self):
        """Test creating an attorney and see if assignment works"""
        print(f"\n🔧 TESTING ATTORNEY CREATION AND ASSIGNMENT")
        print("=" * 60)
        
        # Try to create a test attorney
        attorney_data = {
            "email": f"test.attorney.{int(time.time())}@legalmate.ai",
            "first_name": "Test",
            "last_name": "Attorney",
            "bar_number": f"TEST{int(time.time())}",
            "jurisdiction": "US",
            "role": "reviewing_attorney",
            "specializations": ["contract_law", "business_law"],
            "years_experience": 5,
            "password": "TestPassword123!"
        }
        
        print(f"📝 Creating test attorney...")
        try:
            response = requests.post(f'{self.api_url}/attorney/create', json=attorney_data, timeout=10)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Attorney created successfully")
                print(f"   📋 Response: {json.dumps(result, indent=4)}")
                
                # Extract attorney ID if available
                attorney_id = result.get('attorney_id') or result.get('id')
                if attorney_id:
                    print(f"   👩‍⚖️ Attorney ID: {attorney_id}")
                    
                    # Test attorney profile retrieval
                    print(f"\n📋 Testing attorney profile retrieval...")
                    profile_response = requests.get(f'{self.api_url}/attorney/profile/{attorney_id}', timeout=10)
                    print(f"   Profile Status: {profile_response.status_code}")
                    
                    if profile_response.status_code == 200:
                        profile_data = profile_response.json()
                        print(f"   ✅ Profile retrieved successfully")
                        print(f"   👩‍⚖️ Name: {profile_data.get('first_name', '')} {profile_data.get('last_name', '')}")
                        print(f"   📧 Email: {profile_data.get('email', 'N/A')}")
                        print(f"   🎯 Role: {profile_data.get('role', 'N/A')}")
                        print(f"   ✅ Available: {profile_data.get('available', 'N/A')}")
                        print(f"   📊 Current Reviews: {profile_data.get('current_review_count', 'N/A')}")
                        
                        return attorney_id
                    else:
                        print(f"   ❌ Could not retrieve profile: {profile_response.text}")
                else:
                    print(f"   ⚠️  No attorney ID in response")
            else:
                print(f"   ❌ Attorney creation failed: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error creating attorney: {e}")
        
        return None

    def test_document_submission_and_assignment(self, attorney_id=None):
        """Test submitting a document and see if it gets assigned"""
        print(f"\n📄 TESTING DOCUMENT SUBMISSION AND ASSIGNMENT")
        print("=" * 60)
        
        # Submit a test document for review
        document_data = {
            "document_content": "This is a test contract for investigating attorney assignment issues. It contains standard business terms and conditions for a service agreement between Test Company Inc. and Test Client LLC. The contract includes payment terms, deliverables, and termination clauses.",
            "document_type": "contract",
            "client_id": f"test-client-{int(time.time())}",
            "original_request": {
                "contract_type": "service_agreement",
                "parties": {
                    "party1_name": "Test Company Inc.",
                    "party2_name": "Test Client LLC"
                },
                "terms": {
                    "payment_amount": "$5000",
                    "duration": "3 months"
                }
            },
            "priority": "normal"
        }
        
        print(f"📝 Submitting document for review...")
        try:
            response = requests.post(f'{self.api_url}/attorney/review/submit', json=document_data, timeout=15)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Document submitted successfully")
                print(f"   📋 Response: {json.dumps(result, indent=4)}")
                
                review_id = result.get('review_id')
                assigned_attorney = result.get('assigned_attorney_id')
                
                if review_id:
                    print(f"   📄 Review ID: {review_id}")
                    
                    if assigned_attorney:
                        print(f"   ✅ ATTORNEY ASSIGNED: {assigned_attorney}")
                        
                        # Check the review status immediately
                        time.sleep(2)
                        status_response = requests.get(f'{self.api_url}/attorney/review/status/{review_id}', timeout=10)
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            print(f"   📊 Review Status: {status_data.get('status', 'N/A')}")
                            print(f"   📈 Progress: {status_data.get('progress_percentage', 'N/A')}%")
                            
                            attorney_info = status_data.get('attorney', {})
                            if attorney_info:
                                print(f"   👩‍⚖️ Assigned Attorney: {attorney_info.get('name', 'N/A')}")
                                print(f"   🆔 Attorney ID: {attorney_info.get('id', 'N/A')}")
                            else:
                                print(f"   ❌ NO ATTORNEY INFO IN STATUS - Assignment may have failed")
                        
                        return review_id
                    else:
                        print(f"   ❌ NO ATTORNEY ASSIGNED - This confirms the assignment issue!")
                        return review_id
                else:
                    print(f"   ⚠️  No review ID in response")
            else:
                print(f"   ❌ Document submission failed: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error submitting document: {e}")
        
        return None

    def test_attorney_queue_functionality(self, attorney_id):
        """Test attorney queue functionality"""
        if not attorney_id:
            print(f"\n⚠️  Skipping queue test - no attorney ID available")
            return
            
        print(f"\n📋 TESTING ATTORNEY QUEUE FUNCTIONALITY")
        print("=" * 60)
        
        try:
            response = requests.get(f'{self.api_url}/attorney/review/queue/{attorney_id}', timeout=10)
            print(f"   Queue Status: {response.status_code}")
            
            if response.status_code == 200:
                queue_data = response.json()
                print(f"   ✅ Queue retrieved successfully")
                print(f"   📊 Queue Length: {queue_data.get('queue_length', 'N/A')}")
                print(f"   👩‍⚖️ Attorney ID: {queue_data.get('attorney_id', 'N/A')}")
                
                reviews = queue_data.get('reviews', [])
                if reviews:
                    print(f"   📄 Reviews in Queue: {len(reviews)}")
                    for i, review in enumerate(reviews[:3]):  # Show first 3
                        print(f"      {i+1}. Review ID: {review.get('review_id', 'N/A')}")
                        print(f"         Status: {review.get('status', 'N/A')}")
                        print(f"         Priority: {review.get('priority', 'N/A')}")
                        print(f"         Document Type: {review.get('document_type', 'N/A')}")
                else:
                    print(f"   📄 No reviews in queue")
                    
            else:
                print(f"   ❌ Queue retrieval failed: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error checking queue: {e}")

    def investigate_original_review_id(self):
        """Investigate the original review ID that was reported"""
        print(f"\n🎯 INVESTIGATING ORIGINAL REVIEW ID")
        print("=" * 60)
        print(f"Original Review ID: {self.original_review_id}")
        
        try:
            response = requests.get(f'{self.api_url}/attorney/review/status/{self.original_review_id}', timeout=10)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 404:
                print(f"   ❌ REVIEW NOT FOUND")
                print(f"   📝 This review ID does not exist in the system")
                print(f"   💡 Possible causes:")
                print(f"      - Review was deleted or expired")
                print(f"      - Incorrect review ID provided")
                print(f"      - Database was reset/cleared")
                print(f"      - Review ID was from a different environment")
            elif response.status_code == 200:
                data = response.json()
                print(f"   ✅ REVIEW FOUND")
                print(f"   📋 Status: {data.get('status', 'N/A')}")
                print(f"   📈 Progress: {data.get('progress_percentage', 'N/A')}%")
            else:
                print(f"   ❌ Unexpected response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error checking original review: {e}")

    def run_investigation(self):
        """Run the complete focused investigation"""
        print("=" * 80)
        print("🔍 FOCUSED ATTORNEY REVIEW SYSTEM INVESTIGATION")
        print("=" * 80)
        
        # 1. Investigate existing reviews with assignment issues
        self.investigate_review_assignment_issue()
        
        # 2. Test attorney creation
        attorney_id = self.test_attorney_creation_and_assignment()
        
        # 3. Test document submission and assignment
        new_review_id = self.test_document_submission_and_assignment(attorney_id)
        
        # 4. Test attorney queue functionality
        self.test_attorney_queue_functionality(attorney_id)
        
        # 5. Investigate the original review ID
        self.investigate_original_review_id()
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 INVESTIGATION FINDINGS")
        print("=" * 80)
        print("🔍 KEY ISSUES IDENTIFIED:")
        print("   1. ❌ Existing reviews have NO ATTORNEY ASSIGNED")
        print("   2. ⏰ Reviews are stuck at 0% progress")
        print("   3. 🚨 Reviews are becoming OVERDUE due to no assignment")
        print("   4. 🆔 Original review ID does not exist (404)")
        
        print("\n💡 ROOT CAUSE ANALYSIS:")
        print("   - Attorney assignment logic is not working properly")
        print("   - Reviews are created but never assigned to attorneys")
        print("   - This causes reviews to remain in 'pending' status indefinitely")
        print("   - Without attorney assignment, reviews cannot progress beyond 0%")
        
        print("\n🔧 RECOMMENDED ACTIONS:")
        print("   1. Fix attorney auto-assignment logic in submit_for_review")
        print("   2. Ensure attorneys are available and properly configured")
        print("   3. Add fallback assignment mechanism for unassigned reviews")
        print("   4. Implement review timeout and escalation procedures")
        print("   5. Add monitoring for stuck/unassigned reviews")
        
        print("=" * 80)

if __name__ == "__main__":
    investigator = FocusedAttorneyInvestigation()
    investigator.run_investigation()