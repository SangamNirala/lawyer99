#!/usr/bin/env python3
"""
Comprehensive Progress Analysis Test
"""

import requests
import json
import time
import uuid
from datetime import datetime

def comprehensive_progress_test():
    base_url = "https://cad694cf-d3ea-4711-9a72-668a6a40e52f.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    print("🔍 COMPREHENSIVE PROGRESS ANALYSIS")
    print("=" * 50)
    
    # Test 1: Check if there are existing stuck reviews
    print("\n🔍 TEST 1: Checking for existing stuck reviews")
    
    # Create attorney to check queues
    attorney_response = requests.post(f"{api_url}/attorney/create", json={
        "email": f"test_attorney_{int(time.time())}@example.com",
        "first_name": "Test",
        "last_name": "Attorney",
        "bar_number": f"BAR{int(time.time())}",
        "jurisdiction": "US",
        "role": "reviewing_attorney",
        "specializations": ["contract_law"],
        "years_experience": 5,
        "password": "TestPassword123!"
    }, timeout=10)
    
    if attorney_response.status_code == 200:
        attorney_data = attorney_response.json()
        attorney_id = attorney_data.get('attorney_id')
        print(f"✅ Test attorney created: {attorney_id}")
        
        # Check queue for existing reviews
        queue_response = requests.get(f"{api_url}/attorney/review/queue/{attorney_id}", timeout=10)
        if queue_response.status_code == 200:
            queue_data = queue_response.json()
            reviews = queue_data.get('reviews', [])
            print(f"📋 Found {len(reviews)} existing reviews in attorney queue")
            
            stuck_reviews = []
            for review in reviews:
                review_id = review.get('review_id')
                progress = review.get('progress_percentage', 0)
                status = review.get('status')
                created_at = review.get('created_at')
                
                print(f"   Review {review_id}: {progress}% ({status}) - Created: {created_at}")
                
                if progress == 0 and status == 'pending':
                    stuck_reviews.append(review_id)
                    print(f"   🚨 STUCK REVIEW FOUND: {review_id}")
            
            if stuck_reviews:
                print(f"\n🚨 CRITICAL FINDING: {len(stuck_reviews)} reviews stuck at 0%")
                
                # Test cleanup functionality
                print("\n🧹 Testing cleanup stuck reviews...")
                cleanup_response = requests.post(f"{api_url}/attorney/review/cleanup-stuck", timeout=10)
                if cleanup_response.status_code == 200:
                    cleanup_data = cleanup_response.json()
                    fixed_count = cleanup_data.get('fixed_count', 0)
                    print(f"✅ Cleanup completed: {fixed_count} reviews fixed")
                    
                    # Check the stuck reviews again
                    for stuck_review_id in stuck_reviews[:2]:  # Check first 2
                        print(f"\n🔍 Checking previously stuck review: {stuck_review_id}")
                        status_response = requests.get(f"{api_url}/attorney/review/status/{stuck_review_id}", timeout=10)
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            new_progress = status_data.get('progress_percentage', 0)
                            new_status = status_data.get('status')
                            attorney = status_data.get('assigned_attorney')
                            
                            print(f"   After cleanup - Progress: {new_progress}%, Status: {new_status}")
                            print(f"   Attorney assigned: {'Yes' if attorney else 'No'}")
                            
                            if new_progress > 0:
                                print("   ✅ Review is no longer stuck!")
                            else:
                                print("   🚨 Review still stuck at 0%")
            else:
                print("✅ No stuck reviews found in existing queue")
    
    # Test 2: Create new review and monitor from start
    print(f"\n🔍 TEST 2: Creating new review and monitoring from start")
    
    timestamp = int(time.time())
    client_id = f"client_{timestamp}_{str(uuid.uuid4())[:8]}"
    print(f"📝 Client ID: {client_id}")
    
    # Record consent
    consent_response = requests.post(f"{api_url}/client/consent", json={
        "client_id": client_id,
        "consent_text": "I consent to attorney supervision",
        "ip_address": "192.168.1.100"
    }, timeout=10)
    
    if consent_response.status_code != 200:
        print(f"❌ Consent failed: {consent_response.status_code}")
        return
    
    print("✅ Consent recorded")
    
    # Generate contract
    contract_data = {
        "contract_type": "NDA",
        "parties": {
            "party1_name": "Test Company Inc.",
            "party2_name": "John Doe"
        },
        "terms": {
            "purpose": "Business collaboration evaluation"
        },
        "jurisdiction": "US",
        "client_id": client_id
    }
    
    contract_response = requests.post(f"{api_url}/generate-contract-compliant", 
                                    json=contract_data, timeout=60)
    
    if contract_response.status_code != 200:
        print(f"❌ Contract generation failed: {contract_response.status_code}")
        return
    
    print("✅ Contract generated")
    
    # Extract review ID
    contract_data = contract_response.json()
    suggestions = contract_data.get('suggestions', [])
    review_id = None
    
    for suggestion in suggestions:
        if 'Document submitted for attorney review' in suggestion and 'ID:' in suggestion:
            import re
            uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
            match = re.search(uuid_pattern, suggestion)
            if match:
                review_id = match.group()
                break
    
    if not review_id:
        print("❌ Could not extract review ID")
        return
    
    print(f"✅ Review ID: {review_id}")
    
    # Monitor progress immediately and over time
    print(f"\n📊 Monitoring progress for new review {review_id}")
    
    progress_data = []
    
    # Check immediately (should be 0% pending)
    print("\n🔍 Immediate check (should be 0% pending)")
    status_response = requests.get(f"{api_url}/attorney/review/status/{review_id}", timeout=10)
    if status_response.status_code == 200:
        status_data = status_response.json()
        progress = status_data.get('progress_percentage', 0)
        status = status_data.get('status', 'unknown')
        attorney = status_data.get('assigned_attorney')
        
        progress_data.append({
            'time': 0,
            'progress': progress,
            'status': status,
            'has_attorney': attorney is not None
        })
        
        print(f"   Progress: {progress}%")
        print(f"   Status: {status}")
        print(f"   Attorney assigned: {'Yes' if attorney else 'No'}")
        
        if progress == 0 and status == 'pending':
            print("   ✅ Expected initial state: 0% pending")
        elif progress > 0:
            print("   🚨 UNEXPECTED: Progress > 0% immediately after creation")
        else:
            print(f"   ⚠️  Unexpected state: {progress}% {status}")
    
    # Wait and check again (should progress to in_review with >0%)
    print("\n⏳ Waiting 15 seconds for attorney assignment...")
    time.sleep(15)
    
    print("\n🔍 Check after 15 seconds (should be >0% in_review)")
    status_response = requests.get(f"{api_url}/attorney/review/status/{review_id}", timeout=10)
    if status_response.status_code == 200:
        status_data = status_response.json()
        progress = status_data.get('progress_percentage', 0)
        status = status_data.get('status', 'unknown')
        attorney = status_data.get('assigned_attorney')
        
        progress_data.append({
            'time': 15,
            'progress': progress,
            'status': status,
            'has_attorney': attorney is not None
        })
        
        print(f"   Progress: {progress}%")
        print(f"   Status: {status}")
        print(f"   Attorney assigned: {'Yes' if attorney else 'No'}")
        
        if attorney:
            print(f"   Attorney: {attorney.get('first_name')} {attorney.get('last_name')}")
        
        if progress > 0 and status == 'in_review':
            print("   ✅ Expected progression: >0% in_review")
        elif progress == 0 and status == 'pending':
            print("   🚨 ISSUE: Still stuck at 0% pending after 15 seconds")
        else:
            print(f"   ⚠️  Unexpected state: {progress}% {status}")
    
    # Continue monitoring for another 30 seconds
    for i in range(3):
        wait_time = 10
        print(f"\n⏳ Waiting {wait_time} seconds...")
        time.sleep(wait_time)
        
        total_time = 15 + (i + 1) * wait_time
        print(f"\n🔍 Check after {total_time} seconds")
        
        status_response = requests.get(f"{api_url}/attorney/review/status/{review_id}", timeout=10)
        if status_response.status_code == 200:
            status_data = status_response.json()
            progress = status_data.get('progress_percentage', 0)
            status = status_data.get('status', 'unknown')
            attorney = status_data.get('assigned_attorney')
            
            progress_data.append({
                'time': total_time,
                'progress': progress,
                'status': status,
                'has_attorney': attorney is not None
            })
            
            print(f"   Progress: {progress}%")
            print(f"   Status: {status}")
            print(f"   Attorney assigned: {'Yes' if attorney else 'No'}")
    
    # Analysis
    print(f"\n📊 PROGRESS ANALYSIS FOR REVIEW {review_id}")
    print("=" * 50)
    
    if len(progress_data) >= 2:
        initial = progress_data[0]
        final = progress_data[-1]
        
        print(f"Initial state: {initial['progress']}% ({initial['status']})")
        print(f"Final state: {final['progress']}% ({final['status']})")
        print(f"Progress change: {final['progress'] - initial['progress']:.2f}%")
        
        if initial['progress'] == 0 and final['progress'] == 0:
            print("🚨 CRITICAL ISSUE: Progress stuck at 0% throughout monitoring")
            print("   This confirms the user-reported issue!")
        elif initial['progress'] == 0 and final['progress'] > 0:
            print("✅ Progress advanced from 0% as expected")
        elif initial['progress'] > 0:
            print("⚠️  Started with progress > 0% (unusual but not necessarily wrong)")
        
        # Check attorney assignment pattern
        attorney_assignments = [entry['has_attorney'] for entry in progress_data]
        if not any(attorney_assignments):
            print("🚨 ISSUE: No attorney assigned throughout monitoring period")
        elif all(attorney_assignments):
            print("✅ Attorney consistently assigned")
        else:
            print("⚠️  Attorney assignment changed during monitoring")
        
        # Check status progression
        statuses = [entry['status'] for entry in progress_data]
        unique_statuses = list(set(statuses))
        if len(unique_statuses) == 1:
            if unique_statuses[0] == 'pending':
                print("🚨 ISSUE: Status stuck in 'pending' throughout monitoring")
            else:
                print(f"✅ Consistent status: {unique_statuses[0]}")
        else:
            print(f"✅ Status progression: {' → '.join(statuses)}")
    
    print(f"\n📋 Full progress timeline:")
    for entry in progress_data:
        print(f"   T+{entry['time']:2d}s: {entry['progress']:6.2f}% ({entry['status']}) - Attorney: {'Yes' if entry['has_attorney'] else 'No'}")

if __name__ == "__main__":
    comprehensive_progress_test()