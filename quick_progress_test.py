#!/usr/bin/env python3
"""
Quick Progress Test - Focus on the core issue
"""

import requests
import json
import time
import uuid
from datetime import datetime

def test_progress_issue():
    base_url = "https://b2e17d7f-d715-45b1-acd0-e51dd70d439b.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    print("🔍 QUICK PROGRESS PERCENTAGE TEST")
    print("=" * 40)
    
    # Generate unique client ID
    timestamp = int(time.time())
    client_id = f"client_{timestamp}_{str(uuid.uuid4())[:8]}"
    print(f"📝 Client ID: {client_id}")
    
    # Step 1: Record consent
    print("\n📝 Step 1: Recording consent...")
    consent_response = requests.post(f"{api_url}/client/consent", json={
        "client_id": client_id,
        "consent_text": "I consent to attorney supervision",
        "ip_address": "192.168.1.100"
    }, timeout=10)
    
    if consent_response.status_code == 200:
        print("✅ Consent recorded successfully")
    else:
        print(f"❌ Consent failed: {consent_response.status_code}")
        return
    
    # Step 2: Generate contract with longer timeout
    print("\n📝 Step 2: Generating contract...")
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
    
    try:
        contract_response = requests.post(f"{api_url}/generate-contract-compliant", 
                                        json=contract_data, timeout=60)
        
        if contract_response.status_code == 200:
            print("✅ Contract generated successfully")
            contract_data = contract_response.json()
            
            # Extract review ID
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
            
            if review_id:
                print(f"✅ Review ID extracted: {review_id}")
                
                # Step 3: Monitor progress
                print(f"\n📊 Step 3: Monitoring progress for review {review_id}")
                
                for i in range(6):  # Check 6 times over 60 seconds
                    print(f"\n🔍 Progress check #{i+1}")
                    
                    status_response = requests.get(f"{api_url}/attorney/review/status/{review_id}", timeout=10)
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        progress = status_data.get('progress_percentage', 0)
                        status = status_data.get('status', 'unknown')
                        attorney = status_data.get('assigned_attorney')
                        
                        print(f"   Progress: {progress}%")
                        print(f"   Status: {status}")
                        print(f"   Attorney assigned: {'Yes' if attorney else 'No'}")
                        
                        if attorney:
                            print(f"   Attorney: {attorney.get('first_name')} {attorney.get('last_name')}")
                        
                        if progress > 0:
                            print("✅ Progress is advancing!")
                        elif i == 0:
                            print("⚠️  Starting at 0% - this is expected")
                        else:
                            print("🚨 Progress still stuck at 0%")
                    else:
                        print(f"❌ Failed to get status: {status_response.status_code}")
                    
                    if i < 5:  # Don't sleep after last check
                        print("⏳ Waiting 10 seconds...")
                        time.sleep(10)
                
            else:
                print("❌ Could not extract review ID from response")
                print(f"Suggestions: {suggestions}")
        else:
            print(f"❌ Contract generation failed: {contract_response.status_code}")
            print(f"Response: {contract_response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Contract generation timed out")
        
        # Let's check if there are any recent reviews we can monitor
        print("\n🔍 Checking for recent reviews to monitor...")
        
        # Create a test attorney to check queues
        attorney_response = requests.post(f"{api_url}/attorney/create", json={
            "email": f"test_{int(time.time())}@example.com",
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
            attorney_id = attorney_response.json().get('attorney_id')
            print(f"✅ Test attorney created: {attorney_id}")
            
            # Check queue
            queue_response = requests.get(f"{api_url}/attorney/review/queue/{attorney_id}", timeout=10)
            if queue_response.status_code == 200:
                queue_data = queue_response.json()
                reviews = queue_data.get('reviews', [])
                print(f"📋 Found {len(reviews)} reviews in queue")
                
                for review in reviews:
                    review_id = review.get('review_id')
                    progress = review.get('progress_percentage', 0)
                    status = review.get('status')
                    print(f"   Review {review_id}: {progress}% ({status})")
                    
                    if progress == 0:
                        print(f"🚨 Found stuck review: {review_id}")

if __name__ == "__main__":
    test_progress_issue()