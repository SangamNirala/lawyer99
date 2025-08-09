#!/usr/bin/env python3
"""
Final Progress Percentage Investigation Summary
"""

import requests
import json
import time
import uuid

def final_investigation():
    base_url = "https://7eb15b8a-98e2-46d7-83f4-aba058ad34e0.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    print("🔍 FINAL PROGRESS PERCENTAGE INVESTIGATION")
    print("=" * 60)
    print("User reported: Progress stays at 0% and doesn't increase")
    print("Testing to identify the exact root cause...")
    print()
    
    # Create a complete test scenario
    timestamp = int(time.time())
    client_id = f"client_{timestamp}_{str(uuid.uuid4())[:8]}"
    
    print(f"📝 Test Scenario: Complete document generation flow")
    print(f"   Client ID: {client_id}")
    
    # Step 1: Record consent
    print("\n🔍 STEP 1: Recording client consent")
    consent_response = requests.post(f"{api_url}/client/consent", json={
        "client_id": client_id,
        "consent_text": "I consent to attorney supervision and legal document review",
        "ip_address": "192.168.1.100"
    }, timeout=10)
    
    if consent_response.status_code == 200:
        print("✅ Consent recorded successfully")
    else:
        print(f"❌ Consent failed: {consent_response.status_code}")
        return
    
    # Step 2: Generate compliant contract
    print("\n🔍 STEP 2: Generating compliant contract")
    contract_data = {
        "contract_type": "NDA",
        "parties": {
            "party1_name": "Test Company Inc.",
            "party1_type": "company",
            "party2_name": "John Doe",
            "party2_type": "individual"
        },
        "terms": {
            "purpose": "Business collaboration evaluation",
            "duration": "2 years",
            "confidentiality_level": "high"
        },
        "jurisdiction": "US",
        "client_id": client_id
    }
    
    contract_response = requests.post(f"{api_url}/generate-contract-compliant", 
                                    json=contract_data, timeout=60)
    
    if contract_response.status_code == 200:
        print("✅ Contract generated successfully")
        contract_result = contract_response.json()
        
        # Extract review ID
        suggestions = contract_result.get('suggestions', [])
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
        else:
            print("❌ Could not extract review ID")
            return
    else:
        print(f"❌ Contract generation failed: {contract_response.status_code}")
        return
    
    # Step 3: Monitor progress over time
    print(f"\n🔍 STEP 3: Monitoring progress for review {review_id}")
    print("   This is the critical test to reproduce the user's issue...")
    
    progress_checks = []
    
    # Check immediately
    print("\n📊 Progress Check #1 (Immediate)")
    status_response = requests.get(f"{api_url}/attorney/review/status/{review_id}", timeout=10)
    if status_response.status_code == 200:
        status_data = status_response.json()
        progress = status_data.get('progress_percentage', 0)
        status = status_data.get('status', 'unknown')
        attorney = status_data.get('assigned_attorney')
        
        progress_checks.append({
            'check': 1,
            'time': 0,
            'progress': progress,
            'status': status,
            'has_attorney': attorney is not None,
            'attorney_name': f"{attorney.get('first_name', '')} {attorney.get('last_name', '')}" if attorney else None
        })
        
        print(f"   Progress: {progress}%")
        print(f"   Status: {status}")
        print(f"   Attorney: {'Yes' if attorney else 'No'}")
        
        if progress == 0:
            print("   🚨 CONFIRMED: Progress is at 0% - this matches user report!")
        else:
            print(f"   ⚠️  Progress is {progress}% - not 0% as user reported")
    
    # Check multiple times over 60 seconds
    for i in range(5):
        wait_time = 12  # 12 seconds between checks
        print(f"\n⏳ Waiting {wait_time} seconds...")
        time.sleep(wait_time)
        
        check_num = i + 2
        total_time = (i + 1) * wait_time
        
        print(f"\n📊 Progress Check #{check_num} (After {total_time}s)")
        status_response = requests.get(f"{api_url}/attorney/review/status/{review_id}", timeout=10)
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            progress = status_data.get('progress_percentage', 0)
            status = status_data.get('status', 'unknown')
            attorney = status_data.get('assigned_attorney')
            
            progress_checks.append({
                'check': check_num,
                'time': total_time,
                'progress': progress,
                'status': status,
                'has_attorney': attorney is not None,
                'attorney_name': f"{attorney.get('first_name', '')} {attorney.get('last_name', '')}" if attorney else None
            })
            
            print(f"   Progress: {progress}%")
            print(f"   Status: {status}")
            print(f"   Attorney: {'Yes' if attorney else 'No'}")
            
            if progress == 0:
                print("   🚨 STILL AT 0% - User issue confirmed!")
            elif progress > 0:
                print("   ✅ Progress is advancing")
    
    # Analysis
    print(f"\n📊 FINAL ANALYSIS - PROGRESS MONITORING RESULTS")
    print("=" * 60)
    
    if progress_checks:
        initial_progress = progress_checks[0]['progress']
        final_progress = progress_checks[-1]['progress']
        
        print(f"Initial progress: {initial_progress}%")
        print(f"Final progress: {final_progress}%")
        print(f"Progress change: {final_progress - initial_progress:.3f}%")
        
        # Check if stuck at 0%
        all_zero = all(check['progress'] == 0 for check in progress_checks)
        all_same = all(check['progress'] == initial_progress for check in progress_checks)
        
        if all_zero:
            print("\n🚨 CRITICAL FINDING: Progress stuck at 0% throughout entire monitoring period")
            print("   ✅ USER ISSUE CONFIRMED: This reproduces the exact problem reported")
            issue_confirmed = True
        elif all_same and initial_progress > 0:
            print(f"\n⚠️  FINDING: Progress stuck at {initial_progress}% (not advancing)")
            print("   ⚠️  Different from user report but still an issue")
            issue_confirmed = True
        elif final_progress > initial_progress:
            print("\n✅ FINDING: Progress is advancing normally")
            print("   ❌ CANNOT REPRODUCE user-reported issue")
            issue_confirmed = False
        else:
            print("\n⚠️  FINDING: Unexpected progress behavior")
            issue_confirmed = True
        
        # Check attorney assignment
        attorney_assignments = [check['has_attorney'] for check in progress_checks]
        if not any(attorney_assignments):
            print("\n🚨 CRITICAL ISSUE: No attorney assigned throughout monitoring")
            print("   This could be the root cause of progress issues")
        elif all(attorney_assignments):
            print("\n✅ Attorney consistently assigned")
        else:
            print("\n⚠️  Attorney assignment inconsistent")
        
        # Status analysis
        statuses = [check['status'] for check in progress_checks]
        unique_statuses = list(set(statuses))
        
        if len(unique_statuses) == 1:
            if unique_statuses[0] == 'pending':
                print(f"\n🚨 CRITICAL ISSUE: Status stuck in 'pending' throughout monitoring")
                print("   Reviews should progress from 'pending' to 'in_review'")
            else:
                print(f"\n✅ Consistent status: {unique_statuses[0]}")
        else:
            print(f"\n✅ Status progression detected: {' → '.join(statuses)}")
        
        # Detailed timeline
        print(f"\n📋 DETAILED PROGRESS TIMELINE:")
        print("   Time | Progress | Status    | Attorney")
        print("   -----|----------|-----------|----------")
        for check in progress_checks:
            attorney_display = "Yes" if check['has_attorney'] else "No"
            print(f"   {check['time']:3d}s | {check['progress']:7.2f}% | {check['status']:9s} | {attorney_display}")
        
        return {
            'issue_confirmed': issue_confirmed,
            'initial_progress': initial_progress,
            'final_progress': final_progress,
            'progress_advancing': final_progress > initial_progress,
            'attorney_assigned': any(attorney_assignments),
            'review_id': review_id,
            'client_id': client_id,
            'total_checks': len(progress_checks)
        }
    
    return None

if __name__ == "__main__":
    result = final_investigation()
    
    if result:
        print(f"\n🎯 SUMMARY FOR MAIN AGENT:")
        print("=" * 40)
        if result['issue_confirmed']:
            print("🚨 USER ISSUE CONFIRMED: Progress percentage problem reproduced")
        else:
            print("✅ USER ISSUE NOT REPRODUCED: Progress working normally")
        
        print(f"📊 Progress: {result['initial_progress']:.1f}% → {result['final_progress']:.1f}%")
        print(f"🔄 Advancing: {'Yes' if result['progress_advancing'] else 'No'}")
        print(f"👩‍⚖️ Attorney: {'Assigned' if result['attorney_assigned'] else 'Not Assigned'}")
        print(f"🔍 Review ID: {result['review_id']}")
        print(f"👤 Client ID: {result['client_id']}")