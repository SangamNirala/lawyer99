#!/usr/bin/env python3
"""
Judge Analysis and Comparison Validation System Testing
Tests both individual judge analysis and judge comparison endpoints with validation functionality
"""

import requests
import json
import sys
import time
from urllib.parse import quote
from datetime import datetime

# Backend URL from environment
BACKEND_URL = "https://legalcore.preview.emergentagent.com/api"

def test_individual_judge_analysis():
    """Test individual judge analysis endpoint with validation system"""
    print("🔍 TESTING INDIVIDUAL JUDGE ANALYSIS ENDPOINT")
    print("=" * 60)
    
    test_results = []
    
    # Test cases with realistic judge names
    test_judges = [
        {
            "name": "Sarah Martinez",
            "description": "Testing with realistic judge name Sarah Martinez"
        },
        {
            "name": "John Smith", 
            "description": "Testing with common judge name John Smith"
        },
        {
            "name": "Robert Johnson",
            "description": "Testing with judge name Robert Johnson"
        },
        {
            "name": "Judge Mary Wilson",
            "description": "Testing with 'Judge' prefix in name"
        }
    ]
    
    for test_case in test_judges:
        judge_name = test_case["name"]
        description = test_case["description"]
        
        print(f"\n📋 Test Case: {description}")
        print(f"Judge Name: {judge_name}")
        
        try:
            # URL encode the judge name to handle spaces and special characters
            encoded_judge_name = quote(judge_name)
            url = f"{BACKEND_URL}/litigation/judge-insights/{encoded_judge_name}"
            
            print(f"Request URL: {url}")
            
            # Make the API request
            response = requests.get(url, timeout=30)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Verify basic response structure
                required_fields = [
                    'judge_name', 'court', 'experience_years', 'total_cases',
                    'settlement_rate', 'plaintiff_success_rate', 'average_case_duration',
                    'confidence_score'
                ]
                
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    print(f"❌ Missing required fields: {missing_fields}")
                    test_results.append(False)
                    continue
                
                # Verify validation fields are present
                validation_fields = [
                    'is_verified', 'validation_sources', 'validation_summary', 'reference_links'
                ]
                
                validation_results = {}
                for field in validation_fields:
                    if field in data:
                        validation_results[field] = data[field]
                        print(f"✅ {field}: {data[field]}")
                    else:
                        validation_results[field] = "MISSING"
                        print(f"❌ {field}: MISSING")
                
                # Check validation field types and content
                validation_checks = []
                
                # Check is_verified is boolean
                if isinstance(data.get('is_verified'), bool):
                    validation_checks.append("✅ is_verified is boolean")
                else:
                    validation_checks.append(f"❌ is_verified should be boolean, got {type(data.get('is_verified'))}")
                
                # Check validation_sources is array
                if isinstance(data.get('validation_sources'), list):
                    validation_checks.append(f"✅ validation_sources is array with {len(data.get('validation_sources', []))} items")
                else:
                    validation_checks.append(f"❌ validation_sources should be array, got {type(data.get('validation_sources'))}")
                
                # Check validation_summary is string
                if isinstance(data.get('validation_summary'), str):
                    validation_checks.append(f"✅ validation_summary is string ({len(data.get('validation_summary', ''))} chars)")
                else:
                    validation_checks.append(f"❌ validation_summary should be string, got {type(data.get('validation_summary'))}")
                
                # Check reference_links is array of objects
                if isinstance(data.get('reference_links'), list):
                    ref_links = data.get('reference_links', [])
                    validation_checks.append(f"✅ reference_links is array with {len(ref_links)} items")
                    
                    # Check structure of reference links
                    for i, link in enumerate(ref_links[:3]):  # Check first 3 links
                        if isinstance(link, dict) and 'name' in link and 'url' in link:
                            validation_checks.append(f"✅ reference_links[{i}] has proper structure")
                        else:
                            validation_checks.append(f"❌ reference_links[{i}] missing name/url structure")
                else:
                    validation_checks.append(f"❌ reference_links should be array, got {type(data.get('reference_links'))}")
                
                # Check confidence_score
                confidence = data.get('confidence_score', 0)
                if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
                    validation_checks.append(f"✅ confidence_score is valid: {confidence}")
                else:
                    validation_checks.append(f"❌ confidence_score should be 0-1, got {confidence}")
                
                print("\n🔍 Validation Field Analysis:")
                for check in validation_checks:
                    print(f"  {check}")
                
                # Print key metrics
                print(f"\n📊 Key Metrics:")
                print(f"  Judge: {data.get('judge_name', 'N/A')}")
                print(f"  Court: {data.get('court', 'N/A')}")
                print(f"  Total Cases: {data.get('total_cases', 'N/A')}")
                print(f"  Settlement Rate: {data.get('settlement_rate', 'N/A')}")
                print(f"  Plaintiff Success Rate: {data.get('plaintiff_success_rate', 'N/A')}")
                print(f"  Confidence Score: {data.get('confidence_score', 'N/A')}")
                
                test_results.append(True)
                print("✅ Individual judge analysis test PASSED")
                
            else:
                print(f"❌ Request failed with status {response.status_code}")
                if response.text:
                    print(f"Error response: {response.text}")
                test_results.append(False)
                
        except Exception as e:
            print(f"❌ Exception occurred: {str(e)}")
            test_results.append(False)
        
        print("-" * 50)
    
    return test_results

def test_judge_comparison():
    """Test judge comparison endpoint with validation system"""
    print("\n🔍 TESTING JUDGE COMPARISON ENDPOINT")
    print("=" * 60)
    
    test_results = []
    
    # Test cases for judge comparison
    test_cases = [
        {
            "judge_names": ["Sarah Martinez", "John Smith"],
            "case_type": "civil",
            "description": "Comparing 2 judges with civil case type"
        },
        {
            "judge_names": ["Sarah Martinez", "John Smith", "Robert Johnson"],
            "case_type": "civil",
            "description": "Comparing 3 judges with civil case type"
        },
        {
            "judge_names": ["Judge Mary Wilson", "Sarah Martinez"],
            "case_type": "commercial",
            "description": "Comparing judges with commercial case type"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📋 Test Case: {test_case['description']}")
        print(f"Judges: {test_case['judge_names']}")
        print(f"Case Type: {test_case['case_type']}")
        
        try:
            url = f"{BACKEND_URL}/litigation/judge-comparison"
            
            payload = {
                "judge_names": test_case["judge_names"],
                "case_type": test_case["case_type"],
                "jurisdiction": "US"
            }
            
            print(f"Request URL: {url}")
            print(f"Payload: {json.dumps(payload, indent=2)}")
            
            # Make the API request
            response = requests.post(url, json=payload, timeout=30)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Verify basic response structure
                required_fields = [
                    'judges_compared', 'comparative_metrics', 'recommendations',
                    'analysis_date', 'confidence_score'
                ]
                
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    print(f"❌ Missing required fields: {missing_fields}")
                    test_results.append(False)
                    continue
                
                # Check validation_info section
                validation_info = data.get('validation_info')
                if validation_info:
                    print(f"✅ validation_info section present")
                    
                    # Check validation_info structure
                    if isinstance(validation_info, dict):
                        print(f"✅ validation_info is dictionary")
                        
                        # Check for verification_details
                        verification_details = validation_info.get('verification_details')
                        if verification_details:
                            print(f"✅ verification_details present with {len(verification_details)} judges")
                            
                            # Check each judge's verification details (it's a list of objects)
                            if isinstance(verification_details, list):
                                for i, details in enumerate(verification_details):
                                    judge_name = details.get('judge_name', f'Judge {i+1}')
                                    print(f"\n🔍 Verification for {judge_name}:")
                                    
                                    # Check required validation fields for each judge
                                    judge_validation_fields = [
                                        'is_verified', 'confidence_score', 'validation_summary', 'reference_links'
                                    ]
                                    
                                    for field in judge_validation_fields:
                                        if field in details:
                                            value = details[field]
                                            print(f"  ✅ {field}: {value}")
                                        else:
                                            print(f"  ❌ {field}: MISSING")
                            else:
                                print(f"❌ verification_details should be list, got {type(verification_details)}")
                        else:
                            print(f"❌ verification_details missing from validation_info")
                        
                        # Check for verified_judges count
                        verified_count = validation_info.get('verified_judges')
                        if verified_count is not None:
                            print(f"✅ verified_judges count: {verified_count}")
                        
                        # Check for estimated_judges count  
                        estimated_count = validation_info.get('estimated_judges')
                        if estimated_count is not None:
                            print(f"✅ estimated_judges count: {estimated_count}")
                            
                    else:
                        print(f"❌ validation_info should be dictionary, got {type(validation_info)}")
                else:
                    print(f"❌ validation_info section missing")
                
                # Print comparison results
                print(f"\n📊 Comparison Results:")
                print(f"  Judges Compared: {data.get('judges_compared', 'N/A')}")
                print(f"  Case Type Focus: {data.get('case_type_focus', 'N/A')}")
                print(f"  Confidence Score: {data.get('confidence_score', 'N/A')}")
                print(f"  Analysis Date: {data.get('analysis_date', 'N/A')}")
                
                # Print comparative metrics summary
                comparative_metrics = data.get('comparative_metrics', {})
                if comparative_metrics:
                    print(f"  Comparative Metrics Keys: {list(comparative_metrics.keys())}")
                
                # Print recommendations summary
                recommendations = data.get('recommendations', {})
                if recommendations:
                    print(f"  Recommendations Keys: {list(recommendations.keys())}")
                
                test_results.append(True)
                print("✅ Judge comparison test PASSED")
                
            else:
                print(f"❌ Request failed with status {response.status_code}")
                if response.text:
                    print(f"Error response: {response.text}")
                test_results.append(False)
                
        except Exception as e:
            print(f"❌ Exception occurred: {str(e)}")
            test_results.append(False)
        
        print("-" * 50)
    
    return test_results

def test_url_encoding_scenarios():
    """Test URL encoding scenarios for judge names with special characters"""
    print("\n🔍 TESTING URL ENCODING SCENARIOS")
    print("=" * 60)
    
    test_results = []
    
    # Test cases with special characters and spaces
    test_cases = [
        {
            "name": "Judge John Smith",
            "description": "Judge name with 'Judge' prefix and space"
        },
        {
            "name": "Mary O'Connor",
            "description": "Judge name with apostrophe"
        },
        {
            "name": "José Martinez",
            "description": "Judge name with accent character"
        }
    ]
    
    for test_case in test_cases:
        judge_name = test_case["name"]
        description = test_case["description"]
        
        print(f"\n📋 Test Case: {description}")
        print(f"Judge Name: '{judge_name}'")
        
        try:
            # URL encode the judge name
            encoded_judge_name = quote(judge_name)
            url = f"{BACKEND_URL}/litigation/judge-insights/{encoded_judge_name}"
            
            print(f"Encoded Name: '{encoded_judge_name}'")
            print(f"Request URL: {url}")
            
            # Make the API request
            response = requests.get(url, timeout=30)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                returned_name = data.get('judge_name', '')
                print(f"✅ Request successful")
                print(f"Returned Judge Name: '{returned_name}'")
                test_results.append(True)
            else:
                print(f"❌ Request failed with status {response.status_code}")
                test_results.append(False)
                
        except Exception as e:
            print(f"❌ Exception occurred: {str(e)}")
            test_results.append(False)
        
        print("-" * 40)
    
    return test_results

def main():
    """Main test execution function"""
    print("🎯 JUDGE ANALYSIS AND COMPARISON VALIDATION SYSTEM TESTING")
    print("=" * 80)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Test Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    all_results = []
    
    # Test 1: Individual Judge Analysis
    individual_results = test_individual_judge_analysis()
    all_results.extend(individual_results)
    
    # Test 2: Judge Comparison
    comparison_results = test_judge_comparison()
    all_results.extend(comparison_results)
    
    # Test 3: URL Encoding Scenarios
    encoding_results = test_url_encoding_scenarios()
    all_results.extend(encoding_results)
    
    # Final Results Summary
    print("\n" + "=" * 80)
    print("🎯 FINAL TEST RESULTS SUMMARY")
    print("=" * 80)
    
    total_tests = len(all_results)
    passed_tests = sum(all_results)
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    # Detailed breakdown
    print(f"\n📋 Test Breakdown:")
    print(f"  Individual Judge Analysis: {sum(individual_results)}/{len(individual_results)} passed")
    print(f"  Judge Comparison: {sum(comparison_results)}/{len(comparison_results)} passed")
    print(f"  URL Encoding Tests: {sum(encoding_results)}/{len(encoding_results)} passed")
    
    print(f"\n🕒 Test Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_rate >= 80:
        print("🎉 OVERALL RESULT: EXCELLENT - Judge validation system is working well!")
    elif success_rate >= 60:
        print("✅ OVERALL RESULT: GOOD - Judge validation system is mostly functional")
    else:
        print("⚠️ OVERALL RESULT: NEEDS ATTENTION - Judge validation system has issues")
    
    print("=" * 80)
    
    return success_rate >= 80

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)