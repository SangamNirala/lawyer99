#!/usr/bin/env python3
"""
Partnership Endpoints Testing Script
Focus: Test Partnership Application and Partnership Search endpoints to identify "Invalid partner type error"

Based on code analysis:
- Partnership Application (POST /api/partnerships/apply) uses hardcoded mapping
- Partnership Search (GET /api/partnerships/search) uses {pt.value: pt for pt in PartnerType}
- PartnerType enum values: "technology_partner", "integration_partner", "reseller_partner", "legal_service_provider", etc.
"""

import requests
import sys
import json
from datetime import datetime

class PartnershipTester:
    def __init__(self, base_url="https://legalengine.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.errors_found = []

    def run_test(self, name, method, endpoint, expected_status, data=None, params=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        if params:
            print(f"   Params: {params}")
        if data:
            print(f"   Data: {json.dumps(data, indent=2)}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

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
                    if response.status_code == 400 and "Invalid partner type" in str(error_data):
                        self.errors_found.append({
                            "test": name,
                            "error": error_data,
                            "input": data or params
                        })
                except:
                    print(f"   Error: {response.text}")
                    if response.status_code == 400 and "Invalid partner type" in response.text:
                        self.errors_found.append({
                            "test": name,
                            "error": response.text,
                            "input": data or params
                        })
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timed out after {timeout} seconds")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_partnership_application_valid_types(self):
        """Test Partnership Application with valid partner types"""
        print("\n" + "="*80)
        print("TESTING PARTNERSHIP APPLICATION - VALID PARTNER TYPES")
        print("="*80)
        
        # Based on the hardcoded mapping in server.py
        valid_partner_types = [
            "technology_partner",
            "integration_partner", 
            "channel_partner",
            "reseller_partner",
            "legal_service_provider",
            "software_vendor",
            "consultant",
            "trainer"
        ]
        
        results = {}
        
        for partner_type in valid_partner_types:
            application_data = {
                "organization_name": f"Test {partner_type.replace('_', ' ').title()} Corp",
                "partner_type": partner_type,
                "contact_name": "John Doe",
                "contact_email": "john@example.com",
                "business_info": {
                    "description": f"Testing {partner_type} application",
                    "years_in_business": 5,
                    "employees": 50
                }
            }
            
            success, response = self.run_test(
                f"Partnership Application - {partner_type}",
                "POST",
                "partnerships/apply",
                200,
                application_data
            )
            
            results[partner_type] = {
                "success": success,
                "response": response
            }
        
        return results

    def test_partnership_application_invalid_types(self):
        """Test Partnership Application with invalid partner types that should cause 'Invalid partner type' error"""
        print("\n" + "="*80)
        print("TESTING PARTNERSHIP APPLICATION - INVALID PARTNER TYPES")
        print("="*80)
        
        # Test various invalid partner type values
        invalid_partner_types = [
            "Technology",  # Capitalized
            "TECHNOLOGY_PARTNER",  # All caps
            "Integration",  # Friendly name
            "INTEGRATION_PARTNER",  # All caps
            "Reseller",  # Friendly name
            "RESELLER_PARTNER",  # All caps
            "Legal Service Provider",  # Spaces
            "LEGAL_SERVICE_PROVIDER",  # All caps
            "tech_partner",  # Shortened
            "invalid_type",  # Completely invalid
            "partner",  # Generic
            "",  # Empty string
            "Technology Partner",  # Spaces
            "technology-partner",  # Hyphens instead of underscores
        ]
        
        results = {}
        
        for partner_type in invalid_partner_types:
            application_data = {
                "organization_name": f"Test Invalid Type Corp",
                "partner_type": partner_type,
                "contact_name": "Jane Doe",
                "contact_email": "jane@example.com",
                "business_info": {
                    "description": f"Testing invalid partner type: {partner_type}",
                    "years_in_business": 3,
                    "employees": 25
                }
            }
            
            success, response = self.run_test(
                f"Partnership Application - INVALID: '{partner_type}'",
                "POST",
                "partnerships/apply",
                400,  # Expecting 400 "Invalid partner type" error
                application_data
            )
            
            results[partner_type] = {
                "success": success,
                "response": response,
                "expected_error": True
            }
        
        return results

    def test_partnership_search_valid_types(self):
        """Test Partnership Search with valid partner types"""
        print("\n" + "="*80)
        print("TESTING PARTNERSHIP SEARCH - VALID PARTNER TYPES")
        print("="*80)
        
        # Based on PartnerType enum values
        valid_partner_types = [
            "technology_partner",
            "integration_partner",
            "channel_partner", 
            "reseller_partner",
            "legal_service_provider",
            "software_vendor",
            "consultant",
            "trainer"
        ]
        
        results = {}
        
        for partner_type in valid_partner_types:
            search_params = {
                "partner_type": partner_type,
                "geographic_region": "US",
                "min_rating": 4.0
            }
            
            success, response = self.run_test(
                f"Partnership Search - {partner_type}",
                "GET",
                "partnerships/search",
                200,
                params=search_params
            )
            
            results[partner_type] = {
                "success": success,
                "response": response
            }
        
        return results

    def test_partnership_search_invalid_types(self):
        """Test Partnership Search with invalid partner types that should cause 'Invalid partner type' error"""
        print("\n" + "="*80)
        print("TESTING PARTNERSHIP SEARCH - INVALID PARTNER TYPES")
        print("="*80)
        
        # Test various invalid partner type values
        invalid_partner_types = [
            "Technology",  # Capitalized
            "TECHNOLOGY_PARTNER",  # All caps
            "Integration",  # Friendly name
            "INTEGRATION_PARTNER",  # All caps
            "Reseller",  # Friendly name
            "RESELLER_PARTNER",  # All caps
            "Legal Service Provider",  # Spaces
            "LEGAL_SERVICE_PROVIDER",  # All caps
            "tech_partner",  # Shortened
            "invalid_type",  # Completely invalid
            "partner",  # Generic
            "",  # Empty string
            "Technology Partner",  # Spaces
            "technology-partner",  # Hyphens instead of underscores
        ]
        
        results = {}
        
        for partner_type in invalid_partner_types:
            search_params = {
                "partner_type": partner_type,
                "geographic_region": "US"
            }
            
            success, response = self.run_test(
                f"Partnership Search - INVALID: '{partner_type}'",
                "GET",
                "partnerships/search",
                400,  # Expecting 400 "Invalid partner type" error
                params=search_params
            )
            
            results[partner_type] = {
                "success": success,
                "response": response,
                "expected_error": True
            }
        
        return results

    def test_partnership_search_no_partner_type(self):
        """Test Partnership Search without partner_type parameter (should work)"""
        print("\n" + "="*80)
        print("TESTING PARTNERSHIP SEARCH - NO PARTNER TYPE FILTER")
        print("="*80)
        
        search_params = {
            "geographic_region": "US",
            "min_rating": 3.0
        }
        
        success, response = self.run_test(
            "Partnership Search - No partner_type filter",
            "GET",
            "partnerships/search",
            200,
            params=search_params
        )
        
        return {"no_filter": {"success": success, "response": response}}

    def analyze_mapping_differences(self):
        """Analyze the differences between the two mapping approaches"""
        print("\n" + "="*80)
        print("ANALYSIS: MAPPING DIFFERENCES BETWEEN ENDPOINTS")
        print("="*80)
        
        print("Partnership Application Mapping (Hardcoded):")
        application_mapping = [
            "technology_partner",
            "integration_partner", 
            "channel_partner",
            "reseller_partner",
            "legal_service_provider",
            "software_vendor",
            "consultant",
            "trainer"
        ]
        for mapping in application_mapping:
            print(f"  ✓ {mapping}")
        
        print("\nPartnership Search Mapping (Dynamic from PartnerType enum):")
        print("  Uses: {pt.value: pt for pt in PartnerType}")
        print("  This means it accepts the .value of each PartnerType enum member")
        
        print("\nPotential Issues:")
        print("  1. If PartnerType enum values don't match hardcoded mapping")
        print("  2. Case sensitivity differences")
        print("  3. Missing aliases in hardcoded mapping")
        print("  4. Different error handling approaches")

    def run_comprehensive_test(self):
        """Run all partnership endpoint tests"""
        print("🚀 Starting Comprehensive Partnership Endpoints Testing")
        print("Focus: Identifying 'Invalid partner type error' causes")
        print("="*80)
        
        # Test valid types
        app_valid_results = self.test_partnership_application_valid_types()
        search_valid_results = self.test_partnership_search_valid_types()
        
        # Test invalid types (these should produce the error we're looking for)
        app_invalid_results = self.test_partnership_application_invalid_types()
        search_invalid_results = self.test_partnership_search_invalid_types()
        
        # Test edge cases
        search_no_filter_results = self.test_partnership_search_no_partner_type()
        
        # Analyze mapping differences
        self.analyze_mapping_differences()
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total tests run: {self.tests_run}")
        print(f"Tests passed: {self.tests_passed}")
        print(f"Tests failed: {self.tests_run - self.tests_passed}")
        print(f"Success rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if self.errors_found:
            print(f"\n🎯 FOUND {len(self.errors_found)} 'Invalid partner type' ERRORS:")
            for i, error in enumerate(self.errors_found, 1):
                print(f"\n{i}. Test: {error['test']}")
                print(f"   Input: {error['input']}")
                print(f"   Error: {error['error']}")
        else:
            print("\n⚠️  No 'Invalid partner type' errors found in tests")
        
        return {
            "app_valid": app_valid_results,
            "app_invalid": app_invalid_results,
            "search_valid": search_valid_results,
            "search_invalid": search_invalid_results,
            "search_no_filter": search_no_filter_results,
            "errors_found": self.errors_found,
            "summary": {
                "total_tests": self.tests_run,
                "passed": self.tests_passed,
                "failed": self.tests_run - self.tests_passed,
                "success_rate": (self.tests_passed/self.tests_run)*100 if self.tests_run > 0 else 0
            }
        }

def main():
    """Main function to run partnership endpoint tests"""
    tester = PartnershipTester()
    results = tester.run_comprehensive_test()
    
    # Return appropriate exit code
    if results["summary"]["failed"] == 0:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ {results['summary']['failed']} tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()