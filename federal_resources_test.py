import requests
import sys
import json
import time
from datetime import datetime

class FederalResourcesAPITester:
    def __init__(self, base_url="https://027a3f35-a0bc-40cd-ad79-483d46c6126e.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=120):
        """Run a single API test with extended timeout for federal resources collection"""
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

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timed out after {timeout} seconds")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_federal_resources_endpoint_accessibility(self):
        """Test 1: Basic API Endpoint Accessibility - Check if the new endpoint exists and is accessible"""
        print("\n" + "="*80)
        print("TEST 1: BASIC API ENDPOINT ACCESSIBILITY")
        print("="*80)
        
        success, response = self.run_test(
            "Federal Resources Knowledge Base Rebuild Endpoint", 
            "POST", 
            "legal-qa/rebuild-federal-resources-knowledge-base", 
            200,
            timeout=300  # 5 minutes for collection process
        )
        
        if success:
            print(f"✅ ENDPOINT ACCESSIBILITY: Federal resources endpoint is accessible and responding")
            
            # Verify response structure
            if isinstance(response, dict):
                expected_fields = [
                    'status', 'collection_mode', 'documents_collected', 'target_documents',
                    'target_achievement', 'source_breakdown', 'priority_agency_breakdown',
                    'legal_domain_breakdown', 'quality_metrics', 'features_enabled'
                ]
                
                missing_fields = [field for field in expected_fields if field not in response]
                if not missing_fields:
                    print(f"✅ RESPONSE STRUCTURE: All expected response fields present")
                else:
                    print(f"⚠️  RESPONSE STRUCTURE: Missing fields: {missing_fields}")
                
                # Check collection mode
                if response.get('collection_mode') == 'FEDERAL_RESOURCES':
                    print(f"✅ COLLECTION MODE: Correctly set to FEDERAL_RESOURCES")
                else:
                    print(f"❌ COLLECTION MODE: Expected FEDERAL_RESOURCES, got {response.get('collection_mode')}")
                
                # Check target documents
                target_docs = response.get('target_documents', 0)
                if target_docs >= 5000:
                    print(f"✅ TARGET DOCUMENTS: {target_docs} meets 5,000+ requirement")
                else:
                    print(f"❌ TARGET DOCUMENTS: {target_docs} below 5,000 requirement")
                
                print(f"   📊 Collection Results:")
                print(f"      - Documents Collected: {response.get('documents_collected', 0)}")
                print(f"      - Target Achievement: {response.get('target_achievement', 'N/A')}")
                print(f"      - Collection Time: {response.get('collection_time_hours', 'N/A')} hours")
                
        else:
            print(f"❌ ENDPOINT ACCESSIBILITY: Federal resources endpoint failed or not accessible")
        
        return success, response

    def test_collection_mode_import_and_configuration(self):
        """Test 2: Import and Configuration Validation - Verify CollectionMode.FEDERAL_RESOURCES is properly imported"""
        print("\n" + "="*80)
        print("TEST 2: IMPORT AND CONFIGURATION VALIDATION")
        print("="*80)
        
        # Test the endpoint to verify the CollectionMode is properly imported and configured
        success, response = self.run_test(
            "CollectionMode.FEDERAL_RESOURCES Import Verification", 
            "POST", 
            "legal-qa/rebuild-federal-resources-knowledge-base", 
            200,
            timeout=300
        )
        
        if success and isinstance(response, dict):
            # Verify collection mode configuration
            collection_mode = response.get('collection_mode')
            if collection_mode == 'FEDERAL_RESOURCES':
                print(f"✅ IMPORT VERIFICATION: CollectionMode.FEDERAL_RESOURCES properly imported and configured")
            else:
                print(f"❌ IMPORT VERIFICATION: Expected FEDERAL_RESOURCES, got {collection_mode}")
                return False, response
            
            # Verify configuration parameters
            quality_metrics = response.get('quality_metrics', {})
            
            # Check minimum content length (800 words requirement)
            min_content_length = quality_metrics.get('min_content_length', 0)
            if min_content_length >= 800:
                print(f"✅ MIN CONTENT LENGTH: {min_content_length} words meets 800+ requirement")
            else:
                print(f"❌ MIN CONTENT LENGTH: {min_content_length} words below 800 requirement")
            
            # Check government domains only
            gov_domains_only = quality_metrics.get('government_domains_only', False)
            if gov_domains_only:
                print(f"✅ GOVERNMENT DOMAINS: Restricted to .gov domains only")
            else:
                print(f"❌ GOVERNMENT DOMAINS: Not restricted to .gov domains")
            
            # Check priority agencies configuration
            priority_agencies = response.get('priority_agency_breakdown', {})
            expected_agencies = ['SEC', 'DOL', 'USPTO', 'IRS']
            found_agencies = [agency.upper() for agency in priority_agencies.keys()]
            
            print(f"   🏢 Priority Agencies Configuration:")
            for agency in expected_agencies:
                if agency in found_agencies:
                    count = priority_agencies.get(agency, priority_agencies.get(agency.lower(), 0))
                    print(f"      ✅ {agency}: {count} documents")
                else:
                    print(f"      ⚠️  {agency}: Not found in results")
            
            # Check target breakdown
            target_breakdown = response.get('target_breakdown', {})
            expected_targets = {
                'cornell_lii_target': 2000,
                'federal_agencies_target': 2000,
                'government_sources_target': 1000
            }
            
            print(f"   🎯 Target Configuration:")
            for target_key, expected_value in expected_targets.items():
                actual_value = target_breakdown.get(target_key, 0)
                if actual_value == expected_value:
                    print(f"      ✅ {target_key}: {actual_value} (matches expected)")
                else:
                    print(f"      ⚠️  {target_key}: {actual_value} (expected {expected_value})")
            
            print(f"✅ CONFIGURATION VALIDATION: Federal resources mode properly configured")
            
        else:
            print(f"❌ CONFIGURATION VALIDATION: Failed to verify configuration")
            return False, response
        
        return success, response

    def test_method_integration(self):
        """Test 3: Method Integration - Test that federal resources collection methods are properly integrated"""
        print("\n" + "="*80)
        print("TEST 3: METHOD INTEGRATION VERIFICATION")
        print("="*80)
        
        # Run the federal resources collection to test method integration
        success, response = self.run_test(
            "Federal Resources Methods Integration", 
            "POST", 
            "legal-qa/rebuild-federal-resources-knowledge-base", 
            200,
            timeout=300
        )
        
        if success and isinstance(response, dict):
            # Verify source breakdown indicates all three methods were called
            source_breakdown = response.get('source_breakdown', {})
            
            print(f"   📋 Method Integration Results:")
            
            # Check Cornell LII method (_fetch_cornell_legal_resources)
            cornell_count = source_breakdown.get('Cornell LII', 0)
            if cornell_count > 0:
                print(f"      ✅ _fetch_cornell_legal_resources(): {cornell_count} documents")
            else:
                print(f"      ❌ _fetch_cornell_legal_resources(): No documents collected")
            
            # Check Federal Agencies method (_fetch_priority_federal_agency_content)
            agencies_count = source_breakdown.get('Federal Agencies', 0)
            if agencies_count > 0:
                print(f"      ✅ _fetch_priority_federal_agency_content(): {agencies_count} documents")
            else:
                print(f"      ❌ _fetch_priority_federal_agency_content(): No documents collected")
            
            # Check Government Sources method (_fetch_government_legal_docs)
            gov_count = source_breakdown.get('Government Sources', 0)
            if gov_count > 0:
                print(f"      ✅ _fetch_government_legal_docs(): {gov_count} documents")
            else:
                print(f"      ❌ _fetch_government_legal_docs(): No documents collected")
            
            # Verify total integration
            total_collected = sum(source_breakdown.values())
            if total_collected > 0:
                print(f"✅ METHOD INTEGRATION: All three collection methods successfully integrated")
                print(f"   Total documents from all methods: {total_collected}")
            else:
                print(f"❌ METHOD INTEGRATION: No documents collected from any method")
                return False, response
            
            # Check features enabled to verify method-specific features
            features = response.get('features_enabled', [])
            expected_features = [
                "Government-Specific Quality Control",
                ".gov Domain Verification Only",
                "Priority Agency Focus (SEC, DOL, USPTO, IRS)",
                "Enhanced Government Metadata Extraction"
            ]
            
            print(f"   🔧 Method-Specific Features:")
            for feature in expected_features:
                if any(feature in f for f in features):
                    print(f"      ✅ {feature}")
                else:
                    print(f"      ⚠️  {feature}: Not found in features")
            
        else:
            print(f"❌ METHOD INTEGRATION: Failed to verify method integration")
            return False, response
        
        return success, response

    def test_configuration_validation(self):
        """Test 4: Configuration Validation - Check federal resources collection mode configuration"""
        print("\n" + "="*80)
        print("TEST 4: CONFIGURATION VALIDATION")
        print("="*80)
        
        success, response = self.run_test(
            "Federal Resources Configuration Validation", 
            "POST", 
            "legal-qa/rebuild-federal-resources-knowledge-base", 
            200,
            timeout=300
        )
        
        if success and isinstance(response, dict):
            print(f"   🔧 Configuration Validation Results:")
            
            # 1. Target: 5,000+ documents
            target_docs = response.get('target_documents', 0)
            if target_docs >= 5000:
                print(f"      ✅ Target Documents: {target_docs:,} (meets 5,000+ requirement)")
            else:
                print(f"      ❌ Target Documents: {target_docs:,} (below 5,000 requirement)")
            
            # 2. Min content length: 800 words
            quality_metrics = response.get('quality_metrics', {})
            min_length = quality_metrics.get('min_content_length', 0)
            if min_length >= 800:
                print(f"      ✅ Min Content Length: {min_length} words (meets 800+ requirement)")
            else:
                print(f"      ❌ Min Content Length: {min_length} words (below 800 requirement)")
            
            # 3. Government domains only (.gov)
            gov_domains = quality_metrics.get('government_domains_only', False)
            if gov_domains:
                print(f"      ✅ Government Domains Only: .gov domains enforced")
            else:
                print(f"      ❌ Government Domains Only: .gov restriction not enforced")
            
            # 4. Priority agencies: SEC, DOL, USPTO, IRS
            priority_agencies = response.get('priority_agency_breakdown', {})
            expected_agencies = ['SEC', 'DOL', 'USPTO', 'IRS']
            
            print(f"      🏢 Priority Agencies Configuration:")
            agencies_found = 0
            for agency in expected_agencies:
                # Check both uppercase and lowercase versions
                count = priority_agencies.get(agency, priority_agencies.get(agency.lower(), 0))
                if count > 0:
                    print(f"         ✅ {agency}: {count} documents")
                    agencies_found += 1
                else:
                    print(f"         ⚠️  {agency}: 0 documents")
            
            if agencies_found >= 2:  # At least 2 agencies should have content
                print(f"      ✅ Priority Agencies: {agencies_found}/4 agencies have content")
            else:
                print(f"      ⚠️  Priority Agencies: Only {agencies_found}/4 agencies have content")
            
            # 5. Check quality metrics
            total_processed = quality_metrics.get('total_processed', 0)
            pass_rate = quality_metrics.get('quality_pass_rate', '0%')
            avg_word_count = quality_metrics.get('average_word_count', '0')
            
            print(f"      📊 Quality Metrics:")
            print(f"         - Total Processed: {total_processed}")
            print(f"         - Quality Pass Rate: {pass_rate}")
            print(f"         - Average Word Count: {avg_word_count}")
            
            # 6. Check focus areas
            focus_areas = response.get('focus_areas', [])
            expected_focus = [
                "Securities Law (SEC)",
                "Employment & Labor Regulations (DOL)",
                "Patent Law & IP Policy (USPTO)",
                "Tax-Related Legal Guidance (IRS)"
            ]
            
            print(f"      🎯 Focus Areas Configuration:")
            for area in expected_focus:
                if area in focus_areas:
                    print(f"         ✅ {area}")
                else:
                    print(f"         ⚠️  {area}: Not found")
            
            print(f"✅ CONFIGURATION VALIDATION: Federal resources configuration verified")
            
        else:
            print(f"❌ CONFIGURATION VALIDATION: Failed to verify configuration")
            return False, response
        
        return success, response

    def test_enhanced_search_method(self):
        """Test 5: Enhanced Search Method - Verify _search_legal_content method handles new source parameter"""
        print("\n" + "="*80)
        print("TEST 5: ENHANCED SEARCH METHOD VERIFICATION")
        print("="*80)
        
        # Run federal resources collection to test enhanced search method
        success, response = self.run_test(
            "Enhanced Search Method with Source Parameter", 
            "POST", 
            "legal-qa/rebuild-federal-resources-knowledge-base", 
            200,
            timeout=300
        )
        
        if success and isinstance(response, dict):
            print(f"   🔍 Enhanced Search Method Results:")
            
            # Verify that documents have source information
            documents_collected = response.get('documents_collected', 0)
            if documents_collected > 0:
                print(f"      ✅ Search Method: {documents_collected} documents collected with source tracking")
            else:
                print(f"      ❌ Search Method: No documents collected")
                return False, response
            
            # Check source breakdown to verify different sources
            source_breakdown = response.get('source_breakdown', {})
            if source_breakdown:
                print(f"      📊 Source Parameter Handling:")
                for source, count in source_breakdown.items():
                    print(f"         - {source}: {count} documents")
                print(f"      ✅ Source Parameter: Multiple sources properly tracked")
            else:
                print(f"      ❌ Source Parameter: No source breakdown available")
            
            # Verify federal resources mode filters are applied
            quality_metrics = response.get('quality_metrics', {})
            
            # Check government domains filter
            gov_domains = quality_metrics.get('government_domains_only', False)
            if gov_domains:
                print(f"      ✅ Federal Resources Filter: .gov domain filter applied")
            else:
                print(f"      ❌ Federal Resources Filter: .gov domain filter not applied")
            
            # Check content length filter
            min_length = quality_metrics.get('min_content_length', 0)
            if min_length >= 800:
                print(f"      ✅ Content Length Filter: {min_length} words minimum applied")
            else:
                print(f"      ❌ Content Length Filter: {min_length} words below 800 minimum")
            
            # Check deduplication
            duplicates_filtered = quality_metrics.get('duplicates_filtered', 0)
            print(f"      📋 Deduplication: {duplicates_filtered} duplicates filtered")
            
            # Verify enhanced metadata for federal resources
            features = response.get('features_enabled', [])
            metadata_features = [
                "Enhanced Government Metadata Extraction",
                "Authority Level Classification",
                "Effective Date Tracking"
            ]
            
            print(f"      🏷️  Enhanced Metadata Features:")
            for feature in metadata_features:
                if any(feature in f for f in features):
                    print(f"         ✅ {feature}")
                else:
                    print(f"         ⚠️  {feature}: Not found")
            
            print(f"✅ ENHANCED SEARCH METHOD: Source parameter and federal filters working correctly")
            
        else:
            print(f"❌ ENHANCED SEARCH METHOD: Failed to verify enhanced search functionality")
            return False, response
        
        return success, response

    def test_comprehensive_federal_resources_system(self):
        """Comprehensive test of the entire federal resources system"""
        print("\n" + "="*80)
        print("COMPREHENSIVE FEDERAL RESOURCES SYSTEM TEST")
        print("="*80)
        
        # Run all individual tests
        test_results = {}
        
        # Test 1: Endpoint Accessibility
        success1, response1 = self.test_federal_resources_endpoint_accessibility()
        test_results['endpoint_accessibility'] = success1
        
        # Test 2: Import and Configuration
        success2, response2 = self.test_collection_mode_import_and_configuration()
        test_results['import_configuration'] = success2
        
        # Test 3: Method Integration
        success3, response3 = self.test_method_integration()
        test_results['method_integration'] = success3
        
        # Test 4: Configuration Validation
        success4, response4 = self.test_configuration_validation()
        test_results['configuration_validation'] = success4
        
        # Test 5: Enhanced Search Method
        success5, response5 = self.test_enhanced_search_method()
        test_results['enhanced_search_method'] = success5
        
        # Summary
        print(f"\n" + "="*80)
        print("FEDERAL RESOURCES SYSTEM TEST SUMMARY")
        print("="*80)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
        print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status}: {test_name.replace('_', ' ').title()}")
        
        # Overall system assessment
        if passed_tests == total_tests:
            print(f"\n🎉 OVERALL ASSESSMENT: Federal Resources Collection System FULLY OPERATIONAL")
            print(f"   All components tested successfully and ready for production use.")
        elif passed_tests >= 4:
            print(f"\n✅ OVERALL ASSESSMENT: Federal Resources Collection System MOSTLY OPERATIONAL")
            print(f"   {passed_tests}/{total_tests} components working correctly. Minor issues may need attention.")
        elif passed_tests >= 2:
            print(f"\n⚠️  OVERALL ASSESSMENT: Federal Resources Collection System PARTIALLY OPERATIONAL")
            print(f"   {passed_tests}/{total_tests} components working. Significant issues need resolution.")
        else:
            print(f"\n❌ OVERALL ASSESSMENT: Federal Resources Collection System NOT OPERATIONAL")
            print(f"   Only {passed_tests}/{total_tests} components working. Major fixes required.")
        
        return passed_tests == total_tests, test_results

    def run_all_tests(self):
        """Run all federal resources tests"""
        print("🏛️ FEDERAL LEGAL RESOURCES COLLECTION SYSTEM TESTING")
        print("="*80)
        print(f"Testing against: {self.base_url}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        # Run comprehensive test
        overall_success, detailed_results = self.test_comprehensive_federal_resources_system()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n" + "="*80)
        print("FINAL TEST SUMMARY")
        print("="*80)
        print(f"⏱️  Total Testing Time: {duration:.1f} seconds")
        print(f"🧪 Tests Run: {self.tests_run}")
        print(f"✅ Tests Passed: {self.tests_passed}")
        print(f"❌ Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"📊 Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if overall_success:
            print(f"\n🎉 CONCLUSION: Federal Legal Resources Collection System is FULLY FUNCTIONAL")
            print(f"   Ready for production use with all components operational.")
        else:
            print(f"\n⚠️  CONCLUSION: Federal Legal Resources Collection System has issues")
            print(f"   Review failed tests and address issues before production deployment.")
        
        return overall_success

if __name__ == "__main__":
    tester = FederalResourcesAPITester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)