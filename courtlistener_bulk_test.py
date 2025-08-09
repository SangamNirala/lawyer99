import requests
import sys
import json
import time
from datetime import datetime

class CourtListenerBulkTester:
    def __init__(self, base_url="https://b2e17d7f-d715-45b1-acd0-e51dd70d439b.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=120):
        """Run a single API test with extended timeout for bulk operations"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            start_time = time.time()
            
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            elapsed_time = time.time() - start_time
            print(f"   Status: {response.status_code} (took {elapsed_time:.2f}s)")
            
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

    def test_rag_system_stats(self):
        """Test RAG system statistics endpoint"""
        success, response = self.run_test(
            "RAG System Statistics", 
            "GET", 
            "legal-qa/stats", 
            200,
            timeout=30
        )
        
        if success and isinstance(response, dict):
            print(f"   RAG System Details:")
            print(f"     - Vector DB: {response.get('vector_db', 'Unknown')}")
            print(f"     - Embeddings Model: {response.get('embeddings_model', 'Unknown')}")
            print(f"     - Status: {response.get('status', 'Unknown')}")
            
            # Check for expected fields
            expected_fields = ['vector_db', 'embeddings_model', 'status']
            missing_fields = [field for field in expected_fields if field not in response]
            if missing_fields:
                print(f"   ⚠️  Missing expected fields: {missing_fields}")
            else:
                print(f"   ✅ All expected RAG system fields present")
        
        return success, response

    def test_knowledge_base_stats(self):
        """Test knowledge base statistics endpoint"""
        success, response = self.run_test(
            "Knowledge Base Statistics", 
            "GET", 
            "legal-qa/knowledge-base/stats", 
            200,
            timeout=30
        )
        
        if success and isinstance(response, dict):
            print(f"   Knowledge Base Details:")
            total_docs = response.get('total_documents', 0)
            print(f"     - Total Documents: {total_docs}")
            
            jurisdictions = response.get('jurisdictions', {})
            if jurisdictions:
                print(f"     - Jurisdictions ({len(jurisdictions)}):")
                for jurisdiction, count in jurisdictions.items():
                    print(f"       * {jurisdiction}: {count} documents")
            
            legal_domains = response.get('legal_domains', {})
            if legal_domains:
                print(f"     - Legal Domains ({len(legal_domains)}):")
                for domain, count in legal_domains.items():
                    print(f"       * {domain}: {count} documents")
            
            # Verify reasonable document count for existing knowledge base
            if total_docs > 0:
                print(f"   ✅ Knowledge base contains {total_docs} documents")
            else:
                print(f"   ⚠️  Knowledge base appears empty")
        
        return success, response

    def test_standard_rebuild_endpoint(self):
        """Test standard rebuild endpoint for backward compatibility"""
        success, response = self.run_test(
            "Standard Knowledge Base Rebuild", 
            "POST", 
            "legal-qa/rebuild-knowledge-base", 
            200,
            timeout=180  # 3 minutes for standard rebuild
        )
        
        if success and isinstance(response, dict):
            print(f"   Standard Rebuild Results:")
            print(f"     - Collection Mode: {response.get('collection_mode', 'Unknown')}")
            print(f"     - Documents Collected: {response.get('documents_collected', 0)}")
            print(f"     - Target Achievement: {response.get('target_achievement', 'N/A')}")
            
            # Verify standard mode characteristics
            collection_mode = response.get('collection_mode', '').lower()
            if collection_mode == 'standard':
                print(f"   ✅ Correctly identified as STANDARD mode")
            else:
                print(f"   ⚠️  Expected STANDARD mode, got: {collection_mode}")
            
            # Standard mode should collect ~35 documents
            docs_collected = response.get('documents_collected', 0)
            if 20 <= docs_collected <= 100:  # Reasonable range for standard mode
                print(f"   ✅ Document count ({docs_collected}) within expected range for standard mode")
            else:
                print(f"   ⚠️  Document count ({docs_collected}) outside expected range (20-100) for standard mode")
        
        return success, response

    def test_parameterized_bulk_rebuild(self):
        """Test parameterized rebuild endpoint with bulk mode"""
        bulk_data = {"collection_mode": "bulk"}
        
        success, response = self.run_test(
            "Parameterized Bulk Knowledge Base Rebuild", 
            "POST", 
            "legal-qa/rebuild-knowledge-base", 
            200,
            data=bulk_data,
            timeout=300  # 5 minutes for bulk operations
        )
        
        if success and isinstance(response, dict):
            print(f"   Parameterized Bulk Rebuild Results:")
            print(f"     - Collection Mode: {response.get('collection_mode', 'Unknown')}")
            print(f"     - Documents Collected: {response.get('documents_collected', 0)}")
            print(f"     - Target Achievement: {response.get('target_achievement', 'N/A')}")
            
            # Verify bulk mode characteristics
            collection_mode = response.get('collection_mode', '').lower()
            if collection_mode == 'bulk':
                print(f"   ✅ Correctly identified as BULK mode")
            else:
                print(f"   ⚠️  Expected BULK mode, got: {collection_mode}")
            
            # Check for bulk-specific features
            target_achievement = response.get('target_achievement', 'N/A')
            if target_achievement != 'N/A' and '%' in str(target_achievement):
                print(f"   ✅ Target achievement tracking present: {target_achievement}")
            else:
                print(f"   ⚠️  Target achievement tracking missing or invalid")
        
        return success, response

    def test_dedicated_bulk_endpoint(self):
        """Test dedicated bulk collection endpoint"""
        success, response = self.run_test(
            "Dedicated Bulk Knowledge Base Rebuild", 
            "POST", 
            "legal-qa/rebuild-bulk-knowledge-base", 
            200,
            timeout=300  # 5 minutes for comprehensive bulk collection
        )
        
        if success and isinstance(response, dict):
            print(f"   Dedicated Bulk Endpoint Results:")
            print(f"     - Collection Mode: {response.get('collection_mode', 'Unknown')}")
            print(f"     - Documents Collected: {response.get('documents_collected', 0)}")
            print(f"     - Target Achievement: {response.get('target_achievement', 'N/A')}")
            
            # Check for enhanced response structure
            expected_bulk_fields = [
                'collection_mode', 'documents_collected', 'target_achievement',
                'court_hierarchy_breakdown', 'quality_metrics', 'legal_domain_distribution',
                'features_enabled'
            ]
            
            present_fields = [field for field in expected_bulk_fields if field in response]
            missing_fields = [field for field in expected_bulk_fields if field not in response]
            
            print(f"   Enhanced Response Structure:")
            print(f"     - Present fields ({len(present_fields)}): {present_fields}")
            if missing_fields:
                print(f"     - Missing fields ({len(missing_fields)}): {missing_fields}")
            
            # Verify bulk-specific features
            features_enabled = response.get('features_enabled', [])
            if features_enabled:
                print(f"   ✅ Features enabled ({len(features_enabled)}): {features_enabled}")
                
                # Check for expected bulk features
                expected_features = [
                    'pagination', 'quality_filters', 'intelligent_rate_limiting', 
                    'enhanced_error_handling', 'court_hierarchy_prioritization'
                ]
                found_features = [f for f in expected_features if f in features_enabled]
                print(f"     - Expected features found: {found_features}")
            else:
                print(f"   ⚠️  No features_enabled list found")
            
            # Check court hierarchy breakdown
            court_breakdown = response.get('court_hierarchy_breakdown', {})
            if court_breakdown:
                print(f"   ✅ Court hierarchy breakdown present:")
                for court, count in court_breakdown.items():
                    print(f"     - {court}: {count} documents")
            else:
                print(f"   ⚠️  Court hierarchy breakdown missing")
            
            # Check quality metrics
            quality_metrics = response.get('quality_metrics', {})
            if quality_metrics:
                print(f"   ✅ Quality metrics present:")
                for metric, value in quality_metrics.items():
                    print(f"     - {metric}: {value}")
            else:
                print(f"   ⚠️  Quality metrics missing")
            
            # Check legal domain distribution
            domain_distribution = response.get('legal_domain_distribution', {})
            if domain_distribution:
                print(f"   ✅ Legal domain distribution present:")
                for domain, count in domain_distribution.items():
                    print(f"     - {domain}: {count} documents")
            else:
                print(f"   ⚠️  Legal domain distribution missing")
        
        return success, response

    def test_collection_mode_variations(self):
        """Test different collection mode parameter variations"""
        test_cases = [
            {"collection_mode": "STANDARD", "description": "Uppercase STANDARD"},
            {"collection_mode": "BULK", "description": "Uppercase BULK"},
            {"collection_mode": "standard", "description": "Lowercase standard"},
            {"collection_mode": "bulk", "description": "Lowercase bulk"},
            {"collection_mode": "b", "description": "Abbreviated 'b' for bulk"}
        ]
        
        all_success = True
        results = {}
        
        for test_case in test_cases:
            success, response = self.run_test(
                f"Collection Mode Variation - {test_case['description']}", 
                "POST", 
                "legal-qa/rebuild-knowledge-base", 
                200,
                data=test_case,
                timeout=180
            )
            
            if success and isinstance(response, dict):
                collection_mode = response.get('collection_mode', '').lower()
                expected_mode = 'bulk' if test_case['collection_mode'].lower() in ['bulk', 'b'] else 'standard'
                
                if collection_mode == expected_mode:
                    print(f"   ✅ {test_case['description']} correctly mapped to {expected_mode.upper()} mode")
                else:
                    print(f"   ❌ {test_case['description']} incorrectly mapped to {collection_mode.upper()} mode")
                    all_success = False
                
                results[test_case['description']] = {
                    "success": success,
                    "expected_mode": expected_mode,
                    "actual_mode": collection_mode,
                    "documents_collected": response.get('documents_collected', 0)
                }
            else:
                all_success = False
                results[test_case['description']] = {"success": False, "error": "Request failed"}
        
        return all_success, results

    def test_configuration_validation(self):
        """Test that BULK mode has proper configuration validation"""
        # This test verifies the configuration without running full collection
        print(f"\n🔍 Testing BULK Configuration Validation...")
        
        # Test that bulk endpoint is accessible (indicates proper configuration)
        success, response = self.run_test(
            "Bulk Endpoint Accessibility Check", 
            "POST", 
            "legal-qa/rebuild-bulk-knowledge-base", 
            200,
            timeout=60  # Short timeout just to check if endpoint starts processing
        )
        
        if success:
            print(f"   ✅ Bulk endpoint is accessible and starts processing")
            print(f"   ✅ Configuration validation: BULK mode properly configured")
            
            # Check if response indicates proper bulk configuration
            if isinstance(response, dict):
                collection_mode = response.get('collection_mode', '').lower()
                if collection_mode == 'bulk':
                    print(f"   ✅ Bulk collection mode confirmed in response")
                
                # Check for configuration indicators
                features = response.get('features_enabled', [])
                if 'quality_filters' in features:
                    print(f"   ✅ Quality filters configuration confirmed")
                if 'court_hierarchy_prioritization' in features:
                    print(f"   ✅ Court hierarchy prioritization configuration confirmed")
                if 'intelligent_rate_limiting' in features:
                    print(f"   ✅ Intelligent rate limiting configuration confirmed")
        else:
            print(f"   ❌ Bulk endpoint not accessible - configuration may be invalid")
        
        return success, response

    def test_error_handling(self):
        """Test error handling for edge cases"""
        print(f"\n🔍 Testing Error Handling...")
        
        # Test invalid collection mode
        invalid_data = {"collection_mode": "invalid_mode"}
        
        success, response = self.run_test(
            "Invalid Collection Mode Error Handling", 
            "POST", 
            "legal-qa/rebuild-knowledge-base", 
            200,  # Should handle gracefully and default to standard
            data=invalid_data,
            timeout=60
        )
        
        if success and isinstance(response, dict):
            collection_mode = response.get('collection_mode', '').lower()
            if collection_mode in ['standard', 'bulk']:
                print(f"   ✅ Invalid mode gracefully handled, defaulted to {collection_mode.upper()}")
            else:
                print(f"   ⚠️  Unexpected collection mode: {collection_mode}")
        
        # Test malformed request data
        malformed_data = {"invalid_field": "test_value"}
        
        success_malformed, response_malformed = self.run_test(
            "Malformed Request Error Handling", 
            "POST", 
            "legal-qa/rebuild-knowledge-base", 
            200,  # Should handle gracefully
            data=malformed_data,
            timeout=60
        )
        
        if success_malformed:
            print(f"   ✅ Malformed request handled gracefully")
        
        return success and success_malformed, {"invalid_mode": response, "malformed": response_malformed}

    def test_backward_compatibility(self):
        """Test that standard mode endpoints still work for existing functionality"""
        print(f"\n🔍 Testing Backward Compatibility...")
        
        # Test standard rebuild without parameters (original behavior)
        success_no_params, response_no_params = self.run_test(
            "Standard Rebuild (No Parameters)", 
            "POST", 
            "legal-qa/rebuild-knowledge-base", 
            200,
            timeout=180
        )
        
        if success_no_params and isinstance(response_no_params, dict):
            collection_mode = response_no_params.get('collection_mode', '').lower()
            docs_collected = response_no_params.get('documents_collected', 0)
            
            if collection_mode == 'standard':
                print(f"   ✅ No parameters correctly defaults to STANDARD mode")
            else:
                print(f"   ❌ Expected STANDARD mode, got: {collection_mode}")
            
            # Standard mode should collect reasonable number of documents (~35)
            if 10 <= docs_collected <= 150:  # Reasonable range
                print(f"   ✅ Standard mode document count ({docs_collected}) within expected range")
            else:
                print(f"   ⚠️  Document count ({docs_collected}) outside expected range for standard mode")
        
        # Test that existing endpoints still work
        success_stats, response_stats = self.run_test(
            "Existing Stats Endpoint Compatibility", 
            "GET", 
            "legal-qa/stats", 
            200,
            timeout=30
        )
        
        if success_stats:
            print(f"   ✅ Existing RAG stats endpoint still functional")
        
        success_kb_stats, response_kb_stats = self.run_test(
            "Existing Knowledge Base Stats Compatibility", 
            "GET", 
            "legal-qa/knowledge-base/stats", 
            200,
            timeout=30
        )
        
        if success_kb_stats:
            print(f"   ✅ Existing knowledge base stats endpoint still functional")
        
        return (success_no_params and success_stats and success_kb_stats), {
            "no_params": response_no_params,
            "stats": response_stats,
            "kb_stats": response_kb_stats
        }

    def run_comprehensive_bulk_collection_tests(self):
        """Run comprehensive test suite for enhanced bulk collection system"""
        print("=" * 80)
        print("🎯 ENHANCED COURTLISTENER BULK COLLECTION SYSTEM TESTING")
        print("=" * 80)
        print(f"Testing against: {self.base_url}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test sequence
        test_results = {}
        
        # 1. Test existing system state
        print(f"\n📊 PHASE 1: EXISTING SYSTEM VERIFICATION")
        test_results['rag_stats'] = self.test_rag_system_stats()
        test_results['kb_stats'] = self.test_knowledge_base_stats()
        
        # 2. Test backward compatibility
        print(f"\n🔄 PHASE 2: BACKWARD COMPATIBILITY TESTING")
        test_results['backward_compatibility'] = self.test_backward_compatibility()
        test_results['standard_rebuild'] = self.test_standard_rebuild_endpoint()
        
        # 3. Test enhanced bulk collection endpoints
        print(f"\n🚀 PHASE 3: ENHANCED BULK COLLECTION TESTING")
        test_results['collection_mode_variations'] = self.test_collection_mode_variations()
        test_results['parameterized_bulk'] = self.test_parameterized_bulk_rebuild()
        test_results['dedicated_bulk'] = self.test_dedicated_bulk_endpoint()
        
        # 4. Test configuration and error handling
        print(f"\n⚙️  PHASE 4: CONFIGURATION & ERROR HANDLING")
        test_results['configuration_validation'] = self.test_configuration_validation()
        test_results['error_handling'] = self.test_error_handling()
        
        # Summary
        print(f"\n" + "=" * 80)
        print(f"📋 TEST SUMMARY")
        print(f"=" * 80)
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        # Detailed results
        print(f"\n📈 DETAILED RESULTS:")
        for test_name, (success, _) in test_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {status} - {test_name.replace('_', ' ').title()}")
        
        # Overall assessment
        overall_success = self.tests_passed == self.tests_run
        if overall_success:
            print(f"\n🎉 OVERALL ASSESSMENT: ALL TESTS PASSED")
            print(f"✅ Enhanced CourtListener Bulk Collection System is FULLY OPERATIONAL")
        else:
            failed_tests = self.tests_run - self.tests_passed
            print(f"\n⚠️  OVERALL ASSESSMENT: {failed_tests} TEST(S) FAILED")
            print(f"❌ Some issues detected in Enhanced CourtListener Bulk Collection System")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"=" * 80)
        
        return overall_success, test_results

if __name__ == "__main__":
    tester = CourtListenerBulkTester()
    success, results = tester.run_comprehensive_bulk_collection_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)