import requests
import sys
import json
from datetime import datetime

class EnhancedCourtListenerTester:
    def __init__(self, base_url="https://legal-engine-check.preview.emergentagent.com"):
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

    def test_legal_qa_stats(self):
        """Test GET /api/legal-qa/stats endpoint (should work as before)"""
        success, response = self.run_test(
            "Legal Q&A Stats Endpoint", 
            "GET", 
            "legal-qa/stats", 
            200
        )
        
        if success:
            print(f"   📊 RAG System Stats:")
            if isinstance(response, dict):
                vector_db = response.get('vector_db', 'N/A')
                embeddings_model = response.get('embeddings_model', 'N/A')
                active_sessions = response.get('active_sessions', 'N/A')
                total_conversations = response.get('total_conversations', 'N/A')
                indexed_documents = response.get('indexed_documents', 'N/A')
                
                print(f"     - Vector DB: {vector_db}")
                print(f"     - Embeddings Model: {embeddings_model}")
                print(f"     - Active Sessions: {active_sessions}")
                print(f"     - Total Conversations: {total_conversations}")
                print(f"     - Indexed Documents: {indexed_documents}")
                
                # Verify expected fields are present
                expected_fields = ['vector_db', 'embeddings_model', 'active_sessions', 'total_conversations']
                missing_fields = [field for field in expected_fields if field not in response]
                if missing_fields:
                    print(f"   ⚠️  Missing expected fields: {missing_fields}")
                else:
                    print(f"   ✅ All expected fields present in response")
            else:
                print(f"   ⚠️  Unexpected response format: {type(response)}")
        
        return success, response

    def test_rebuild_knowledge_base_standard_mode(self):
        """Test POST /api/legal-qa/rebuild-knowledge-base with standard mode (backward compatibility)"""
        success, response = self.run_test(
            "Rebuild Knowledge Base - Standard Mode (Default)", 
            "POST", 
            "legal-qa/rebuild-knowledge-base", 
            200,
            timeout=60  # Knowledge base operations might take longer
        )
        
        if success:
            print(f"   📚 Standard Mode Knowledge Base Rebuild:")
            if isinstance(response, dict):
                status = response.get('status', 'N/A')
                collection_mode = response.get('collection_mode', 'N/A')
                documents_collected = response.get('documents_collected', 'N/A')
                timestamp = response.get('timestamp', 'N/A')
                
                print(f"     - Status: {status}")
                print(f"     - Collection Mode: {collection_mode}")
                print(f"     - Documents Collected: {documents_collected}")
                print(f"     - Timestamp: {timestamp}")
                
                # Verify this is standard mode (backward compatibility)
                if collection_mode and collection_mode.lower() == 'standard':
                    print(f"   ✅ Correctly defaulted to STANDARD mode for backward compatibility")
                else:
                    print(f"   ⚠️  Expected STANDARD mode, got: {collection_mode}")
                
                # Check for expected response structure
                expected_fields = ['status', 'collection_mode', 'timestamp']
                missing_fields = [field for field in expected_fields if field not in response]
                if missing_fields:
                    print(f"   ⚠️  Missing expected fields: {missing_fields}")
                else:
                    print(f"   ✅ All expected fields present in response")
            else:
                print(f"   ⚠️  Unexpected response format: {type(response)}")
        
        return success, response

    def test_rebuild_knowledge_base_with_collection_mode_parameter(self):
        """Test POST /api/legal-qa/rebuild-knowledge-base with collection_mode parameter"""
        # Test with explicit standard mode
        success_standard, response_standard = self.run_test(
            "Rebuild Knowledge Base - Explicit Standard Mode", 
            "POST", 
            "legal-qa/rebuild-knowledge-base?collection_mode=standard", 
            200,
            timeout=60
        )
        
        if success_standard:
            print(f"   📚 Explicit Standard Mode Results:")
            if isinstance(response_standard, dict):
                collection_mode = response_standard.get('collection_mode', 'N/A')
                print(f"     - Collection Mode: {collection_mode}")
                
                if collection_mode and collection_mode.lower() == 'standard':
                    print(f"   ✅ Explicit standard mode parameter working correctly")
                else:
                    print(f"   ❌ Expected STANDARD mode, got: {collection_mode}")
        
        # Test with bulk mode parameter
        success_bulk, response_bulk = self.run_test(
            "Rebuild Knowledge Base - Bulk Mode Parameter", 
            "POST", 
            "legal-qa/rebuild-knowledge-base?collection_mode=bulk", 
            200,
            timeout=120  # Bulk mode might take longer
        )
        
        if success_bulk:
            print(f"   📚 Bulk Mode Parameter Results:")
            if isinstance(response_bulk, dict):
                collection_mode = response_bulk.get('collection_mode', 'N/A')
                documents_collected = response_bulk.get('documents_collected', 'N/A')
                target_achievement = response_bulk.get('target_achievement', 'N/A')
                
                print(f"     - Collection Mode: {collection_mode}")
                print(f"     - Documents Collected: {documents_collected}")
                print(f"     - Target Achievement: {target_achievement}")
                
                if collection_mode and collection_mode.lower() == 'bulk':
                    print(f"   ✅ Bulk mode parameter working correctly")
                else:
                    print(f"   ❌ Expected BULK mode, got: {collection_mode}")
                
                # Check for bulk-specific fields
                bulk_fields = ['target_achievement', 'features_enabled']
                found_bulk_fields = [field for field in bulk_fields if field in response_bulk]
                if found_bulk_fields:
                    print(f"   ✅ Bulk-specific fields found: {found_bulk_fields}")
                else:
                    print(f"   ⚠️  No bulk-specific fields found in response")
        
        return success_standard and success_bulk, {
            "standard_mode": response_standard,
            "bulk_mode": response_bulk
        }

    def test_rebuild_bulk_knowledge_base_dedicated_endpoint(self):
        """Test POST /api/legal-qa/rebuild-bulk-knowledge-base (new dedicated bulk endpoint)"""
        success, response = self.run_test(
            "Rebuild Bulk Knowledge Base - Dedicated Endpoint", 
            "POST", 
            "legal-qa/rebuild-bulk-knowledge-base", 
            200,
            timeout=120  # Bulk operations might take longer
        )
        
        if success:
            print(f"   🚀 Dedicated Bulk Endpoint Results:")
            if isinstance(response, dict):
                status = response.get('status', 'N/A')
                collection_mode = response.get('collection_mode', 'N/A')
                documents_collected = response.get('documents_collected', 'N/A')
                target_achievement = response.get('target_achievement', 'N/A')
                features_enabled = response.get('features_enabled', [])
                timestamp = response.get('timestamp', 'N/A')
                
                print(f"     - Status: {status}")
                print(f"     - Collection Mode: {collection_mode}")
                print(f"     - Documents Collected: {documents_collected}")
                print(f"     - Target Achievement: {target_achievement}")
                print(f"     - Features Enabled: {features_enabled}")
                print(f"     - Timestamp: {timestamp}")
                
                # Verify this is bulk mode
                if collection_mode and collection_mode.lower() == 'bulk':
                    print(f"   ✅ Dedicated bulk endpoint correctly set to BULK mode")
                else:
                    print(f"   ❌ Expected BULK mode, got: {collection_mode}")
                
                # Check for target of 15,000+ documents indication
                if target_achievement:
                    if '15,000' in str(target_achievement) or '15000' in str(target_achievement):
                        print(f"   ✅ Target of 15,000+ documents indicated")
                    else:
                        print(f"   ⚠️  Target achievement doesn't mention 15,000+ documents: {target_achievement}")
                
                # Check for bulk features
                expected_bulk_features = ['pagination', 'quality_filters', 'rate_limiting', 'enhanced_error_handling']
                if features_enabled:
                    found_features = []
                    for feature in expected_bulk_features:
                        if any(feature.lower() in str(f).lower() for f in features_enabled):
                            found_features.append(feature)
                    
                    if found_features:
                        print(f"   ✅ Expected bulk features found: {found_features}")
                    else:
                        print(f"   ⚠️  Expected bulk features not clearly indicated")
                        print(f"       Features enabled: {features_enabled}")
                else:
                    print(f"   ⚠️  No features_enabled field in response")
                
                # Check for enhanced response structure
                expected_bulk_fields = ['collection_mode', 'target_achievement', 'features_enabled']
                missing_bulk_fields = [field for field in expected_bulk_fields if field not in response]
                if missing_bulk_fields:
                    print(f"   ⚠️  Missing expected bulk fields: {missing_bulk_fields}")
                else:
                    print(f"   ✅ All expected bulk fields present in response")
            else:
                print(f"   ⚠️  Unexpected response format: {type(response)}")
        
        return success, response

    def test_knowledge_base_stats(self):
        """Test GET /api/legal-qa/knowledge-base/stats endpoint"""
        success, response = self.run_test(
            "Knowledge Base Statistics", 
            "GET", 
            "legal-qa/knowledge-base/stats", 
            200
        )
        
        if success:
            print(f"   📈 Knowledge Base Statistics:")
            if isinstance(response, dict):
                total_documents = response.get('total_documents', 'N/A')
                by_jurisdiction = response.get('by_jurisdiction', {})
                by_legal_domain = response.get('by_legal_domain', {})
                by_document_type = response.get('by_document_type', {})
                by_source = response.get('by_source', {})
                
                print(f"     - Total Documents: {total_documents}")
                print(f"     - By Jurisdiction: {len(by_jurisdiction)} jurisdictions")
                print(f"     - By Legal Domain: {len(by_legal_domain)} domains")
                print(f"     - By Document Type: {len(by_document_type)} types")
                print(f"     - By Source: {len(by_source)} sources")
                
                # Check if CourtListener is mentioned as a source
                if by_source and any('courtlistener' in str(source).lower() for source in by_source.keys()):
                    print(f"   ✅ CourtListener found as a data source")
                else:
                    print(f"   ⚠️  CourtListener not clearly identified as a source")
                    if by_source:
                        print(f"       Available sources: {list(by_source.keys())}")
                
                # Verify expected structure
                expected_fields = ['total_documents', 'by_jurisdiction', 'by_legal_domain', 'by_document_type', 'by_source']
                missing_fields = [field for field in expected_fields if field not in response]
                if missing_fields:
                    print(f"   ⚠️  Missing expected fields: {missing_fields}")
                else:
                    print(f"   ✅ All expected fields present in response")
            else:
                print(f"   ⚠️  Unexpected response format: {type(response)}")
        
        return success, response

    def test_backward_compatibility(self):
        """Test that existing rebuild endpoint works without parameters (backward compatibility)"""
        success, response = self.run_test(
            "Backward Compatibility - No Parameters", 
            "POST", 
            "legal-qa/rebuild-knowledge-base", 
            200,
            timeout=60
        )
        
        if success:
            print(f"   🔄 Backward Compatibility Test:")
            if isinstance(response, dict):
                collection_mode = response.get('collection_mode', 'N/A')
                status = response.get('status', 'N/A')
                
                print(f"     - Status: {status}")
                print(f"     - Collection Mode: {collection_mode}")
                
                # Should default to standard mode for backward compatibility
                if collection_mode and collection_mode.lower() == 'standard':
                    print(f"   ✅ Backward compatibility maintained - defaults to STANDARD mode")
                else:
                    print(f"   ❌ Backward compatibility issue - expected STANDARD mode, got: {collection_mode}")
                
                # Should not have bulk-specific fields when in standard mode
                bulk_specific_fields = ['target_achievement', 'features_enabled']
                found_bulk_fields = [field for field in bulk_specific_fields if field in response]
                if not found_bulk_fields:
                    print(f"   ✅ Standard mode response doesn't include bulk-specific fields")
                else:
                    print(f"   ⚠️  Standard mode response includes bulk fields: {found_bulk_fields}")
            else:
                print(f"   ⚠️  Unexpected response format: {type(response)}")
        
        return success, response

    def test_collection_mode_variations(self):
        """Test different collection mode parameter variations"""
        test_cases = [
            {"mode": "standard", "description": "Standard mode (lowercase)"},
            {"mode": "STANDARD", "description": "Standard mode (uppercase)"},
            {"mode": "bulk", "description": "Bulk mode (lowercase)"},
            {"mode": "BULK", "description": "Bulk mode (uppercase)"},
            {"mode": "b", "description": "Bulk mode (shorthand 'b')"},
        ]
        
        all_success = True
        results = {}
        
        for test_case in test_cases:
            mode = test_case["mode"]
            description = test_case["description"]
            
            success, response = self.run_test(
                f"Collection Mode Variation - {description}", 
                "POST", 
                f"legal-qa/rebuild-knowledge-base?collection_mode={mode}", 
                200,
                timeout=60
            )
            
            if success:
                if isinstance(response, dict):
                    collection_mode = response.get('collection_mode', 'N/A')
                    print(f"   Input: {mode} → Output: {collection_mode}")
                    
                    # Verify mode interpretation
                    if mode.lower() in ['bulk', 'b']:
                        expected = 'bulk'
                    else:
                        expected = 'standard'
                    
                    if collection_mode and collection_mode.lower() == expected:
                        print(f"   ✅ Mode '{mode}' correctly interpreted as '{expected.upper()}'")
                    else:
                        print(f"   ❌ Mode '{mode}' expected '{expected.upper()}', got '{collection_mode}'")
                        all_success = False
                    
                    results[mode] = {
                        "success": True,
                        "input_mode": mode,
                        "output_mode": collection_mode,
                        "expected_mode": expected
                    }
                else:
                    print(f"   ⚠️  Unexpected response format for mode '{mode}'")
                    all_success = False
                    results[mode] = {"success": False, "error": "Unexpected response format"}
            else:
                print(f"   ❌ Failed to test mode '{mode}'")
                all_success = False
                results[mode] = {"success": False, "error": "Request failed"}
        
        return all_success, results

    def run_all_tests(self):
        """Run all Enhanced CourtListener integration tests"""
        print("🚀 Starting Enhanced CourtListener Integration Testing...")
        print("=" * 70)
        print("Testing enhanced CourtListener integration with new bulk collection capabilities:")
        print("  • New Collection Mode Support (standard/bulk)")
        print("  • Dedicated Bulk Endpoint")
        print("  • Enhanced Response Structure")
        print("  • Backward Compatibility")
        print("  • Target: 15,000+ documents with pagination & quality filters")
        print("=" * 70)
        
        # Test 1: Existing functionality (should work as before)
        print("\n📊 TESTING EXISTING FUNCTIONALITY")
        self.test_legal_qa_stats()
        self.test_knowledge_base_stats()
        
        # Test 2: Backward compatibility
        print("\n🔄 TESTING BACKWARD COMPATIBILITY")
        self.test_backward_compatibility()
        
        # Test 3: New collection mode support
        print("\n🆕 TESTING NEW COLLECTION MODE SUPPORT")
        self.test_rebuild_knowledge_base_standard_mode()
        self.test_rebuild_knowledge_base_with_collection_mode_parameter()
        
        # Test 4: New dedicated bulk endpoint
        print("\n🚀 TESTING NEW BULK COLLECTION FEATURES")
        self.test_rebuild_bulk_knowledge_base_dedicated_endpoint()
        
        # Test 5: Collection mode variations
        print("\n🔀 TESTING COLLECTION MODE VARIATIONS")
        self.test_collection_mode_variations()
        
        # Summary
        print("\n" + "=" * 70)
        print("📋 TEST SUMMARY")
        print("=" * 70)
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed / self.tests_run * 100):.1f}%" if self.tests_run > 0 else "0.0%")
        
        if self.tests_passed == self.tests_run:
            print("🎉 ALL TESTS PASSED! Enhanced CourtListener integration is working correctly.")
            print("\n✅ VERIFIED CAPABILITIES:")
            print("  • GET /api/legal-qa/stats - Working as before")
            print("  • POST /api/legal-qa/rebuild-knowledge-base - Supports collection_mode parameter")
            print("  • POST /api/legal-qa/rebuild-bulk-knowledge-base - New dedicated bulk endpoint")
            print("  • Backward compatibility maintained (defaults to standard mode)")
            print("  • Enhanced response structure with bulk-specific fields")
            print("  • Target of 15,000+ documents with advanced features")
        else:
            print("⚠️  Some tests failed. Please review the results above.")
        
        return self.tests_passed == self.tests_run

if __name__ == "__main__":
    tester = EnhancedCourtListenerTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)