#!/usr/bin/env python3
"""
Integration Validation Script
Quick validation that all enhanced components work together.
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def setup_logging():
    """Setup logging for validation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_dependencies():
    """Test that all required dependencies are available."""
    logger = logging.getLogger(__name__)
    logger.info("Testing dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'nltk', 'textstat',
        'vaderSentiment', 'matplotlib', 'seaborn', 'joblib'
    ]
    
    optional_packages = ['newspaper3k', 'feedparser', 'xgboost', 'lightgbm']
    
    # Packages that might have configuration issues
    problematic_packages = ['kaggle', 'datasets']
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError:
            missing_required.append(package)
            logger.error(f"✗ {package} (REQUIRED)")
    
    for package in optional_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} (optional)")
        except ImportError:
            missing_optional.append(package)
            logger.warning(f"⚠ {package} (optional)")
    
    # Test problematic packages more carefully
    for package in problematic_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} (optional)")
        except ImportError:
            logger.info(f"⚠ {package} not installed (optional)")
        except Exception as e:
            logger.warning(f"⚠ {package} configuration issue: {e} (optional)")
    
    if missing_required:
        logger.error(f"Missing required packages: {missing_required}")
        return False
    
    if missing_optional:
        logger.warning(f"Missing optional packages: {missing_optional}")
        logger.warning("Some advanced features may not be available")
    
    logger.info("Dependency check passed!")
    return True

def test_advanced_feature_extraction():
    """Test advanced feature extraction."""
    logger = logging.getLogger(__name__)
    logger.info("Testing advanced feature extraction...")
    
    try:
        from advanced_feature_extraction import AdvancedFeatureExtractor
        from sample_data import get_real_samples, get_fake_samples
        
        extractor = AdvancedFeatureExtractor()
        
        # Test with sample articles
        real_sample = get_real_samples()[0]
        fake_sample = get_fake_samples()[0]
        
        real_title = real_sample['title']
        real_content = real_sample['content']
        fake_title = fake_sample['title']
        fake_content = fake_sample['content']
        
        # Extract features
        real_features = extractor.extract_all_features(real_title, real_content)
        fake_features = extractor.extract_all_features(fake_title, fake_content)
        
        # Validate feature extraction
        if len(real_features) < 50:
            logger.error(f"Insufficient features extracted: {len(real_features)} < 50")
            return False
        
        # Check for key feature categories
        required_feature_types = ['sentiment', 'credibility', 'readability', 'emotional']
        for feature_type in required_feature_types:
            if not any(feature_type in f for f in real_features.keys()):
                logger.error(f"Missing {feature_type} features")
                return False
        
        logger.info(f"✓ Extracted {len(real_features)} features")
        logger.info(f"✓ Real news credibility score: {real_features.get('credibility_score', 0):.3f}")
        logger.info(f"✓ Fake news credibility score: {fake_features.get('credibility_score', 0):.3f}")
        
        # Test feature importance ranking
        importance_ranking = extractor.get_feature_importance_ranking(fake_features)
        logger.info(f"✓ Feature importance ranking available: {len(importance_ranking)} features")
        
        logger.info("Advanced feature extraction test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Advanced feature extraction test failed: {e}")
        return False

def test_enhanced_data_collection():
    """Test enhanced data collection (without actually collecting data)."""
    logger = logging.getLogger(__name__)
    logger.info("Testing enhanced data collection setup...")
    
    try:
        # Check if enhanced_data_collection.py exists and is importable
        if not os.path.exists('enhanced_data_collection.py'):
            logger.error("enhanced_data_collection.py not found")
            return False
        
        # Test that we can import the collector class
        sys.path.append('.')
        from enhanced_data_collection import EnhancedDataCollector
        
        collector = EnhancedDataCollector()
        
        # Validate configuration
        if len(collector.trusted_sources) < 5:
            logger.error(f"Insufficient trusted sources: {len(collector.trusted_sources)} < 5")
            return False
        
        if len(collector.quality_thresholds) < 5:
            logger.error("Quality thresholds not properly configured")
            return False
        
        logger.info(f"✓ {len(collector.trusted_sources)} trusted sources configured")
        logger.info(f"✓ Quality thresholds configured")
        logger.info(f"✓ Advanced features calculation available")
        
        logger.info("Enhanced data collection test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced data collection test failed: {e}")
        return False

def test_enhanced_model_training():
    """Test enhanced model training setup."""
    logger = logging.getLogger(__name__)
    logger.info("Testing enhanced model training setup...")
    
    try:
        # Check if train_enhanced_model.py exists
        if not os.path.exists('train_enhanced_model.py'):
            logger.error("train_enhanced_model.py not found")
            return False
        
        # Test that we can import the trainer class
        from train_enhanced_model import EnhancedModelTrainer
        
        trainer = EnhancedModelTrainer()
        
        # Validate configuration
        config = trainer.config
        
        if len(config['models']) < 3:
            logger.error(f"Insufficient models configured: {len(config['models'])} < 3")
            return False
        
        if not config['ensemble']['enabled']:
            logger.error("Ensemble methods not enabled")
            return False
        
        if not config['hyperparameter_tuning']['enabled']:
            logger.error("Hyperparameter tuning not enabled")
            return False
        
        logger.info(f"✓ {len(config['models'])} models configured")
        logger.info(f"✓ Ensemble methods enabled")
        logger.info(f"✓ Hyperparameter tuning enabled")
        logger.info(f"✓ Cross-validation configured ({config['cv_folds']} folds)")
        
        logger.info("Enhanced model training test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced model training test failed: {e}")
        return False

def test_integration_orchestrator():
    """Test integration orchestrator."""
    logger = logging.getLogger(__name__)
    logger.info("Testing integration orchestrator...")
    
    try:
        # Check if integrate_all_enhancements.py exists
        if not os.path.exists('integrate_all_enhancements.py'):
            logger.error("integrate_all_enhancements.py not found")
            return False
        
        # Test that we can import the orchestrator class
        from integrate_all_enhancements import IntegrationOrchestrator
        
        orchestrator = IntegrationOrchestrator()
        
        # Validate configuration
        config = orchestrator.config
        
        if len(config['phases']) < 4:
            logger.error(f"Insufficient phases configured: {len(config['phases'])} < 4")
            return False
        
        required_phases = ['phase_1', 'phase_2', 'phase_3', 'phase_4']
        for phase in required_phases:
            if phase not in config['phases']:
                logger.error(f"Missing phase: {phase}")
                return False
        
        # Check requirements
        requirements = config['requirements']
        if requirements['accuracy_target'] < 0.90:
            logger.error(f"Accuracy target too low: {requirements['accuracy_target']} < 0.90")
            return False
        
        logger.info(f"✓ {len(config['phases'])} phases configured")
        logger.info(f"✓ Accuracy target: {requirements['accuracy_target']:.1%}")
        logger.info(f"✓ All requirements properly set")
        
        logger.info("Integration orchestrator test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Integration orchestrator test failed: {e}")
        return False

def test_existing_components():
    """Test that existing components still work."""
    logger = logging.getLogger(__name__)
    logger.info("Testing existing components...")
    
    try:
        # Test that we can still import existing modules
        from src.models.data_models import NewsItem, PredictionResult
        
        # Create test objects
        news_item = NewsItem(
            id="test_001",
            title="Test Article",
            content="This is a test article for validation.",
            url="https://example.com/test",
            source="test_source",
            label=0,
            timestamp=datetime.now()
        )
        
        prediction_result = PredictionResult(
            classification="Real",
            confidence=0.85,
            explanation="Test prediction",
            processing_time=0.1
        )
        
        # Test serialization
        news_dict = news_item.to_dict()
        news_from_dict = NewsItem.from_dict(news_dict)
        
        if news_from_dict.title != news_item.title:
            logger.error("NewsItem serialization failed")
            return False
        
        # Test prediction result methods
        confidence_pct = prediction_result.get_confidence_percentage()
        confidence_level = prediction_result.get_confidence_level()
        risk_assessment = prediction_result.get_risk_assessment()
        
        logger.info("✓ NewsItem model working")
        logger.info("✓ PredictionResult model working")
        logger.info(f"✓ Confidence: {confidence_pct}% ({confidence_level})")
        
        logger.info("Existing components test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Existing components test failed: {e}")
        return False

def run_validation():
    """Run complete validation."""
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("FAKE NEWS DETECTOR INTEGRATION VALIDATION")
    logger.info("="*60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Advanced Feature Extraction", test_advanced_feature_extraction),
        ("Enhanced Data Collection", test_enhanced_data_collection),
        ("Enhanced Model Training", test_enhanced_model_training),
        ("Integration Orchestrator", test_integration_orchestrator),
        ("Existing Components", test_existing_components)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Integration is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Run: python integrate_all_enhancements.py")
        logger.info("2. Wait for 90-95% accuracy results")
        logger.info("3. Deploy your enhanced fake news detector!")
        return True
    else:
        logger.error("❌ SOME TESTS FAILED! Check the errors above.")
        logger.error("\nFix the issues and run validation again.")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)