#!/usr/bin/env python3
"""
Debug version of the Streamlit app to identify loading issues.
"""
import streamlit as st
import time
import os
import sys
import traceback

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="Fake News Detector - Debug",
    page_icon="🔧",
    layout="wide"
)

def test_imports():
    """Test all imports step by step."""
    results = {}
    
    try:
        st.write("🔍 Testing imports...")
        
        # Test 1: Basic imports
        st.write("1. Testing basic imports...")
        import logging
        results['logging'] = "✅ OK"
        
        # Test 2: Sample data
        st.write("2. Testing sample data...")
        from sample_data import get_sample_data
        sample_data = get_sample_data()
        results['sample_data'] = f"✅ OK ({len(sample_data)} samples)"
        
        # Test 3: ContentProcessor
        st.write("3. Testing ContentProcessor...")
        from src.preprocessing.content_processor import ContentProcessor
        results['content_processor'] = "✅ OK"
        
        # Test 4: PredictionService
        st.write("4. Testing PredictionService...")
        from src.api.prediction_service import PredictionService
        results['prediction_service'] = "✅ OK"
        
        # Test 5: MLClassifierInterface
        st.write("5. Testing MLClassifierInterface...")
        from src.models.base import MLClassifierInterface
        results['ml_interface'] = "✅ OK"
        
        # Test 6: Initialize ContentProcessor
        st.write("6. Initializing ContentProcessor...")
        processor = ContentProcessor()
        results['processor_init'] = "✅ OK"
        
        # Test 7: Fit processor with sample data
        st.write("7. Fitting processor with sample data...")
        mock_texts = [item['content'] for item in sample_data]
        mock_labels = [item['label'] for item in sample_data]
        processor.fit(mock_texts, mock_labels)
        results['processor_fit'] = "✅ OK"
        
        # Test 8: Create mock classifier
        st.write("8. Creating mock classifier...")
        class MockClassifier(MLClassifierInterface):
            def __init__(self):
                super().__init__()
                self.is_trained = True
            
            def predict(self, features):
                import numpy as np
                prediction = 1 if np.random.random() > 0.6 else 0
                confidence = 0.7 + np.random.random() * 0.25
                return prediction, confidence
            
            def train(self, X_train, y_train):
                self.is_trained = True
                
            def evaluate(self, X_test, y_test):
                from src.models.data_models import ModelMetrics
                return ModelMetrics(
                    accuracy=0.85, precision=0.82, recall=0.88, 
                    f1_score=0.85, confusion_matrix=[[45, 5], [6, 44]], roc_auc=0.85
                )
                
            def save_model(self, filepath):
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'w') as f:
                    f.write("mock_model")
                    
            def load_model(self, filepath):
                self.is_trained = True
            
            def get_feature_importance(self):
                return {"suspicious": 0.8, "breaking": 0.6, "exclusive": 0.5}
        
        classifier = MockClassifier()
        results['mock_classifier'] = "✅ OK"
        
        # Test 9: Create PredictionService
        st.write("9. Creating PredictionService...")
        service = PredictionService(
            classifier=classifier,
            content_processor=processor,
            enable_caching=True
        )
        results['service_creation'] = "✅ OK"
        
        # Test 10: Test prediction
        st.write("10. Testing prediction...")
        test_text = "This is a test news article about technology and science."
        result = service.predict(test_text)
        results['test_prediction'] = f"✅ OK ({result.classification})"
        
        return results, service
        
    except Exception as e:
        results['error'] = f"❌ {str(e)}"
        st.error(f"Error during testing: {str(e)}")
        st.code(traceback.format_exc())
        return results, None

def main():
    """Main debug application."""
    st.title("🔧 Fake News Detector - Debug Mode")
    st.write("This debug version tests each component step by step.")
    
    if st.button("Run Diagnostic Tests"):
        with st.spinner("Running tests..."):
            results, service = test_imports()
        
        st.subheader("📊 Test Results")
        for test_name, result in results.items():
            st.write(f"**{test_name}:** {result}")
        
        if service:
            st.success("🎉 All tests passed! The system should work properly.")
            
            # Test the service
            st.subheader("🧪 Live Test")
            test_text = st.text_area("Enter test text:", value="This is a sample news article for testing.")
            
            if st.button("Test Prediction") and test_text:
                with st.spinner("Making prediction..."):
                    try:
                        result = service.predict(test_text)
                        st.success(f"Prediction: {result.classification} ({result.confidence:.1%} confidence)")
                        st.json(result.to_dict())
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
                        st.code(traceback.format_exc())
        else:
            st.error("❌ Tests failed. Check the error messages above.")
    
    st.subheader("💡 Next Steps")
    st.write("""
    If all tests pass:
    1. The main app should work properly
    2. Try running `streamlit run app.py`
    
    If tests fail:
    1. Check the error messages above
    2. Run `python check_requirements.py`
    3. Run `python validate_integration.py`
    """)

if __name__ == "__main__":
    main()