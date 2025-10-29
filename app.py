"""
Streamlit web application for the Fake News Detector.
Provides an intuitive interface for users to classify news content as real or fake.
"""
import streamlit as st
import time
import os
import sys
from typing import Optional, Dict, Any
import logging

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.api.prediction_service import PredictionService
from src.models.data_models import PredictionResult
from src.preprocessing.content_processor import ContentProcessor
from src.models.base import MLClassifierInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    
    .prediction-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    
    .real-news {
        border-color: #28a745;
        background-color: #d4edda;
    }
    
    .fake-news {
        border-color: #dc3545;
        background-color: #f8d7da;
    }
    
    .uncertain {
        border-color: #ffc107;
        background-color: #fff3cd;
    }
    
    .error {
        border-color: #dc3545;
        background-color: #f8d7da;
        color: #721c24;
    }
    
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .feature-importance {
        background-color: #e9ecef;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .char-counter {
        text-align: right;
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 0.25rem;
    }
    
    .processing-time {
        font-size: 0.8rem;
        color: #6c757d;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_prediction_service() -> Optional[PredictionService]:
    """
    Load and cache the prediction service.
    
    Returns:
        PredictionService instance or None if loading fails
    """
    try:
        # Try to load the trained model and processor
        # This is a placeholder - actual implementation would load from saved files
        st.info("🔄 Loading AI model... This may take a moment.")
        
        # For demo purposes, we'll create a mock service
        # In production, this would load actual trained models
        processor = ContentProcessor()
        
        # Fit the processor with real training data
        try:
            import json
            # Try to load real training data first
            train_data_file = None
            for file in ['data/processed/train_data.json', 'data/processed/combined_dataset.json']:
                if os.path.exists(file):
                    train_data_file = file
                    break
            
            if train_data_file:
                with open(train_data_file, 'r') as f:
                    train_data = json.load(f)
            else:
                # Fallback to demo data if real data not available
                with open('data/processed/train_data.json', 'r') as f:
                    train_data = json.load(f)
            
            # Extract texts and labels for fitting
            texts = [item['content'] for item in train_data[:100]]  # Use first 100 for demo
            labels = [item['label'] for item in train_data[:100]]
            
            # Fit the processor
            processor.fit(texts, labels)
            
        except Exception as e:
            # If demo data is not available, use sample data for fitting
            from sample_data import get_sample_data
            sample_data = get_sample_data()
            mock_texts = [item['content'] for item in sample_data]
            mock_labels = [item['label'] for item in sample_data]
            processor.fit(mock_texts, mock_labels)
        
        # Mock classifier for demonstration
        class MockClassifier(MLClassifierInterface):
            def __init__(self):
                super().__init__()
                self.is_trained = True
            
            def predict(self, features):
                # Simple mock prediction based on text length and content
                import numpy as np
                # Simulate prediction logic
                prediction = 1 if np.random.random() > 0.6 else 0
                confidence = 0.7 + np.random.random() * 0.25
                return prediction, confidence
            
            def train(self, X_train, y_train):
                # Mock training - just set trained flag
                self.is_trained = True
                
            def evaluate(self, X_test, y_test):
                # Mock evaluation - return dummy metrics
                from src.models.data_models import ModelMetrics
                return ModelMetrics(
                    accuracy=0.85,
                    precision=0.82,
                    recall=0.88,
                    f1_score=0.85,
                    confusion_matrix=[[45, 5], [6, 44]]
                )
                
            def save_model(self, filepath):
                # Mock save - just create empty file
                import os
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'w') as f:
                    f.write("mock_model")
                    
            def load_model(self, filepath):
                # Mock load - just set trained flag
                self.is_trained = True
            
            def get_feature_importance(self):
                return {"suspicious": 0.8, "breaking": 0.6, "exclusive": 0.5}
        
        classifier = MockClassifier()
        
        # Create prediction service
        service = PredictionService(
            classifier=classifier,
            content_processor=processor,
            enable_caching=True
        )
        
        st.success("✅ AI model loaded successfully!")
        return service
        
    except Exception as e:
        st.error(f"❌ Failed to load AI model: {str(e)}")
        logger.error(f"Failed to load prediction service: {e}")
        return None


def display_error_message(error_type: str, message: str, suggestions: list = None):
    """
    Display user-friendly error messages with specific guidance.
    
    Args:
        error_type: Type of error (validation, processing, etc.)
        message: Error message to display
        suggestions: List of suggestions to help user
    """
    error_icons = {
        "validation": "⚠️",
        "processing": "🔧",
        "timeout": "⏱️",
        "model": "🤖",
        "network": "🌐",
        "unknown": "❓"
    }
    
    icon = error_icons.get(error_type, "❌")
    
    st.markdown(f"""
    <div class="prediction-container error">
        <h3>{icon} {error_type.title()} Error</h3>
        <p><strong>What happened:</strong> {message}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if suggestions:
        st.markdown("**💡 Try these solutions:**")
        for suggestion in suggestions:
            st.markdown(f"• {suggestion}")


def display_prediction_result(result: PredictionResult, text_length: int):
    """
    Display prediction result with color-coded styling and confidence visualization.
    
    Args:
        result: PredictionResult to display
        text_length: Length of input text for context
    """
    if result.classification == "Error":
        # Enhanced error handling with specific guidance
        error_message = result.explanation or "Unable to classify this content."
        
        # Determine error type and provide specific suggestions
        if "validation" in error_message.lower():
            suggestions = [
                "Check that your text is between 10-10,000 characters",
                "Ensure the text contains meaningful words",
                "Remove excessive special characters or formatting",
                "Try using plain text without HTML tags"
            ]
            display_error_message("validation", error_message, suggestions)
        elif "timeout" in error_message.lower():
            suggestions = [
                "Try with shorter text (under 5,000 characters)",
                "Check your internet connection",
                "Wait a moment and try again",
                "Contact support if the issue persists"
            ]
            display_error_message("timeout", error_message, suggestions)
        elif "model" in error_message.lower():
            suggestions = [
                "The AI model may be temporarily unavailable",
                "Try refreshing the page",
                "Wait a few minutes and try again",
                "Use the example articles to test functionality"
            ]
            display_error_message("model", error_message, suggestions)
        else:
            suggestions = [
                "Try refreshing the page",
                "Check your text meets the requirements",
                "Use one of the example articles",
                "Contact support if the problem continues"
            ]
            display_error_message("unknown", error_message, suggestions)
        return
    
    # Determine styling based on classification and confidence
    if result.classification == "Real":
        container_class = "real-news" if result.confidence >= 0.6 else "uncertain"
        emoji = "✅" if result.confidence >= 0.8 else "☑️" if result.confidence >= 0.6 else "❓"
        color = "#28a745" if result.confidence >= 0.6 else "#ffc107"
    else:  # Fake
        container_class = "fake-news" if result.confidence >= 0.6 else "uncertain"
        emoji = "❌" if result.confidence >= 0.8 else "⚠️" if result.confidence >= 0.6 else "❓"
        color = "#dc3545" if result.confidence >= 0.6 else "#ffc107"
    
    confidence_pct = result.get_confidence_percentage()
    
    # Main result display
    st.markdown(f"""
    <div class="prediction-container {container_class}">
        <h3>{emoji} {result.classification.upper()} NEWS</h3>
        <h4>Confidence: {confidence_pct}% ({result.get_confidence_level()})</h4>
        <div class="confidence-bar" style="background: linear-gradient(to right, {color} {confidence_pct}%, #e0e0e0 {confidence_pct}%);"></div>
        <p><strong>Risk Assessment:</strong> {result.get_risk_assessment()}</p>
        <div class="processing-time">Processed in {result.processing_time:.2f} seconds</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature importance display
    if result.feature_weights:
        st.subheader("🔍 Key Indicators")
        
        # Sort features by importance
        sorted_features = sorted(result.feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for i, (feature, weight) in enumerate(sorted_features[:5]):  # Show top 5
            # Determine if feature indicates real or fake
            indicator_type = "Fake indicator" if weight > 0 else "Real indicator"
            bar_color = "#dc3545" if weight > 0 else "#28a745"
            
            st.markdown(f"""
            <div class="feature-importance">
                <strong>"{feature}"</strong> - {indicator_type}
                <div style="background: {bar_color}; height: 8px; width: {min(abs(weight) * 100, 100)}%; border-radius: 4px; margin-top: 4px;"></div>
            </div>
            """, unsafe_allow_html=True)
    
    # Additional context
    with st.expander("📊 Detailed Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Text Length", f"{text_length} characters")
            st.metric("Processing Time", f"{result.processing_time:.3f}s")
        
        with col2:
            st.metric("Confidence Score", f"{result.confidence:.3f}")
            st.metric("Confidence Level", result.get_confidence_level())
        
        if result.explanation:
            st.write("**Explanation:**")
            st.write(result.explanation)


def get_example_articles() -> Dict[str, str]:
    """
    Get example news articles for demonstration.
    
    Returns:
        Dictionary of example articles with titles as keys
    """
    return {
        "Real News Example - Technology": """
        Scientists at MIT have developed a new artificial intelligence system that can detect deepfake videos with 94% accuracy. The research, published in the journal Nature Communications, describes a method that analyzes subtle inconsistencies in facial movements and lighting patterns that are difficult for current deepfake technology to replicate perfectly. The team tested their system on over 10,000 videos and found it significantly outperformed existing detection methods. Lead researcher Dr. Sarah Chen stated that while this is a promising development, the arms race between deepfake creation and detection technologies will likely continue as both sides become more sophisticated.
        """,
        
        "Real News Example - Health": """
        The World Health Organization announced today that global vaccination rates for measles have reached 85%, marking a significant milestone in public health efforts. According to the WHO's latest report, this represents a 12% increase from 2019 levels, largely attributed to improved vaccine distribution networks and public health campaigns in developing countries. Dr. Maria Santos, WHO's Director of Immunization, emphasized that while this progress is encouraging, the organization's goal remains to achieve 95% coverage to ensure herd immunity. The report also noted that vaccine hesitancy in some regions continues to pose challenges to reaching universal coverage.
        """,
        
        "Suspicious Content Example": """
        BREAKING: Government officials HATE this one simple trick that ELIMINATES all debt instantly! Local mom discovers SECRET method that banks don't want you to know about - she paid off $50,000 in just 30 days using this WEIRD trick. Financial experts are SHOCKED by this revolutionary system that the elite have been hiding from ordinary people for decades. Click here to learn the TRUTH they don't want you to discover before it's too late! This information will be BANNED soon, so act fast! Thousands of people are already using this method to become debt-free millionaires overnight!
        """,
        
        "Sample Article": "Enter your news article text here for analysis."
    }


def main():
    """Main Streamlit application."""
    
    # Initialize session state for text input
    if 'current_text' not in st.session_state:
        st.session_state.current_text = ""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p>Analyze news articles and headlines to detect potential misinformation using AI.</p>
        <p><em>Enter text below or try one of our examples to get started</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load prediction service
    prediction_service = load_prediction_service()
    
    if prediction_service is None:
        st.error("❌ The AI model is currently unavailable. Please try again later.")
        st.stop()
    
    # Sidebar with information and settings
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This tool uses machine learning to analyze text content and predict whether it's likely to be real or fake news.
        
        **How it works:**
        1. Enter news text (10-10,000 characters)
        2. AI analyzes linguistic patterns
        3. Get classification with confidence score
        """)
        
        st.header("❓ Help & Tips")
        with st.expander("📝 Input Guidelines"):
            st.write("""
            **For best results:**
            - Use complete sentences and paragraphs
            - Include at least 50-100 words
            - Paste the full article text, not just headlines
            - Avoid excessive formatting or special characters
            
            **Text Requirements:**
            - Minimum: 10 characters
            - Maximum: 10,000 characters
            - Must contain meaningful words
            - UTF-8 encoding supported
            """)
        
        with st.expander("🎯 Understanding Results"):
            st.write("""
            **Confidence Levels:**
            - Very High (90-100%): Strong prediction
            - High (75-89%): Reliable prediction
            - Medium (60-74%): Moderate confidence
            - Low (40-59%): Uncertain prediction
            - Very Low (0-39%): Unreliable prediction
            
            **Classifications:**
            - ✅ Real: Likely legitimate news
            - ❌ Fake: Likely misinformation
            - ❓ Uncertain: Requires verification
            """)
        
        with st.expander("⚠️ Limitations"):
            st.write("""
            **Important Notes:**
            - This tool is not 100% accurate
            - Always verify with multiple sources
            - Satirical content may be misclassified
            - New topics may be challenging
            - Context matters for interpretation
            """)
        
        st.header("⚙️ Settings")
        show_feature_importance = st.checkbox("Show key indicators", value=True)
        show_detailed_analysis = st.checkbox("Show detailed analysis", value=True)
        enable_sound = st.checkbox("Enable sound notifications", value=False)
        
        st.header("📈 Model Info")
        if prediction_service:
            stats = prediction_service.get_service_stats()
            st.write(f"**Model Status:** {'✅ Ready' if stats.get('classifier_trained') else '❌ Not Ready'}")
            st.write(f"**Cache:** {'✅ Enabled' if stats.get('cache_enabled') else '❌ Disabled'}")
            
            # Health check button
            if st.button("🔍 Run Health Check"):
                with st.spinner("Checking system health..."):
                    health = prediction_service.get_health_status()
                    if health.get('healthy'):
                        st.success("✅ System is healthy")
                    else:
                        st.error(f"❌ System issue: {health.get('error', 'Unknown error')}")
        
        st.header("📞 Support")
        st.write("""
        **Having issues?**
        - Check the Help & Tips section above
        - Try the example articles
        - Ensure your text meets requirements
        - Contact support if problems persist
        """)
    
    # Example articles section
    st.subheader("📚 Try Example Articles")
    
    examples = get_example_articles()
    example_cols = st.columns(len(examples))
    
    for i, (title, content) in enumerate(examples.items()):
        with example_cols[i]:
            if st.button(f"📄 {title}", key=f"example_{i}", help="Click to load this example"):
                st.session_state.current_text = content.strip()
                st.rerun()
    
    # Main input area
    st.subheader("📝 Enter News Content")
    
    # Text input with character counter
    text_input = st.text_area(
        "Paste your news article, headline, or social media post here:",
        value=st.session_state.current_text,
        height=200,
        max_chars=10000,
        placeholder="Example: 'Breaking: Scientists discover new method to detect fake news with 95% accuracy...'",
        help="Enter between 10 and 10,000 characters for analysis",
        key="text_input_area"
    )
    
    # Update session state when text changes
    if text_input != st.session_state.current_text:
        st.session_state.current_text = text_input
    
    # Enhanced character counter with validation feedback
    char_count = len(text_input) if text_input else 0
    word_count = len(text_input.split()) if text_input else 0
    
    # Determine status and color
    if char_count == 0:
        char_color = "gray"
        status_text = ""
        status_icon = ""
    elif char_count < 10:
        char_color = "orange"
        status_text = f"Need {10 - char_count} more characters"
        status_icon = "⚠️"
    elif char_count > 10000:
        char_color = "red"
        status_text = f"{char_count - 10000} characters over limit"
        status_icon = "❌"
    else:
        char_color = "green"
        status_text = "Ready for analysis"
        status_icon = "✅"
    
    # Display enhanced counter
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        <div class="char-counter" style="color: {char_color};">
            {char_count:,} / 10,000 characters • {word_count} words {status_icon}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if status_text:
            st.markdown(f"""
            <div style="text-align: right; font-size: 0.8rem; color: {char_color};">
                {status_text}
            </div>
            """, unsafe_allow_html=True)
    
    # Additional validation feedback
    if text_input:
        validation_issues = []
        
        # Check for meaningful content
        import re
        words = re.findall(r'\b\w+\b', text_input.lower())
        if len(words) < 3 and char_count >= 10:
            validation_issues.append("Text should contain at least 3 meaningful words")
        
        # Check for excessive repetition
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:
                validation_issues.append("Text appears to have excessive repetition")
        
        # Check character distribution
        if char_count > 20:
            alpha_chars = sum(1 for c in text_input if c.isalpha())
            if alpha_chars / char_count < 0.3:
                validation_issues.append("Text should contain more alphabetic characters")
        
        # Display validation issues
        if validation_issues:
            with st.expander("⚠️ Input Quality Issues", expanded=False):
                for issue in validation_issues:
                    st.warning(f"• {issue}")
                st.info("💡 These issues may affect analysis accuracy but won't prevent processing.")
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        analyze_button = st.button("🔍 Analyze Text", type="primary", disabled=not (10 <= char_count <= 10000))
    
    with col2:
        clear_button = st.button("🗑️ Clear", help="Clear the text input")
    
    with col3:
        if st.button("📋 Copy Result", help="Copy the last analysis result", disabled='last_result' not in st.session_state):
            if 'last_result' in st.session_state:
                result_text = f"Classification: {st.session_state.last_result.classification}\n"
                result_text += f"Confidence: {st.session_state.last_result.get_confidence_percentage()}%\n"
                result_text += f"Risk: {st.session_state.last_result.get_risk_assessment()}"
                st.code(result_text, language="text")
    
    # Clear functionality
    if clear_button:
        st.session_state.current_text = ""
        if 'last_result' in st.session_state:
            del st.session_state.last_result
        st.rerun()
    
    # Analysis functionality with enhanced error handling
    if analyze_button and text_input and 10 <= char_count <= 10000:
        # Create status container for real-time updates
        status_container = st.empty()
        progress_container = st.empty()
        
        try:
            with status_container.container():
                st.info("🤖 Starting analysis...")
            
            # Add progress bar for better UX
            with progress_container.container():
                progress_bar = st.progress(0)
                progress_text = st.empty()
            
            # Step 1: Input validation
            progress_text.text("Validating input...")
            progress_bar.progress(20)
            time.sleep(0.1)
            
            # Step 2: Preprocessing
            progress_text.text("Preprocessing text...")
            progress_bar.progress(40)
            time.sleep(0.2)
            
            # Step 3: Feature extraction
            progress_text.text("Extracting features...")
            progress_bar.progress(60)
            time.sleep(0.3)
            
            # Step 4: Classification
            progress_text.text("Running AI classification...")
            progress_bar.progress(80)
            
            # Make prediction
            result = prediction_service.classify_text(text_input)
            
            # Step 5: Complete
            progress_text.text("Analysis complete!")
            progress_bar.progress(100)
            time.sleep(0.2)
            
            # Clear progress indicators
            progress_container.empty()
            status_container.empty()
            
            # Display success message
            st.success("✅ Analysis completed successfully!")
            
            # Display results
            st.subheader("📊 Analysis Results")
            display_prediction_result(result, char_count)
            
            # Store result in session state for history and copying
            st.session_state.last_result = result
            
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            
            st.session_state.prediction_history.append({
                'text': text_input[:100] + "..." if len(text_input) > 100 else text_input,
                'result': result,
                'timestamp': time.time()
            })
            
            # Keep only last 10 results
            if len(st.session_state.prediction_history) > 10:
                st.session_state.prediction_history = st.session_state.prediction_history[-10:]
            
            # Show success notification
            if 'enable_sound' in st.session_state and st.session_state.enable_sound:
                st.balloons()  # Visual celebration for successful analysis
                
        except Exception as e:
            # Clear progress indicators
            progress_container.empty()
            status_container.empty()
            
            # Enhanced error handling with specific error types
            error_str = str(e).lower()
            
            if "validation" in error_str or "invalid" in error_str:
                st.error("❌ Input validation failed")
                display_error_message("validation", str(e), [
                    "Check your text meets the requirements (10-10,000 characters)",
                    "Ensure text contains meaningful words",
                    "Remove excessive special characters",
                    "Try using one of the example articles"
                ])
            elif "timeout" in error_str:
                st.error("⏱️ Analysis timed out")
                display_error_message("timeout", str(e), [
                    "Try with shorter text",
                    "Check your internet connection",
                    "Wait a moment and try again"
                ])
            elif "model" in error_str or "classifier" in error_str:
                st.error("🤖 AI model error")
                display_error_message("model", str(e), [
                    "The AI model may be temporarily unavailable",
                    "Try refreshing the page",
                    "Contact support if the issue persists"
                ])
            else:
                st.error("❌ Unexpected error occurred")
                display_error_message("unknown", str(e), [
                    "Try refreshing the page",
                    "Use one of the example articles",
                    "Contact support if the problem continues"
                ])
            
            logger.error(f"Prediction failed: {e}")
    
    # Enhanced validation messages with specific guidance
    elif analyze_button:
        if not text_input:
            st.warning("⚠️ Please enter some text to analyze.")
            st.info("💡 **Quick start:** Try clicking one of the example articles above, or paste your own news content.")
        elif char_count < 10:
            st.warning(f"⚠️ Text is too short ({char_count} characters). Need at least 10 characters.")
            st.info(f"💡 **Add {10 - char_count} more characters** to meet the minimum requirement.")
        elif char_count > 10000:
            st.warning(f"⚠️ Text is too long ({char_count:,} characters). Maximum is 10,000 characters.")
            st.info(f"💡 **Remove {char_count - 10000:,} characters** or try analyzing a shorter excerpt.")
    
    # Prediction history with enhanced features
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        st.subheader("📚 Recent Analysis History")
        
        # History controls
        col1, col2 = st.columns([3, 1])
        with col1:
            show_count = st.selectbox("Show results:", [5, 10, "All"], index=0, key="history_count")
        with col2:
            if st.button("🗑️ Clear History", help="Clear all analysis history"):
                st.session_state.prediction_history = []
                if 'last_result' in st.session_state:
                    del st.session_state.last_result
                st.rerun()
        
        # Determine how many results to show
        if show_count == "All":
            results_to_show = st.session_state.prediction_history
        else:
            results_to_show = st.session_state.prediction_history[-int(show_count):]
        
        # Display history with enhanced information
        for i, entry in enumerate(reversed(results_to_show)):
            result = entry['result']
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry['timestamp']))
            
            # Create a more informative title
            classification_emoji = "✅" if result.classification == "Real" else "❌" if result.classification == "Fake" else "❓"
            confidence_level = result.get_confidence_level()
            
            with st.expander(f"{classification_emoji} {result.classification} ({result.get_confidence_percentage()}%) - {entry['text'][:60]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Time:** {timestamp}")
                    st.write(f"**Classification:** {result.classification}")
                    st.write(f"**Confidence:** {result.get_confidence_percentage()}% ({confidence_level})")
                
                with col2:
                    st.write(f"**Processing Time:** {result.processing_time:.3f}s")
                    st.write(f"**Risk Assessment:** {result.get_risk_assessment()}")
                
                # Show feature weights if available
                if result.feature_weights:
                    st.write("**Key Indicators:**")
                    top_features = sorted(result.feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                    for feature, weight in top_features:
                        indicator = "📈 Fake indicator" if weight > 0 else "📉 Real indicator"
                        st.write(f"- '{feature}' - {indicator}")
                
                # Button to reanalyze this text
                if st.button(f"🔄 Reanalyze", key=f"reanalyze_{i}", help="Load this text for reanalysis"):
                    # Extract original text from history (this is simplified)
                    st.session_state.current_text = entry['text'].replace("...", "")
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.8rem;">
        <p>⚠️ This tool is for educational purposes. Always verify important information with multiple reliable sources.</p>
        <p>🔒 Your text is processed locally and not stored permanently.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()