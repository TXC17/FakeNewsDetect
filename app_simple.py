#!/usr/bin/env python3
"""
Simplified Streamlit app for testing - minimal version to debug loading issues.
"""
import streamlit as st
import time
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="Fake News Detector - Simple",
    page_icon="🔍",
    layout="wide"
)

def simple_prediction(text):
    """Simple mock prediction for testing."""
    # Basic heuristics for demo
    suspicious_words = ['shocking', 'breaking', 'urgent', 'exclusive', 'secret', 'exposed']
    
    text_lower = text.lower()
    suspicious_count = sum(1 for word in suspicious_words if word in text_lower)
    
    # Simple scoring
    if suspicious_count >= 2:
        return "Fake", 0.75 + (suspicious_count * 0.05)
    elif len(text) < 50:
        return "Uncertain", 0.45
    else:
        return "Real", 0.70 + (len(text) / 1000 * 0.1)

def main():
    """Main application."""
    st.title("🔍 Fake News Detector (Simple Version)")
    st.write("This is a simplified version for testing. Enter text below to analyze:")
    
    # Text input
    user_text = st.text_area(
        "Enter news article text:",
        height=200,
        placeholder="Paste your news article text here..."
    )
    
    if st.button("Analyze Text", type="primary"):
        if user_text.strip():
            with st.spinner("Analyzing..."):
                time.sleep(1)  # Simulate processing
                classification, confidence = simple_prediction(user_text)
                
                # Display results
                if classification == "Real":
                    st.success(f"✅ **{classification.upper()} NEWS** ({confidence:.0%} confidence)")
                elif classification == "Fake":
                    st.error(f"❌ **{classification.upper()} NEWS** ({confidence:.0%} confidence)")
                else:
                    st.warning(f"❓ **{classification.upper()}** ({confidence:.0%} confidence)")
                
                # Show analysis details
                with st.expander("Analysis Details"):
                    st.write(f"**Text Length:** {len(user_text)} characters")
                    st.write(f"**Word Count:** {len(user_text.split())} words")
                    
                    suspicious_words = ['shocking', 'breaking', 'urgent', 'exclusive', 'secret', 'exposed']
                    found_words = [word for word in suspicious_words if word in user_text.lower()]
                    if found_words:
                        st.write(f"**Suspicious Words Found:** {', '.join(found_words)}")
                    else:
                        st.write("**Suspicious Words Found:** None")
        else:
            st.warning("Please enter some text to analyze.")
    
    # Sample texts
    st.subheader("📝 Try These Examples")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Real News Example"):
            st.session_state.sample_text = """The Federal Reserve announced today that it has decided to maintain the current interest rate at 5.25%. According to Fed Chair Jerome Powell, this decision reflects the committee's assessment of current economic conditions and inflation trends. The move was widely expected by economists and financial markets."""
    
    with col2:
        if st.button("Fake News Example"):
            st.session_state.sample_text = """SHOCKING: Government Hiding MASSIVE Secret That Will Change Everything! You won't believe what they don't want you to know! This incredible discovery will blow your mind and change everything you thought you knew. Share this before it gets BANNED!"""
    
    # Display sample text if selected
    if hasattr(st.session_state, 'sample_text'):
        st.text_area("Sample Text (copy to analyze above):", value=st.session_state.sample_text, height=100)

if __name__ == "__main__":
    main()