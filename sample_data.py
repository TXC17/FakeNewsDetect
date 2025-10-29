#!/usr/bin/env python3
"""
High-quality sample data for testing and validation.
Contains realistic examples of real and fake news for model validation.
"""

# Real news samples - high quality, factual content
REAL_NEWS_SAMPLES = [
    {
        'title': 'Federal Reserve Announces Interest Rate Decision',
        'content': 'The Federal Reserve announced today that it has decided to maintain the current interest rate at 5.25%. According to Fed Chair Jerome Powell, this decision reflects the committee\'s assessment of current economic conditions and inflation trends. The move was widely expected by economists and financial markets. Powell stated that the committee will continue to monitor economic data closely.',
        'source': 'reuters',
        'label': 0
    },
    {
        'title': 'New Study Shows Benefits of Exercise for Mental Health',
        'content': 'A comprehensive study published in the Journal of Health Psychology has revealed new insights into the relationship between physical exercise and mental well-being. The research, conducted by scientists at Harvard Medical School, analyzed data from 15,000 participants over three years. The findings suggest that regular exercise significantly reduces symptoms of depression and anxiety.',
        'source': 'medical_journal',
        'label': 0
    },
    {
        'title': 'Climate Summit Reaches Agreement on Emissions Targets',
        'content': 'World leaders at the international climate summit have reached a consensus on new emissions reduction targets. The agreement, signed by representatives from 195 countries, commits to reducing global carbon emissions by 45% by 2030. Environmental scientists have praised the agreement as a significant step forward in addressing climate change.',
        'source': 'environmental_news',
        'label': 0
    }
]

# Fake news samples - typical patterns of misinformation
FAKE_NEWS_SAMPLES = [
    {
        'title': 'SHOCKING: Government Hiding MASSIVE Secret That Will Change Everything!',
        'content': 'You won\'t believe what they don\'t want you to know! This incredible discovery will blow your mind and change everything you thought you knew. Mainstream media is covering this up, but we have the TRUTH! Government officials HATE this one simple trick that exposes their lies. Share this before it gets BANNED!',
        'source': 'conspiracy_site',
        'label': 1
    },
    {
        'title': 'DOCTORS HATE HIM: Local Man Discovers Miracle Cure That AMAZES Scientists',
        'content': 'A local resident has discovered a simple method that cures all diseases instantly! This amazing discovery has been kept secret by big pharma for decades, but now the truth is finally revealed. Thousands of people are already using this method with INCREDIBLE results. Sarah from Texas cured her diabetes overnight! Don\'t let them hide this from you any longer!',
        'source': 'clickbait_health',
        'label': 1
    },
    {
        'title': 'BREAKING: Aliens Land in Major City, Government Covers Up Evidence',
        'content': 'URGENT UPDATE: Multiple witnesses report alien spacecraft landing in downtown area last night. Government agents immediately arrived to suppress all evidence and silence witnesses. Mainstream media refuses to report this SHOCKING truth! The deep state doesn\'t want you to know about this MASSIVE cover-up. Wake up, people!',
        'source': 'ufo_conspiracy',
        'label': 1
    }
]

def get_sample_data():
    """Get all sample data combined."""
    return REAL_NEWS_SAMPLES + FAKE_NEWS_SAMPLES

def get_real_samples():
    """Get only real news samples."""
    return REAL_NEWS_SAMPLES

def get_fake_samples():
    """Get only fake news samples."""
    return FAKE_NEWS_SAMPLES