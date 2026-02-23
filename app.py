import streamlit as st
from data_cleaning import get_seasonal_data, format_historical_data, get_available_years
from predict import get_prediction, get_rag_prediction

# Page config
st.set_page_config(
    page_title="F1 Constructor Predictor",
    page_icon="🏎️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #E10600;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #1a1a2e;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #E10600;
    }
    .historical-box {
        background-color: #16213e;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0f3460;
    }
    .rag-badge {
        background-color: #00d26a;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🏎️ F1 Constructor Championship Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Llama 3 8B + RAG</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Settings")

# Prediction mode
st.sidebar.subheader("🧠 Prediction Mode")
use_rag = st.sidebar.toggle("Use RAG (Vector Database)", value=True, 
                            help="RAG retrieves relevant historical data from vector database for better context")

if use_rag:
    st.sidebar.success("✅ RAG Mode Active")
    st.sidebar.caption("Using ChromaDB + Sentence Transformers")
else:
    st.sidebar.info("📝 Legacy Mode")
    st.sidebar.caption("Using formatted text context")

# Get available years
available_years = get_available_years()
min_year = min(available_years)
max_year = max(available_years)

# Year range selector (only for legacy mode)
if not use_rag:
    st.sidebar.subheader("Historical Data Range")
    start_year = st.sidebar.slider("Start Year", min_year, max_year, max(min_year, max_year - 4))
    end_year = st.sidebar.slider("End Year", start_year, max_year, max_year)
else:
    start_year = max_year - 4
    end_year = max_year

# Target year
st.sidebar.subheader("Prediction Target")
target_year = st.sidebar.number_input("Predict Season", min_value=end_year + 1, max_value=2030, value=end_year + 1)

# Temperature
st.sidebar.subheader("Model Settings")
temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.8, 0.1, 
                                help="Higher = more creative, Lower = more deterministic")

# RAG query (optional)
if use_rag:
    st.sidebar.subheader("🔍 Custom Query (Optional)")
    custom_query = st.sidebar.text_input("Focus on specific aspect", 
                                          placeholder="e.g., Red Bull dominance",
                                          help="Add specific context for the prediction")
else:
    custom_query = ""

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    if use_rag:
        st.header("🗄️ Vector Database Context")
        st.caption("Retrieved from ChromaDB using semantic search")
    else:
        st.header("📊 Historical Data")
    
    # Show historical data
    try:
        seasonal = get_seasonal_data(start_year, end_year)
        historical_text = format_historical_data(seasonal, start_year, end_year)
        
        with st.expander("View Raw Data", expanded=False):
            st.dataframe(seasonal[['year', 'name', 'points', 'position', 'wins']], width='stretch')
        
        if use_rag:
            if 'rag_context' in st.session_state:
                st.text_area("Retrieved Context", st.session_state['rag_context'], height=400, disabled=True)
            else:
                st.info("Context will be shown after generating prediction")
        else:
            st.text_area("Formatted Historical Data", historical_text, height=400, disabled=True)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

with col2:
    st.header(f"🔮 {target_year} Prediction")
    
    if use_rag:
        st.caption("🧠 Using RAG with vector embeddings")
    
    # Predict button
    if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
        with st.spinner("Generating prediction with Llama 3 8B..."):
            try:
                if use_rag:
                    prediction, context = get_rag_prediction(custom_query, target_year, temperature)
                    st.session_state['prediction'] = prediction
                    st.session_state['rag_context'] = context
                else:
                    prediction, _ = get_prediction(start_year, end_year, target_year, temperature)
                    st.session_state['prediction'] = prediction
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display prediction
    if 'prediction' in st.session_state:
        st.text_area("Prediction Results", st.session_state['prediction'], height=400, disabled=True)
    else:
        st.info("Click 'Generate Prediction' to get the AI prediction")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit • Llama 3 8B via Ollama • ChromaDB Vector Store</p>
    <p>RAG: Retrieval Augmented Generation for better predictions</p>
</div>
""", unsafe_allow_html=True)
