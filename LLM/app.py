# ============================================================
# AVINS AGRICULTURE AI ASSISTANT
# Built with Streamlit + Mistral API
# Modular architecture for easy future expansion
# ============================================================

import time
import streamlit as st
from mistralai import Mistral

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Agriculture AI Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Agriculture Theme
# ============================================================
st.markdown("""
<style>
    /* ---- Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
    }

    /* ---- Background ---- */
    .stApp {
        background: linear-gradient(160deg, #0f2d0f 0%, #1a3d1a 40%, #0d2b0d 100%);
        color: #e8f5e8;
    }

    /* ---- Header ---- */
    .main-header {
        background: linear-gradient(135deg, #1e5c1e, #2d7a2d);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        border: 1px solid #3a9a3a;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #a8e6a8;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #7dc87d;
        margin: 6px 0 0 0;
        font-size: 0.95rem;
    }

    /* ---- Chat messages ---- */
    .stChatMessage {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        margin-bottom: 8px !important;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #122d12, #0d200d) !important;
        border-right: 1px solid #2d5a2d !important;
    }
    [data-testid="stSidebar"] * {
        color: #b8ddb8 !important;
    }

    /* ---- Badges ---- */
    .topic-badge {
        display: inline-block;
        background: rgba(60, 140, 60, 0.25);
        border: 1px solid #3a8a3a;
        border-radius: 20px;
        padding: 4px 12px;
        margin: 3px;
        font-size: 0.78rem;
        color: #8fcc8f;
    }

    /* ---- Status box ---- */
    .status-box {
        background: rgba(255, 80, 80, 0.1);
        border: 1px solid rgba(255,80,80,0.3);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #ff9999;
        font-size: 0.92rem;
    }

    /* ---- Input box ---- */
    .stChatInput textarea {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid #3a7a3a !important;
        color: #e0f0e0 !important;
        border-radius: 12px !important;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: linear-gradient(135deg, #2d6e2d, #3a8a3a) !important;
        color: #e0f5e0 !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 8px 18px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3a8a3a, #4aaa4a) !important;
        transform: translateY(-1px) !important;
    }

    /* ---- Divider ---- */
    hr {
        border-color: #2d5a2d !important;
    }

    /* ---- Spinner ---- */
    .stSpinner > div {
        border-top-color: #4aaa4a !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# API CONFIGURATION
# ============================================================
API_KEY = "pkf7qDcAL2N2msKtVyjLdxsHGQ7GwQl9"  # Replace with your Mistral API key
client = Mistral(api_key=API_KEY)
MODEL = "mistral-small-latest"


# ============================================================
# PROMPT TEMPLATES
# ============================================================

# Classifier prompt — embedded in user message (free tier compatible)
CLASSIFIER_PROMPT_TEMPLATE = """You are a strict agriculture topic classifier.

Reply with EXACTLY one word: YES or NO.

Agriculture topics that should get YES:
- Crops, seeds, planting, harvesting, crop rotation
- Soil health, soil pH, compost, mulching
- Fertilizers (NPK, organic, chemical, bio-fertilizers)
- Irrigation, drip irrigation, water management
- Plant diseases, fungal infections, bacterial diseases
- Pest control, insects, rodents, weed management
- Farming equipment and techniques
- Livestock and poultry farming
- Aquaculture, fisheries
- Greenhouse and hydroponics
- Horticulture, floriculture, agronomy
- Post-harvest storage and food preservation
- Agricultural subsidies and schemes
- Weather and season effects on farming
- Organic farming and sustainable agriculture
- Plant nutrient deficiencies (nitrogen, phosphorus, etc.)

Topics that should get NO:
- General coding, math, science unrelated to farming
- Entertainment, movies, sports, politics
- Medical advice for humans
- Finance, travel, geography unrelated to farming
- General cooking recipes (unless about growing the ingredient)

Question to classify: {question}

Reply only YES or NO."""


# Expert agronomist prompt — embedded as user/assistant pair (free tier compatible)
EXPERT_SYSTEM_MESSAGE = """You are AgriBot, an expert AI agronomist and farming advisor with 20+ years of experience.

Your role:
- Give practical, actionable farming advice that real farmers can apply immediately
- Provide specific recommendations: fertilizer quantities, application timings, dosages
- Suggest sustainable and eco-friendly practices whenever possible
- Use simple, clear language that farmers can understand
- Structure answers with bullet points or numbered steps
- Cover soil health, crop management, pest control, irrigation, and plant diseases
- Tailor advice to small-scale and large-scale farmers
- If asked about plant deficiency symptoms, diagnose and recommend solutions

Always be warm, helpful, and farmer-friendly in your tone."""

EXPERT_ACK_MESSAGE = "Understood! I'm AgriBot, your expert agriculture assistant. I'm here to help you with all farming questions — crops, soil, pests, fertilizers, irrigation, and more. How can I help you today?"


# ============================================================
# MODULE 1: AGRICULTURE QUESTION CLASSIFIER
# ============================================================
def classify_question(question: str) -> bool:
    """
    Classifies whether a question is agriculture-related.
    Returns True if YES (agriculture), False if NO (not agriculture).
    Uses user-role message only (Mistral free tier compatible).
    """
    try:
        time.sleep(0.5)  # Prevent rate limiting

        prompt = CLASSIFIER_PROMPT_TEMPLATE.format(question=question)

        response = client.chat.complete(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.choices[0].message.content.strip().upper()
        return result.startswith("YES")

    except Exception as e:
        st.error(f"⚠️ Classifier error: {e}")
        return False


# ============================================================
# MODULE 2: EXPERT AGRICULTURE RESPONSE GENERATOR
# ============================================================
def get_expert_response(chat_history: list) -> str:
    """
    Generates an expert agriculture response using full chat history.
    Injects system behavior via user/assistant message pair (free tier compatible).
    """
    try:
        time.sleep(0.5)  # Prevent rate limiting

        # Inject expert persona at the start of every conversation
        injected_messages = [
            {"role": "user",    "content": EXPERT_SYSTEM_MESSAGE},
            {"role": "assistant", "content": EXPERT_ACK_MESSAGE},
        ] + chat_history

        response = client.chat.complete(
            model=MODEL,
            messages=injected_messages
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Response error: {e}"


# ============================================================
# MODULE 3: SESSION STATE INITIALIZER
# ============================================================
def init_session():
    """Initializes all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_questions" not in st.session_state:
        st.session_state.total_questions = 0
    if "blocked_questions" not in st.session_state:
        st.session_state.blocked_questions = 0


# ============================================================
# MODULE 4: SIDEBAR RENDERER
# ============================================================
def render_sidebar():
    """Renders the sidebar with info, topics, and controls."""
    with st.sidebar:
        st.markdown("## 🌾 AgriBot")
        st.markdown("*Your Expert Farming Assistant*")
        st.divider()

        # Stats
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions", st.session_state.total_questions)
        with col2:
            st.metric("Blocked", st.session_state.blocked_questions)

        st.divider()

        # Topics covered
        st.markdown("### 🌱 Topics I Cover")
        topics = [
            "🌾 Crops & Seeds", "🪱 Soil Health", "💧 Irrigation",
            "🧪 Fertilizers", "🐛 Pest Control", "🌿 Organic Farming",
             "🌡️ Weather & Seasons", "🏥 Plant Diseases",
            "🔬 Nutrient Deficiency"
        ]
        badges_html = "".join([f'<span class="topic-badge">{t}</span>' for t in topics])
        st.markdown(badges_html, unsafe_allow_html=True)

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_questions = 0
            st.session_state.blocked_questions = 0
            st.rerun()

        st.caption("Powered by Mistral AI · Built for Farmers")


# ============================================================
# MODULE 5: CHAT HISTORY RENDERER
# ============================================================
def render_chat_history():
    """Displays all previous chat messages."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ============================================================
# MODULE 6: MAIN CHAT HANDLER
# ============================================================
def handle_user_input(prompt: str):
    """
    Main logic: classify → block or respond.
    """
    st.session_state.total_questions += 1

    # Step 1: Classify the question
    with st.spinner("🔍 Analyzing your farming question..."):
        is_agriculture = classify_question(prompt)

    # Step 2: Block non-agriculture questions
    if not is_agriculture:
        st.session_state.blocked_questions += 1

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.markdown("""
<div class="status-box">
❌ <strong>I can only answer agriculture-related questions.</strong><br><br>
Please ask me about:<br>
🌱 Crops &amp; planting · 🪱 Soil health · 💧 Irrigation<br>
🧪 Fertilizers · 🐛 Pest control · 🌿 Organic farming<br>
🐄 Livestock · 🌾 Harvesting · 🏥 Plant diseases
</div>
""", unsafe_allow_html=True)

    # Step 3: Generate expert agriculture response
    else:
        # Save and show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and show assistant response
        with st.spinner("🌱 Growing your answer..."):
            reply = get_expert_response(st.session_state.messages)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)


# ============================================================
# MAIN APP ENTRY POINT
# ============================================================
def main():
    """Main function — renders the full app."""

    # Initialize session
    init_session()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌾 Agriculture AI Assistant</h1>
        <p>Expert farming advice for crops, soil, pests, fertilizers, irrigation & more</p>
    </div>
    """, unsafe_allow_html=True)

    # Render sidebar
    render_sidebar()

    # Welcome message for new sessions
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("""
👋 **Welcome! I'm AgriBot — your expert agriculture assistant.**

I can help you with:
- 🌾 **Crop management** — planting, growth stages, harvesting
- 🪱 **Soil health** — pH, nutrients, compost, mulching
- 🧪 **Fertilizers** — NPK ratios, dosages, application timing
- 🐛 **Pest & disease control** — identification and treatment
- 💧 **Irrigation** — drip, sprinkler, water scheduling
- 🌿 **Organic farming** — sustainable practices
- 🔬 **Nutrient deficiencies** — diagnosis and correction

Ask me anything about farming! 🚜
            """)

    # Render chat history
    render_chat_history()

    # Chat input
    prompt = st.chat_input("Ask your farming question here... 🌱")

    # Handle input
    if prompt:
        handle_user_input(prompt)


# ============================================================
# RUN APP
# ============================================================
if __name__ == "__main__":
    main()