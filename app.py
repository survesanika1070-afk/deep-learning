import streamlit as st
from rag import ask_football_question


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="FootballAI",
    page_icon="🏆",
    layout="wide"
)


# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #777;
    font-size: 18px;
    margin-bottom: 30px;
}

.chat-box {
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================

st.markdown(
    '<div class="main-title">🏆 FootballAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your Generative AI Football Assistant</div>',
    unsafe_allow_html=True
)


# =========================
# SESSION HISTORY
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("Football AI Chatbot ⚽")

    st.write(
        "Ask questions about football"
    )

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# =========================
# DISPLAY CHAT HISTORY
# =========================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================
# USER INPUT
# =========================

question = st.chat_input(
    "Ask me a football question..."
)


if question:

    # =========================
    # SHOW USER MESSAGE
    # =========================

    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # =========================
    # GENERATE RESPONSE
    # =========================

    with st.chat_message("assistant"):

        with st.spinner(
            "🏆 FootballAI is thinking..."
        ):

            try:

                answer = ask_football_question(
                    question
                )

                st.markdown(answer)

            except Exception as e:

                answer = (
                    "Sorry, something went wrong. "
                    "Please try again."
                )

                st.error(str(e))


    # =========================
    # SAVE ASSISTANT RESPONSE
    # =========================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )