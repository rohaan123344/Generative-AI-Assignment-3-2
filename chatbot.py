
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Roam Chatbot",
    page_icon="🤖",
    layout="wide"
)

load_dotenv()

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #edf4ff, #f3f0ff, #e9faff);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #172554, #312e81);
}

[data-testid="stSidebar"] * {
    color: white;
}

.main-title {
    background: linear-gradient(100deg, #4338ca, #2563eb, #0891b2);
    padding: 30px;
    border-radius: 0px 0px 25px 25px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 12px 30px rgba(37, 99, 235, 0.2);
}

.main-title h1 {
    color: white;
    font-size: 38px;
}

.main-title p {
    color: white;
    font-size: 16px;
}

div.stButton > button {
    background: linear-gradient(90deg, #4f46e5, #0891b2);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 18px;
    font-weight: bold;
}

div.stButton > button:hover {
    border: 1px solid #ffffff;
    color: white;
}

[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
}

[data-testid="stChatMessage"] {
    background: white;
    border-radius: 15px;
    padding: 12px;
    margin-bottom: 12px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.title("⚙️ Chatbot Controls")

    st.write("Personalise how your AI assistant responds.")

    api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Enter your Groq API key."
    )

    st.markdown("---")

    st.subheader("🤖 Choose AI Model")

    model_options = {
        "GPT-OSS 20B - Fast": "openai/gpt-oss-20b",
        "GPT-OSS 120B - Powerful": "openai/gpt-oss-120b"
    }

    selected_model = st.selectbox(
        "Select Model",
        list(model_options.keys())
    )

    model_name = model_options[selected_model]

    st.markdown("---")

    st.subheader("🎛️ Model Settings")

    # IMPORTANT: Define temperature BEFORE using it in ChatGroq
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower values give more focused answers."
    )

    max_tokens = st.slider(
        "Maximum Response Length",
        min_value=256,
        max_value=4096,
        value=1024,
        step=256
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.info(
        "💡 Quick Tip\n\n"
        "Ask clear questions and include details "
        "when you need a more useful answer."
    )


# ---------------- MAIN HEADER ----------------
st.markdown("""
<div class="main-title">
    <h1>🤖 Aicademy 360</h1>
    <p>Your friendly AI classroom assistant · Ask questions,
    explore ideas, and learn something new.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💬 Total Messages", len(st.session_state.messages))

with col2:
    st.metric("🤖 Selected Model", selected_model)

with col3:
    st.metric("🌡️ Temperature", temperature)


# ---------------- STARTER QUESTIONS ----------------
st.subheader("✨ Try asking something")

col1, col2, col3 = st.columns(3)

starter_question = None

with col1:
    if st.button("📚 Explain Python", use_container_width=True):
        starter_question = "Explain Python in simple words."

with col2:
    if st.button("🧠 What is AI?", use_container_width=True):
        starter_question = "What is Artificial Intelligence?"

with col3:
    if st.button("💻 Give coding tips", use_container_width=True):
        starter_question = "Give me some useful coding tips."


# ---------------- DISPLAY CHAT HISTORY ----------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------- USER INPUT ----------------
user_input = st.chat_input("Ask Aicademy 360 anything...")


# Use either a typed question or a starter question
if starter_question:
    prompt = starter_question
else:
    prompt = user_input


# ---------------- GENERATE AI RESPONSE ----------------
if prompt:

    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")

    else:

        # Save and display user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        try:

            # Create Groq model
            llm = ChatGroq(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Prepare conversation history
            chat_history = [
                SystemMessage(
                    content=(
                        "You are Aicademy 360, a friendly AI classroom "
                        "assistant. Explain things clearly using simple "
                        "words. Give helpful examples when needed."
                    )
                )
            ]

            for message in st.session_state.messages[:-1]:

                if message["role"] == "user":
                    chat_history.append(
                        HumanMessage(content=message["content"])
                    )

                elif message["role"] == "assistant":
                    chat_history.append(
                        AIMessage(content=message["content"])
                    )

            # Add current question
            chat_history.append(
                HumanMessage(content=prompt)
            )

            # Get AI response
            with st.chat_message("assistant"):

                with st.spinner("Aicademy 360 is thinking..."):

                    response = llm.invoke(chat_history)

                    answer = response.content

                    st.markdown(answer)

            # Save assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            st.rerun()

        except Exception as e:

            st.error(f"Something went wrong: {e}")

            # Remove the unanswered user message
            if (
                st.session_state.messages
                and st.session_state.messages[-1]["role"] == "user"
            ):
                st.session_state.messages.pop()


# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown(
    "<center>✨ Built with Streamlit, LangChain and Groq | Aicademy 360</center>",
    unsafe_allow_html=True
)
