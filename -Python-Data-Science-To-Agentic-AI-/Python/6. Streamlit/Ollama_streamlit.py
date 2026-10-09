import streamlit as st
import ollama

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "deepseek-r1:1.5b"
MAX_HISTORY = 20  # only the last N messages are sent to the model (faster replies)

st.set_page_config(
    page_title="Gaurav Navghare | Ollama Chat",
    page_icon="💬",
    layout="wide",
)

client = ollama.Client(host=OLLAMA_HOST)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&display=swap');

:root {
    --ink: #0f1620;
    --panel: #172230;
    --panel-2: #1e2c3d;
    --line: #2a3a4f;
    --text: #e8eef5;
    --muted: #8fa3b8;
    --accent: #f2b45a;
}

html, body, [class*="css"], .stApp {
    font-family: 'Instrument Sans', system-ui, sans-serif;
}
.stApp { background: var(--ink); color: var(--text); }

/* hide default Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.block-container { max-width: 860px; padding-top: 2rem; padding-bottom: 6rem; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] * { color: var(--text); }

/* title */
.app-title { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; }
.app-sub   { color: var(--muted); margin: 0.15rem 0 1.5rem 0; font-size: 0.95rem; }

/* welcome */
.welcome h1 { font-size: 2.4rem; font-weight: 700; letter-spacing: -0.03em; margin: 3rem 0 0.4rem 0; }
.welcome p  { color: var(--muted); font-size: 1.05rem; margin-bottom: 1.5rem; }

/* chat messages */
div[data-testid="stChatMessage"] {
    background: transparent;
    border-radius: 14px;
    padding: 0.9rem 1rem;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    background: var(--panel-2);
    border: 1px solid var(--line);
}
div[data-testid="stChatMessage"] p { line-height: 1.65; }

/* chat input */
div[data-testid="stChatInput"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
}
div[data-testid="stChatInput"]:focus-within { border-color: var(--accent); }

/* buttons */
.stButton > button {
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.7rem 1rem;
    text-align: left;
    transition: border-color 0.15s ease, background 0.15s ease;
}
.stButton > button:hover { border-color: var(--accent); background: var(--panel-2); color: var(--text); }
.stButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* reasoning expander */
details { background: var(--panel); border: 1px solid var(--line) !important; border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def list_models():
    """Return the names of models installed in Ollama."""
    try:
        names = []
        for m in ollama.Client(host=OLLAMA_HOST).list().models:
            name = getattr(m, "model", None) or (m.get("name") if isinstance(m, dict) else None)
            if name:
                names.append(name)
        return names
    except Exception:
        return []


def split_thinking(raw: str):
    """Split deepseek-r1 style output into (reasoning, answer)."""
    if "<think>" not in raw:
        return "", raw
    _, rest = raw.split("<think>", 1)
    if "</think>" in rest:
        thinking, answer = rest.split("</think>", 1)
        return thinking.strip(), answer.strip()
    return rest.strip(), ""  # still thinking


def build_messages(system_prompt: str):
    history = st.session_state.messages[-MAX_HISTORY:]
    msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    if system_prompt.strip():
        msgs.insert(0, {"role": "system", "content": system_prompt.strip()})
    return msgs


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")

    models = list_models()
    if models:
        index = models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0
        model = st.selectbox("Model", models, index=index)
    else:
        model = st.text_input("Model", value=DEFAULT_MODEL)
        st.caption("Could not read the model list. Is Ollama running?")

    temperature = st.slider("Creativity", 0.0, 1.5, 0.7, 0.1)
    show_reasoning = st.toggle("Show model reasoning", value=True)
    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful, concise assistant.",
        height=100,
    )

    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built by Gaurav Navghare")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    '<p class="app-title">Gaurav Navghare</p>'
    '<p class="app-sub">Chat with a local model, running on your own machine.</p>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Chat history
# ----------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        if msg.get("thinking") and show_reasoning:
            with st.expander("Show reasoning"):
                st.markdown(msg["thinking"])
        st.markdown(msg["content"])

# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
prompt = st.chat_input("Message your model...")
if not prompt and st.session_state.pending:
    prompt = st.session_state.pending
    st.session_state.pending = None

# Welcome screen with suggestions (only on an empty chat)
welcome = st.empty()
if not st.session_state.messages and not prompt:
    with welcome.container():
        st.markdown(
            '<div class="welcome"><h1>What can I help with?</h1>'
            "<p>Pick a starter or type your own message below.</p></div>",
            unsafe_allow_html=True,
        )
        suggestions = [
            "Explain Python lists vs tuples with examples",
            "Give me a 4-week plan to learn data science",
            "Write a Python function to clean a CSV file",
            "What is an AI agent? Explain simply",
        ]
        cols = st.columns(2)
        for i, text in enumerate(suggestions):
            if cols[i % 2].button(text, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending = text
                st.rerun()

# ----------------------------------------------------------------------------
# Generate (streaming)
# ----------------------------------------------------------------------------
if prompt:
    welcome.empty()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        think_box = st.empty()
        answer_box = st.empty()
        answer_box.markdown("▍")

        raw = ""
        native_thinking = ""
        try:
            stream = client.chat(
                model=model,
                messages=build_messages(system_prompt),
                stream=True,
                options={"temperature": temperature},
                keep_alive="30m",  # keep the model loaded so next replies start instantly
            )
            for chunk in stream:
                raw += chunk.message.content or ""
                native_thinking += getattr(chunk.message, "thinking", None) or ""

                thinking, answer = split_thinking(raw)
                thinking = (native_thinking + thinking).strip()

                if answer:
                    if thinking:
                        think_box.empty()
                    answer_box.markdown(answer + " ▍")
                elif thinking and show_reasoning:
                    think_box.markdown(f"*Thinking...*\n\n{thinking}")

            thinking, answer = split_thinking(raw)
            thinking = (native_thinking + thinking).strip()
            answer_box.markdown(answer or "_(empty response)_")
            think_box.empty()
            if thinking and show_reasoning:
                with think_box.expander("Show reasoning"):
                    st.markdown(thinking)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "thinking": thinking}
            )

        except Exception as e:
            think_box.empty()
            answer_box.error(
                f"Could not get a response from Ollama: {e}\n\n"
                f"Check that Ollama is running (`ollama serve`) and that the model is "
                f"installed (`ollama pull {model}`)."
            )
            st.session_state.messages.pop()  # remove the unanswered user message