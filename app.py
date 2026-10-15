
import streamlit as st
from groq import Groq

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="BhashaAI | Local Language Q&A",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom Styling ----------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #183B56, #2979A8);
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }

    .hero p {
        color: #EAF5FC;
        font-size: 16px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 650;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 600;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session History ----------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------- Language Options ----------
languages = [
    "English", "Hindi", "Marathi", "Gujarati",
    "Bengali", "Tamil", "Telugu", "Kannada"
]

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🌐 BhashaAI")
    st.caption("Local Language Question & Answer System")
    st.divider()

    st.subheader("Language Settings")

    question_language = st.selectbox(
        "Question language",
        languages,
        index=0
    )

    answer_language = st.selectbox(
        "Answer language",
        languages,
        index=2
    )

    st.divider()

    st.metric(
        "Questions in this session",
        len(st.session_state["history"])
    )

    st.caption(
        "Choose a language for your question and a language "
        "for the translated question and answer."
    )

# ---------- Main Header ----------
st.markdown("""
<div class="hero">
    <h1>🌏 Ask in Your Language</h1>
    <p>
        Understand information across languages.
        Enter a question, translate it, and receive an AI-generated answer.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------- Question Input ----------
st.markdown(
    '<div class="section-title">✍️ Ask a Question</div>',
    unsafe_allow_html=True
)

question = st.text_area(
    "Your question",
    placeholder="Type your question here...",
    height=130,
    label_visibility="collapsed"
)

generate = st.button(
    "🌐 Translate Question & Get Answer",
    type="primary",
    use_container_width=True
)

# ---------- Answer Generation ----------
if generate:
    if not question.strip():
        st.warning("Please enter a question before continuing.")

    else:
        try:
            client = Groq(
                api_key=st.secrets["GROQ_API_KEY"]
            )

            with st.spinner(
                "Translating your question and generating an answer..."
            ):
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": f"""
You are a helpful multilingual question-answering assistant.

The original question is in {question_language}.
Translate it accurately into {answer_language}, then answer it
in {answer_language}.

Use exactly this format:

Translated Question:
[Translated question]

Answer:
[Clear, helpful answer]

Do not change the meaning of the original question.
If you are unsure about an answer, say so in {answer_language}.
"""
                        },
                        {
                            "role": "user",
                            "content": question
                        }
                    ],
                    temperature=0.3
                )

            result = response.choices[0].message.content

            # Store successful result in session history
            st.session_state["history"].append({
                "question": question,
                "question_language": question_language,
                "answer_language": answer_language,
                "result": result
            })

            st.session_state["latest_result"] = result
            st.session_state["latest_question"] = question

        except Exception as error:
            st.error("Unable to generate an answer.")
            st.caption(f"Technical details: {error}")

# ---------- Latest Result ----------
if "latest_result" in st.session_state:
    st.divider()
    st.subheader("✨ Generated Result")

    st.caption(
        f"From {st.session_state['history'][-1]['question_language']} "
        f"to {st.session_state['history'][-1]['answer_language']}"
    )

    st.markdown(st.session_state["latest_result"])

    st.download_button(
        label="⬇️ Download Answer",
        data=st.session_state["latest_result"],
        file_name="language_answer.txt",
        mime="text/plain",
        use_container_width=True
    )

# ---------- History ----------
st.divider()
st.subheader("📚 Question & Answer History")

history = st.session_state["history"]

if history:
    top_left, top_right = st.columns([3, 1])

    with top_left:
        st.caption(
            f"You have asked {len(history)} question(s) in this session."
        )

    with top_right:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state["history"] = []
            st.session_state.pop("latest_result", None)
            st.session_state.pop("latest_question", None)
            st.rerun()

    for number, item in enumerate(reversed(history), start=1):
        with st.expander(f"Question {number}: {item['question'][:65]}"):
            st.markdown("**Original question**")
            st.write(item["question"])

            st.caption(
                f"Question: {item['question_language']} | "
                f"Answer: {item['answer_language']}"
            )

            st.markdown("**Translated question and answer**")
            st.markdown(item["result"])

else:
    st.info(
        "Your history is empty. Ask your first question to see it here."
    )

# ---------- Footer ----------
st.divider()
st.caption(
    "BhashaAI · B.Sc. Data Science Mini Project · "
    "Powered by Streamlit and Groq"
)
