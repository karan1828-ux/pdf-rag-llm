import streamlit as st
import tempfile
from app import PDFRAG

st.title("PDF RAG BOT")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar calculator
st.sidebar.header("Calculator")
calc_expr = st.sidebar.text_input("Enter arithmetic expression (e.g., 2+2*5):")
if st.sidebar.button("Calculate") and calc_expr:
    # Use a dummy PDF path for calculator, as it doesn't need the PDF
    rag = PDFRAG("test_sample.pdf")
    calc_result = rag.calculate(calc_expr)
    st.sidebar.markdown(f"**Result:** {calc_result}")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Sidebar joke generator
st.sidebar.markdown("""
    <div style='margin-top: 20px; padding: 16px; border-radius: 12px; background: linear-gradient(90deg, #f9d423 0%, #ff4e50 100%); color: #222; font-weight: bold; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
        <span style='font-size: 1.1em;'>🎲 Want a random joke? Click below!</span>
    </div>
""", unsafe_allow_html=True)

if st.sidebar.button("Generate Joke", key="joke_btn_sidebar"):
    rag = PDFRAG("test_sample.pdf")
    joke = rag.get_joke()
    st.sidebar.markdown(f"""
        <div style='margin-top: 10px; padding: 14px; border-radius: 10px; background: #fffbe7; color: #333; border: 1px solid #ffe082; font-size: 1.05em;'>
            <b>😂 Joke:</b> {joke}
        </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name
    st.success("PDF uploaded successfully!")
    question = st.text_input("Enter your question about the PDF:")

    if st.button("Get Answer") and question:
        rag = PDFRAG(tmp_pdf_path)
        # Restore previous chat history in memory
        rag.memory.chat_memory.messages = st.session_state.chat_history
        with st.spinner("Generating answer..."):
            answer = rag.answer_query(question)
        # Update chat history
        st.session_state.chat_history = rag.memory.chat_memory.messages
        st.markdown(f"**Answer:** {answer}")

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### Conversation History")
        for msg in st.session_state.chat_history:
            if msg.type == "human":
                st.markdown(f"**You:** {msg.content}")
            elif msg.type == "ai":
                st.markdown(f"**Assistant:** {msg.content}") 
