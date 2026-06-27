import streamlit as st
from dotenv import load_dotenv
from src.research_assistant import AIResearchAssistant
import uuid

load_dotenv()

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍"
)

st.title("🔍 AI Research Assistant")

# Create session_id
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Create assistant once
if "assistant" not in st.session_state:
    st.session_state.assistant = AIResearchAssistant()

assistant = st.session_state.assistant

# Track document status
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

# Track chunk count
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# Load document
if st.button("Load Document"):

    documents = assistant.load_document(
        "future_of_ai.txt"
    )

    chunk_count = assistant.add_documents(
        documents
    )

    st.session_state.document_loaded = True
    st.session_state.chunk_count = chunk_count

# Show status permanently
if st.session_state.document_loaded:

    st.success(
        f"Document loaded successfully. "
        f"{st.session_state.chunk_count} chunks stored."
    )

st.divider()

# Question input
query = st.text_input(
    "Ask a question about the document"
)

if st.button("Research"):

    if not st.session_state.document_loaded:
        st.warning("Please load the document first.")

    elif not query:
        st.warning("Please enter a question.")

    else:

        with st.spinner("Researching..."):

            # Retrieve documents for debugging
            docs = assistant.retriver(query)

            # Generate response
            response = assistant.research(
                query,
                session_id=st.session_state.session_id
            )

        # Debug View
        with st.expander("Retrieved Context"):

            st.write(f"Retrieved {len(docs)} documents")

            for i, doc in enumerate(docs, start=1):

                st.markdown(f"### Chunk {i}")

                st.write(doc.page_content)

                if hasattr(doc, "metadata"):
                    st.caption(f"Metadata: {doc.metadata}")

                st.divider()

        st.subheader("Answer")
        st.write(response.answer)

        st.subheader("Confidence")
        st.write(response.confidence)

        if response.key_quotes:
            st.subheader("Key Quotes")

            for quote in response.key_quotes:
                st.write(f"• {quote}")

        if response.follow_up_questions:
            st.subheader("Follow-up Questions")

            for question in response.follow_up_questions:
                st.write(f"• {question}")