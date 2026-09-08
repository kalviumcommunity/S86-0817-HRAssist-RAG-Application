import os
import requests
import streamlit as st
from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================

load_dotenv()

API_URL = os.getenv(
    "RAG_API_URL",
    "http://localhost:8000"
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="HR Assist",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# Header
# ============================================================

st.title("🤖 HR Assist")
st.caption(
    "AI-powered HR assistant using Retrieval-Augmented Generation"
)

st.divider()


# ============================================================
# Session state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# Display previous messages
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📚 View Sources"):

                for source in message["sources"]:

                    st.markdown(
                        f"**Source:** "
                        f"{source.get('source', 'Unknown')}"
                    )

                    if source.get("chunk_id"):
                        st.write(
                            f"Chunk ID: "
                            f"{source['chunk_id']}"
                        )

                    if source.get("section"):
                        st.write(
                            f"Section: "
                            f"{source['section']}"
                        )

                    if source.get("page"):
                        st.write(
                            f"Page: "
                            f"{source['page']}"
                        )

                    st.divider()


# ============================================================
# API request
# ============================================================

def ask_question(question):

    response = requests.post(
        f"{API_URL}/query",
        json={
            "question": question
        },
        timeout=60
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# Chat input
# ============================================================

question = st.chat_input(
    "Ask an HR question..."
)


if question:

    # --------------------------------------------------------
    # Display user question
    # --------------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching HR documents and generating answer..."
        ):

            try:

                result = ask_question(
                    question
                )

                answer = result.get(
                    "answer",
                    "No answer returned."
                )

                sources = result.get(
                    "sources",
                    []
                )


                # ------------------------------------------------
                # Display answer
                # ------------------------------------------------

                st.markdown(answer)


                # ------------------------------------------------
                # Display sources
                # ------------------------------------------------

                if sources:

                    with st.expander(
                        "📚 View Sources",
                        expanded=True
                    ):

                        for source in sources:

                            st.markdown(
                                f"**📄 "
                                f"{source.get('source', 'Unknown source')}**"
                            )

                            if source.get("chunk_id"):

                                st.write(
                                    f"Chunk ID: "
                                    f"{source['chunk_id']}"
                                )

                            if source.get("section"):

                                st.write(
                                    f"Section: "
                                    f"{source['section']}"
                                )

                            if source.get("page"):

                                st.write(
                                    f"Page: "
                                    f"{source['page']}"
                                )

                            st.divider()

                else:

                    st.info(
                        "No supporting sources were returned."
                    )


                # ------------------------------------------------
                # Save conversation
                # ------------------------------------------------

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })


            # ----------------------------------------------------
            # Error handling
            # ----------------------------------------------------

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to the RAG backend. "
                    "Make sure the backend server is running."
                )


            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ The request took too long. "
                    "Please try again."
                )


            except requests.exceptions.HTTPError as error:

                st.error(
                    f"❌ RAG API request failed: {error}"
                )


            except Exception as error:

                st.error(
                    f"❌ Something went wrong: {error}"
                )


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("About HR Assist")

    st.write(
        """
        HR Assist uses Retrieval-Augmented Generation
        to answer questions using your HR document corpus.
        """
    )

    st.subheader("Features")

    st.write("✅ Grounded answers")
    st.write("✅ Source citations")
    st.write("✅ Semantic retrieval")
    st.write("✅ Context-based responses")
    st.write("✅ Error handling")

    st.divider()

    st.caption(
        f"Backend: {API_URL}"
    )