import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from core.vector_store import build_vector_store, load_vector_store, get_retriever
from langchain_core.output_parsers import StrOutputParser

def get_llm():
    """Use Gemini for RAG answers so Mistral rate limits do not stop chat."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set in the .env file")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        api_key=api_key,
        temperature=0.3,
    )

def format_docs(docs) -> str:
    """Convert retriever output into prompt context.

    Most LangChain versions return ``list[Document]`` here, but some runnable
    compositions yield batches as ``list[list[Document]]``. Normalize both
    shapes before accessing ``page_content``.
    """
    flattened_docs = []
    for item in docs:
        if isinstance(item, (list, tuple)):
            flattened_docs.extend(item)
        else:
            flattened_docs.append(item)

    return "\n\n".join(doc.page_content for doc in flattened_docs)

def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store,k=4)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [(
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""
        ),
        ("human","{question}")]
    )

    # full LCEL RAG pipeline
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        } 
        | prompt | llm | StrOutputParser()
    )

    return rag_chain

def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever()

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""
        ),
        ("human", "{question}")
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt | llm | StrOutputParser()
    )

    return rag_chain

def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    try:
        answer = rag_chain.invoke(question)
    except Exception as error:
        # Preserve the CLI session and give the user a concise, useful error.
        answer = f"Unable to answer this question right now: {error}"
    print(f"answer: {answer}")
    return answer
