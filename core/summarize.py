# Import necessary modules from LangChain and Python standard library
from langchain_mistralai import ChatMistralAI              # Mistral AI model wrapper
from langchain_core.prompts import ChatPromptTemplate      # For structuring prompts
from langchain_core.output_parsers import StrOutputParser  # To parse model outputs into strings
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting long text into chunks
from langchain_core.runnables import RunnableLambda, RunnablePassthrough # For chaining operations
import os

# Function to initialize and return the Mistral LLM instance
def get_llm():
    return ChatMistralAI(
        model="ministral-3b-2512", 
        mistral_api_key=os.getenv("MISTRAL_API_KEY")  # Load API key from environment variables
    )

# Function to split a long transcript into smaller overlapping chunks
def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,   # Each chunk will have ~3000 characters
        chunk_overlap=200  # Overlap between chunks to preserve context
    )
    return splitter.split_text(transcript)

# Function to summarize the entire transcript
def summarize(transcript: str) -> str:
    llm = get_llm()

    # Prompt template for summarizing individual chunks
    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ]
    )

    # Chain: prompt → LLM → output parser
    map_chain = map_prompt | llm | StrOutputParser()

    # Split transcript into chunks
    chunks = split_transcript(transcript)

    # Summarize each chunk individually
    chunk_summarise = [map_chain.invoke({"text": chunk}) for chunk in chunks]

    # Combine all chunk summaries into one string
    combined = "\n\n".join(chunk_summarise)

    # Prompt template for combining partial summaries into a final professional summary
    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. Combine these partial summaries into one final professional meeting summary in bullet points."
            ),
            ("human", "{text}"),
        ]
    )

    # Chain: passthrough → lambda (wrap text) → prompt → LLM → output parser
    combined_chain = (
        RunnableLambda(lambda x: {"text": x}) 
        | combined_prompt 
        | llm 
        | StrOutputParser()
    )

    # Return the final combined summary
    return combined_chain.invoke(combined)

# Function to generate a short professional meeting title
def generate_title(transcript: str) -> str:
    llm = get_llm()

    # Chain: passthrough → lambda (wrap text) → prompt → LLM → output parser
    title_chain = (
        RunnableLambda(lambda x: {"text": x}) 
        | ChatPromptTemplate.from_messages([
            (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title (max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}")
        ])     
        | llm
        | StrOutputParser()
    )

    # Use only the first 2000 characters of transcript for title generation
    return title_chain.invoke(transcript[:2000])
