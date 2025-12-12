import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

from pydantic import BaseModel, Field 

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.tools import tool

from langchain.agents import create_agent

from langchain_google_genai import ChatGoogleGenerativeAI 

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

from tools.calculator import financial_calculator

@tool
def calculator_tool(operation: str, values: List[float]) -> str:
    """
    A dedicated, precise financial and mathematical calculator. 
    Use for operations like 'percentage_change', 'ratio', 'add', 'subtract', 'multiply', 'divide'.
    Example: calculator_tool(operation='percentage_change', values=[1000.0, 1500.0])
    """
    return financial_calculator(operation, values)

class RetrievalInput(BaseModel):
    """Input schema for the financial document retrieval tool."""
    query: str = Field(description="The factual question to search for (e.g., 'Net Profit figure for Q3 2024').")
    year: int = Field(description="The specific year to filter by (e.g., 2024). Use 0 or -1 if the year is unknown.")
    quarter: str = Field(description="The specific quarter to filter by (e.g., 'Q3', 'Q4'). Use 'Unknown' if the quarter is not specified.")
    report_type: str = Field(description="The type of document to search (e.g., 'Quarterly Financial Statement', 'Earnings Presentation').")

@tool(args_schema=RetrievalInput)
def financial_document_retriever(query: str, year: int, quarter: str, report_type: str) -> str:
    """
    Retrieves specific financial data and context from FAB documents using vector search
    with strict metadata filtering (year, quarter, report_type) for precision.
    The output includes the source document and a snippet of the found text.
    """
    if not os.path.exists(CHROMA_PATH):
        return "Error: Vector database not found. Please run ingest.py first."
        
    print(f"\n--- RETRIEVER CALLED ---")
    print(f"Query: {query}, Filters: Year={year}, Quarter={quarter}, Type={report_type}")
    
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

        chroma_filters = {}
        if year > 0:
            chroma_filters['year'] = year
        if quarter and quarter != 'Unknown':
            chroma_filters['quarter'] = quarter
        if report_type and report_type != 'General':
            chroma_filters['report_type'] = report_type
            
        print(f"Applying filters: {chroma_filters}")
        
        results = vectorstore.similarity_search_with_score(
            query=query, 
            k=3, 
            filter=chroma_filters if chroma_filters else None
        )
        
        if not results:
            return "No relevant financial documents found matching the query and filters. Try adjusting the filter parameters."

        retrieved_data = []
        for doc, score in results:
            metadata = doc.metadata
            retrieved_data.append(f"""
                --- SOURCE DOCUMENT ---
                Document: {metadata.get('source')} ({metadata.get('year')} {metadata.get('quarter')}, Type: {metadata.get('report_type')})
                Score: {score:.4f}
                Content Snippet:
                {doc.page_content}
                --- END SOURCE ---
            """)
            
        return "Retrieved Data:\n" + "\n".join(retrieved_data)

    except Exception as e:
        return f"An error occurred during retrieval: {e}"


def setup_agent():
    """Sets up the main Orchestrator Agent using the Gemini LLM and tools."""
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", 
        temperature=0.0,
        api_key=GEMINI_API_KEY
    )
    
    tools = [calculator_tool, financial_document_retriever]

    system_prompt = f"""
        You are the FAB Financial Analysis Agent, a world-class financial expert. Your primary goal is to answer complex questions 
        about First Abu Dhabi Bank (FAB) financial performance. You MUST adhere to the following rules:

        1.  **Multi-Hop Reasoning:** Deconstruct complex user queries into sequential steps (retrieval -> calculation -> synthesis).
        2.  **Tool Usage (MANDATORY):** - For factual data, **ALWAYS** use the `financial_document_retriever` tool with precise `year`, `quarter`, and `report_type` filters.
            - For all calculations (ratios, percentages, sums, differences), **ALWAYS** use the `calculator_tool`.
        3.  **Accuracy and Verification:** All numerical figures must be traceable to the retrieved documents. Do not hallucinate.
        4.  **Formatting:** When presenting numbers, always include the currency (AED or equivalent) and clearly specify if the figure is in thousands, millions, or billions. Use proper financial notation (e.g., 5,200 million AED).
        5.  **Final Output:** Your final answer must be comprehensive, include the *calculated figures* in the main text, and clearly cite the *source documents* used from the retrieval results.

        **CRITICAL INSTRUCTION for Retrieval:** When using `financial_document_retriever`, meticulously specify the `year`, `quarter`, and `report_type` from the user's query or your analysis plan. If you need data from multiple quarters, you MUST call the retriever multiple times.
    """
    
    agent_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        debug=False,
    )

    return agent_graph

def run_query(query: str, agent_executor):
    """Executes a single query against the agent."""
    print(f"\n=======================================================")
    print(f"USER QUERY: {query}")
    print(f"=======================================================")
    
    try:
        inputs = {"messages": [{"role": "user", "content": query}]}

        if hasattr(agent_executor, "run"):
            result = agent_executor.run(inputs)
            print(f"\n--- FINAL ANSWER ---")
            print(result)
            print(f"--- END FINAL ANSWER ---\n")
        elif hasattr(agent_executor, "stream"):
            for chunk in agent_executor.stream(inputs, stream_mode="updates"):
                print(chunk)
        else:
            print("Agent object does not support `run` or `stream` execution methods.")
    except Exception as e:
        print(f"An error occurred during agent execution: {e}")
        print("This often happens if the LLM tries to call a tool with incorrect arguments or if the context length is exceeded.")

if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is not set. Please set it in your environment or .env file.")
    elif not os.path.exists(CHROMA_PATH):
        print(f"ERROR: Chroma database not found at {CHROMA_PATH}. Please run `python ingest.py` first.")
    else:
        agent_executor = setup_agent()
        query_1 = "What was the year-over-year percentage change in Net Profit between Q3 2023 and Q3 2024? Calculate the growth rate and explain the key factors driving this change."
        run_query(query_1, agent_executor)

        query_2 = "Compare FAB's loan-to-deposit ratio between Q4 2022 and Q4 2023. Has the bank's lending activity increased or decreased relative to its deposit base?"
        run_query(query_2, agent_executor)
        
        query_3 = "If FAB's net interest income was 7,895 million AED in one period and 9,120 million AED in the next, what is the growth percentage?"
        run_query(query_3, agent_executor)