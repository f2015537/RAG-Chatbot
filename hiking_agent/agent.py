from google.adk.agents.llm_agent import Agent
from google.adk.tools import FunctionTool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def retrieve_info(query: str) -> str:
    """Search for information about hikes in Washington state."""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = Chroma(
            collection_name="washington_hikes",
            embedding_function=embeddings,
            persist_directory="./chroma_db",
        )
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        print(f"Querying retrieve_info with: {query}")
        print("--------------------------------------------------------------")
        results = retriever.invoke(query)
        print(f"Retrieved documents: {len(results)} matches found")
        for i, doc in enumerate(results):
            print(f"Document {i + 1}: {doc.page_content[:100]}...")
        print("--------------------------------------------------------------")

        content = "\n".join([doc.page_content for doc in results])
        if not content:
            print(f"No content retrieved for query: {query}")
            return f"No information found for '{query}'."

        print(f"Returning content: {content[:200]}...")
        return content
    except Exception as e:
        print(f"Error in retrieve_info: {e}")
        return f"Error retrieving information for '{query}'. Please try again."


retrieve_tool = FunctionTool(func=retrieve_info)

root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='A helpful assistant for answering questions about hikes in Washington state.',
    instruction=(
        'You are a Washington state hiking expert. '
        'You MUST use the "retrieve_info" tool to look up trail information before answering. '
        'Use it to answer questions about hike difficulty, distance, elevation gain, best season to visit, '
        'trail conditions, permit requirements, and recommendations for hikes near specific locations. '
        'If the tool returns no relevant results, say so honestly rather than guessing.'
    ),
    tools=[retrieve_tool],
)
