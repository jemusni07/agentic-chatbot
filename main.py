# Chatbot with PostgreSQL Integration

# Objectives:
# 1. Use different message types - Human Message and AI Message
# 2. Maintain a full conversation history using both messagetypes
# 3. Use GPT-4o model using LangChain's ChatOpenAI
# 4. Create a sophisticated conversation LookupError
# 5. Connect to PostgreSQL database and save conversations

# Main Goal: Create a form of memory for our Agent with database persistence

import os 
from typing import TypedDict, List, Union, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import uuid
from datetime import datetime
from langchain_chroma import Chroma
from langchain_core.tools import tool
from operator import add as add_messages

load_dotenv()

# Database Connection Functions
def connect_to_db():
    """Connect to PostgreSQL database"""
    try:
        connection = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'your_database_name'),
            user=os.getenv('DB_USER', 'your_username')
        )
        print("✅ Successfully connected to PostgreSQL!")
        return connection
    except psycopg2.Error as e:
        print(f"❌ Error connecting to PostgreSQL: {e}")
        return None

def setup_database():
    """Create the conversations table if it doesn't exist"""
    conn = connect_to_db()
    if not conn:
        return False
    
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(100),
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        print("✅ Conversations table created/verified")
        return True
    except psycopg2.Error as e:
        print(f"❌ Error setting up database: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def get_last_conversations(limit: int = 5, for_context: bool = False):
    """Retrieve the last N conversations from the database"""
    conn = connect_to_db()
    if not conn:
        if not for_context:
            print("❌ Could not connect to database")
        return None
    
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cursor.execute("""
            SELECT id, session_id, content, timestamp 
            FROM chat_history 
            ORDER BY timestamp DESC 
            LIMIT %s;
        """, (limit,))
        
        conversations = cursor.fetchall()
        
        if not for_context and conversations:
            print(f"\n📚 Last {len(conversations)} conversations:")
            print("=" * 60)
            for i, conv in enumerate(conversations, 1):
                print(f"\n{i}. Session: {conv['session_id'][:8]}...")
                print(f"   Date: {conv['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Content:\n{conv['content'][:200]}{'...' if len(conv['content']) > 200 else ''}")
                print("-" * 40)
        elif not for_context:
            print("\n📚 No conversations found in database")
        
        return conversations
        
    except psycopg2.Error as e:
        if not for_context:
            print(f"❌ Error retrieving conversations: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def prepare_context_from_conversations(conversations, max_conversations: int = 3):
    """Prepare context string from previous conversations"""
    if not conversations:
        return ""
    
    context_parts = []
    for i, conv in enumerate(conversations[:max_conversations]):
        context_parts.append(f"Previous Conversation {i+1} (Session: {conv['session_id'][:8]}...):")
        context_parts.append(f"Date: {conv['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        context_parts.append(conv['content'])
        context_parts.append("-" * 50)
    
    return "\n".join(context_parts)

def save_conversation_to_db(conversation_history: List[Union[HumanMessage, AIMessage]], session_id: str):
    """Save the entire conversation as one combined entry to PostgreSQL database"""
    conn = connect_to_db()
    if not conn:
        print("❌ Could not connect to database to save conversation")
        return False
    
    cursor = conn.cursor()
    
    try:
        print(f"💾 Saving conversation to database (Session ID: {session_id})...")
        
        # Combine all messages into one conversation text
        conversation_text = ""
        for message in conversation_history:
            if isinstance(message, HumanMessage):
                conversation_text += f"Human: {message.content}\n\n"
            elif isinstance(message, AIMessage):
                conversation_text += f"AI: {message.content}\n\n"
        
        # Save the entire conversation as one record
        cursor.execute("""
            INSERT INTO chat_history (session_id, content) 
            VALUES (%s, %s);
        """, (session_id, conversation_text.strip()))
        
        conn.commit()
        
        print(f"✅ Successfully saved complete conversation to database!")
        return True
        
    except psycopg2.Error as e:
        print(f"❌ Error saving conversation to database: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

# LangGraph Chatbot Setup
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# Initialize models
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Vector store will be loaded only when needed
chat_vectorstore = None

def load_chat_history_vectorstore():
    """Load the pre-trained chat history vector store (only when needed for search)"""
    global chat_vectorstore
    
    if chat_vectorstore is not None:
        return chat_vectorstore
        
    persist_directory = "./chat_history_rag"
    collection_name = "chat_conversations"
    
    try:
        if os.path.exists(persist_directory):
            chat_vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            print("✅ Loaded chat history vector store for search")
            return chat_vectorstore
        else:
            print("⚠️ Chat history vector store not found. Run train_chat_history.py first.")
            return None
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        return None

def search_vector_database(query: str) -> str:
    """Search vector database for relevant previous conversations"""
    # Load vector store only when actually needed for search
    vectorstore = load_chat_history_vectorstore()
    
    if not vectorstore:
        return ""
    
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )
        
        docs = retriever.invoke(query)
        
        if not docs:
            return ""
        
        results = []
        for i, doc in enumerate(docs, 1):
            session_id = doc.metadata.get('session_id', 'Unknown')[:8]
            date = doc.metadata.get('date', 'Unknown')
            content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            results.append(f"Previous Conversation {i} (Session: {session_id}..., Date: {date}):\n{content}\n")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Error searching: {str(e)}"

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    last_five_context: str
    needs_vector_search: bool
    needs_general_knowledge: bool

def classify_message_intent(message: str) -> str:
    """Classify if message is conversational or a question needing information retrieval"""
    classification_prompt = f"""
    Classify this message as either "conversational" or "question":
    
    - "conversational": Greetings, thanks, acknowledgments, casual chat, emotional expressions, simple responses
    - "question": Requests for specific information, explanations, how-to queries, recall of past discussions, complex topics
    
    Message: "{message}"
    
    Respond with just one word: "conversational" or "question"
    """
    
    try:
        response = llm.invoke([HumanMessage(content=classification_prompt)])
        intent = response.content.strip().lower()
        return intent if intent in ["conversational", "question"] else "question"  # default to question if unclear
    except:
        return "question"  # default to question on error

def try_with_last_five(state: AgentState) -> AgentState:
    """Step 2: Try to answer with last 5 conversations from PostgreSQL"""
    print("🔄 Node: try_with_last_five - Processing message...")
    
    # Get the latest user message
    latest_message = state["messages"][-1].content
    
    # Classify message intent
    message_intent = classify_message_intent(latest_message)
    print(f"💭 Message classified as: {message_intent}")
    
    # For conversational messages, respond directly without extensive search
    if message_intent == "conversational":
        # Simple context-aware response for conversational messages
        context_prompt = f"""
        Current conversation:
        """
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                context_prompt += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                context_prompt += f"AI: {msg.content}\n"
        
        context_prompt += f"\nRespond naturally to this conversational message:"
        
        response = llm.invoke([HumanMessage(content=context_prompt)])
        state["needs_vector_search"] = False
        print(f"\nAI: {response.content}")
        state["messages"] = list(state["messages"]) + [AIMessage(content=response.content)]
        return state
    
    # For questions, use full context and potentially search vector DB
    context_prompt = f"""
    Recent chat history from previous sessions:
    {state['last_five_context']}
    
    Current conversation:
    """
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            context_prompt += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            context_prompt += f"AI: {msg.content}\n"
    
    context_prompt += f"\nCan you answer the current question based on the recent history and current conversation? If you can answer confidently, provide the answer. If you need more information from older conversations, just say 'NEED_MORE_INFO'."
    
    response = llm.invoke([HumanMessage(content=context_prompt)])
    
    # Check if we need vector search
    if "NEED_MORE_INFO" in response.content:
        state["needs_vector_search"] = True
        print("🔍 Last 5 conversations not sufficient, will search vector database...")
    else:
        state["needs_vector_search"] = False
        print(f"\nAI: {response.content}")
        # Add response to conversation
        state["messages"] = list(state["messages"]) + [AIMessage(content=response.content)]
    
    return state

def search_and_respond(state: AgentState) -> AgentState:
    """Step 3: Search vector database and provide response if sufficient"""
    print("🔄 Node: search_and_respond - Searching vector database...")
    
    current_query = state["messages"][-1].content
    print("🔍 Searching vector database for more information...")
    
    vector_results = search_vector_database(current_query)
    
    # Check if vector search found relevant information
    if not vector_results or "No additional relevant information found" in vector_results:
        print("🤔 Vector database search insufficient, will use general knowledge...")
        state["needs_general_knowledge"] = True
        return state
    
    # Combine everything for final answer
    final_prompt = f"""
    Recent chat history from previous sessions:
    {state['last_five_context']}
    
    Additional information from older conversations:
    {vector_results}
    
    Current conversation:
    """
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            final_prompt += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            final_prompt += f"AI: {msg.content}\n"
    
    final_prompt += f"\nCan you answer the question using the information from previous conversations? If the information is sufficient, provide a complete answer. If not, just say 'NEED_GENERAL_KNOWLEDGE'."
    
    final_response = llm.invoke([HumanMessage(content=final_prompt)])
    
    # Check if we need general knowledge
    if "NEED_GENERAL_KNOWLEDGE" in final_response.content:
        state["needs_general_knowledge"] = True
        print("🤔 Vector search context insufficient, will use general knowledge...")
    else:
        state["needs_general_knowledge"] = False
        print(f"\nAI: {final_response.content}")
        # Add response to conversation
        state["messages"] = list(state["messages"]) + [AIMessage(content=final_response.content)]
    
    return state

def general_knowledge_response(state: AgentState) -> AgentState:
    """Step 4: Use general LLM knowledge when database searches are insufficient"""
    print("🔄 Node: general_knowledge_response - Using general LLM knowledge...")
    
    current_query = state["messages"][-1].content
    
    # Build prompt with available context but rely on general knowledge
    general_prompt = f"""
    You are a helpful AI assistant. The user has asked a question that wasn't found in previous conversation history.
    
    Current conversation:
    """
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            general_prompt += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            general_prompt += f"AI: {msg.content}\n"
    
    general_prompt += f"\nAnswer the question using your general knowledge. Be helpful and informative:"
    
    general_response = llm.invoke([HumanMessage(content=general_prompt)])
    print(f"\nAI: {general_response.content}")
    
    # Add response to conversation
    state["messages"] = list(state["messages"]) + [AIMessage(content=general_response.content)]
    return state

def should_search_vector(state: AgentState) -> str:
    """Route: check if we need vector search or we're done with this turn"""
    next_node = "search_vector" if state.get("needs_vector_search", False) else END
    print(f"🔀 Router: should_search_vector - Going to: {next_node}")
    return next_node

def should_use_general_knowledge(state: AgentState) -> str:
    """Route: check if we need general knowledge or we're done with this turn"""
    next_node = "general_knowledge" if state.get("needs_general_knowledge", False) else END
    print(f"🔀 Router: should_use_general_knowledge - Going to: {next_node}")
    return next_node

# Build LangGraph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("try_last_five", try_with_last_five)
graph.add_node("search_vector", search_and_respond)
graph.add_node("general_knowledge", general_knowledge_response)

# Add edges
graph.add_edge(START, "try_last_five")
graph.add_conditional_edges(
    "try_last_five",
    should_search_vector,
    {"search_vector": "search_vector", END: END}
)
graph.add_conditional_edges(
    "search_vector",
    should_use_general_knowledge,
    {"general_knowledge": "general_knowledge", END: END}
)
graph.add_edge("general_knowledge", END)

agent = graph.compile()

def main():
    """Main function to run the chatbot"""
    print("🚀 Starting Agentic Chatbot with Database Integration...\n")
    
    # 1. Setup database
    use_database = setup_database()
    
    # 2. Get last 5 conversations from postgres
    last_five_context = ""
    if use_database:
        print("🧠 Loading last 5 conversations from database...")
        past_conversations = get_last_conversations(limit=5, for_context=True)
        last_five_context = prepare_context_from_conversations(past_conversations, max_conversations=5)
        if last_five_context:
            print("✅ Last 5 conversations loaded")
        else:
            print("ℹ️ No previous conversations found")
    
    # Generate unique session ID for this conversation
    session_id = str(uuid.uuid4())
    
    # Start chatbot
    print("💬 Starting chatbot (type 'exit' to quit)...\n")
    conversation_history = []

    user_input = input("Enter: ")
    while user_input != "exit":
        # Add user message to current session
        conversation_history.append(HumanMessage(content=user_input))
        
        # Process with LangGraph: last 5 -> vector DB if needed -> general knowledge -> maintain session context
        result = agent.invoke({
            "messages": conversation_history,
            "last_five_context": last_five_context,
            "needs_vector_search": False,
            "needs_general_knowledge": False
        })
        
        # Update conversation history with the result
        conversation_history = result["messages"]
        
        user_input = input("Enter: ")

    # 4. Save conversation to database when exit
    if use_database and conversation_history:
        save_conversation_to_db(conversation_history, session_id)

if __name__ == "__main__":
    main()
