import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from datetime import datetime

load_dotenv()

# Embedding Model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

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

def load_all_conversations():
    """Load all conversations from the database"""
    conn = connect_to_db()
    if not conn:
        return []
    
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cursor.execute("""
            SELECT id, session_id, content, timestamp 
            FROM chat_history 
            ORDER BY timestamp ASC;
        """)
        
        conversations = cursor.fetchall()
        print(f"📚 Loaded {len(conversations)} conversations from database")
        return conversations
        
    except psycopg2.Error as e:
        print(f"❌ Error loading conversations: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

def conversations_to_documents(conversations):
    """Convert conversations to LangChain Document objects"""
    documents = []
    
    for conv in conversations:
        # Create document with conversation content
        doc = Document(
            page_content=conv['content'],
            metadata={
                'session_id': conv['session_id'],
                'timestamp': conv['timestamp'].isoformat(),
                'conversation_id': conv['id'],
                'date': conv['timestamp'].strftime('%Y-%m-%d'),
                'source': 'chat_history'
            }
        )
        documents.append(doc)
    
    print(f"📄 Created {len(documents)} documents from conversations")
    return documents

def setup_vector_store(documents):
    """Create and populate ChromaDB vector store"""
    
    # Text splitter for conversations (smaller chunks since conversations are already segmented)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for conversations
        chunk_overlap=100,
        separators=["\n\nHuman:", "\n\nAI:", "\n\n", "\n"]  # Split on conversation turns
    )
    
    # Split documents
    docs_split = text_splitter.split_documents(documents)
    print(f"📝 Split into {len(docs_split)} chunks")
    
    # Vector store setup
    persist_directory = "./chat_history_rag"
    collection_name = "chat_conversations"
    
    # Create directory if it doesn't exist
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    try:
        # Create ChromaDB vector store
        vectorstore = Chroma.from_documents(
            documents=docs_split,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print(f"✅ Created ChromaDB vector store with {len(docs_split)} chunks!")
        print(f"📁 Saved to: {persist_directory}")
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error setting up ChromaDB: {str(e)}")
        return None

def test_vector_store(vectorstore):
    """Test the vector store with a sample query"""
    if not vectorstore:
        return
    
    print("\n🔍 Testing vector store...")
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Test queries
    test_queries = [
        "What did we talk about recently?",
        "Previous conversations",
        "What questions have been asked?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            docs = retriever.invoke(query)
            if docs:
                print(f"Found {len(docs)} relevant chunks:")
                for i, doc in enumerate(docs[:2], 1):  # Show first 2 results
                    print(f"  {i}. Session: {doc.metadata.get('session_id', 'Unknown')[:8]}...")
                    print(f"     Date: {doc.metadata.get('date', 'Unknown')}")
                    print(f"     Content: {doc.page_content[:100]}...")
            else:
                print("  No results found")
        except Exception as e:
            print(f"  Error: {e}")

def main():
    """Main training function"""
    print("🚀 Starting Chat History Training for Vector Database...\n")
    
    # Step 1: Load conversations from database
    print("Step 1: Loading conversations from PostgreSQL...")
    conversations = load_all_conversations()
    
    if not conversations:
        print("❌ No conversations found. Make sure you have chat history in your database.")
        return
    
    # Step 2: Convert to documents
    print("\nStep 2: Converting conversations to documents...")
    documents = conversations_to_documents(conversations)
    
    # Step 3: Create vector store
    print("\nStep 3: Creating vector store...")
    vectorstore = setup_vector_store(documents)
    
    # Step 4: Test the vector store
    print("\nStep 4: Testing vector store...")
    test_vector_store(vectorstore)
    
    print("\n🎉 Chat history training completed!")
    print("Your conversations are now available as a vector database for RAG queries.")
    print("Vector store saved to: ./chat_history_rag")

if __name__ == "__main__":
    main() 