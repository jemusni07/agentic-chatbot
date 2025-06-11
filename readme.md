## Stateful Conversational AI Chatbot

### Overview

Most AI chatbots are stateless. They are designed to be used for a session and when the session is over, the conversation history is lost. But there are cases where we want to maintain a conversation history like a chatbot for a customer support agent where the interaction can be personalized.

One of the solution to this is to save all conversation in a database, let's say Postgres, and train a vector database out of this database. This is a hybrid setup. What we want to achieve is to optimize for cost and speed while maintaining long term and short term memory.

This setup utilizes LangGraph to route context retrieval for a query on three different hierarchical levels.

1. Last 5 conversations (Postgres)
2. RAG search on Vector database for context ouside of the last 5 conversations (ChromaDB)
3. General knowledge base of the LLM (OpenAI)


![Stateful converasation with hierarchical context retrieval](./images/stateful_chatbot.png)

### Last 5 conversations with Postgres
For a conversational message, it is safe to assume that the user's inquiry can usually be answered using the last 5 conversations. A simple retrieval from a database be it SQL, NoSQL, or a cache may be sufficient and offers a cost efficient way context retrieval. It may also be faster to retrieve from Postgres than do vector search on a large vector database.

Why only the last 5 conversations? More historical conversations(even the whole conversation history), although possible for today's large language models like OpenAI's GPT-4o and Anthropic Claude's Sonnet 4, will be too expensive as the cost of context retrieval increases exponentially with the number of tokens. 

Let's say you are asking for a simple question where the answer only needs one conversation in the history and you use all of your 20 conversations from the past as context, it is simply too cost inefficient.

### Fallback to Vector Database
If the last 5 conversations are not enough to answer the user's question, the conversation will be routed to a vector search. This means, the langgraph agent will now look for the top 2 relevant document on the current version of the vector database.

For this setup, I am using local vector database with ChromaDB. Only the relevant documents retrieved from the vector database will be used as context for the LLM. This avoids the unnecessary use of many tokens. 

We can have a workflow where we update the vector database from what is in postgres on an interval that will best suit the needs of the application. The downside of this is that when we will be using a different llms, we have to retrain the whole vector database because different llms may require different embeddings.


### Fallback to General Knowledge Base
This will be the last fall back for questions and queries that are not part of the user's chat history. It will simply answer the question using whatever knowledge the LLM is trained on.


### How the Context Works 

The conversation history accumulates within the current session:

**Query 1:**
* PostgreSQL context: Last 5 conversations from previous sessions
* Current session: Just Query 1
* Gets processed → AI Response 1

**Query 2:**
* PostgreSQL context: Same last 5 conversations from previous sessions
* Current session: Query 1 + AI Response 1 + Query 2
* Gets processed → AI Response 2

**Query 3:**
* PostgreSQL context: Same last 5 conversations from previous sessions
* Current session: Query 1 + Response 1 + Query 2 + Response 2 + Query 3
* Gets processed → AI Response 3

**The Memory Layers:**
1. PostgreSQL: Last 5 complete conversations from previous sessions (loaded once at startup)
2. Current Session: All Q&As from current session (accumulates in memory)
3. Vector Database: Searched only when needed for older historical context

**Key Point:**
The current session conversation history grows and is passed to every node, so Query 2 definitely has the context of Query 1's exchange.
Only when you type "exit" does the entire current session get saved as one conversation to PostgreSQL, which will then be available as context for future sessions.
Each new query builds on the growing conversation context within that session! 🧠



### Unit Test

#### Chat History and Postgres

I made 6 conversations prior to this test. Gave the database several facts. I explicitly put my name in the very first conversation which will be out of reach with by the "last 5 conversations" context

These are the chat history stored in the database.

![Chat history Stored in Postgres](/images/chat_history_postgres.png)

#### Chat History and Vector Database (ChromaDB)

I trained everry conversation that is in the postgres database for RAG search.

![RAG Training](/images/rag_training.png)


#### Test Hierarchical Context Retrieval

I tested the hierarchical context retrieval with three queries:

1. A query answerable in the last 5 conversations
2. A query answerable in the vector database
3. A query answerable in the general knowledge base

The results are as follows:

![Unit Test](/images/unit_test.png)


We were able to answer all of the queries with the hierarchical context retrieval. LangGraph was able to route the query to the correct context retrieval method.