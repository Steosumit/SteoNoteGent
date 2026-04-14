# SteoNoteGent

Personal RAG-powered note assistant that answers questions about your notes using AI with web search capabilities.

## ✨ Technologies

- **LangGraph** - Agent workflow orchestration
- **LangChain** - LLM framework and tools
- **Google Gemini Flash** - Primary LLM
- **HuggingFace Embeddings** - Sentence transformer embeddings (all-MiniLM-L6-v2)
- **DuckDuckGo Search** - Web search tool
- **CSV Loader** - Data loading from CSV notes

## 🚀 Features

- Retrieval Augmented Generation (RAG) for personal notes
- Natural language querying of schedule and tasks
- Web search integration for expanded knowledge
- Tool-augmented AI agent with dual-tool workflow
- In-memory vector storage for fast retrieval
- JSON-structured responses with confidence scores

## 📍 The Process

This project started as an exploration into building AI agents with tool-augmented capabilities. The goal was simple: create an assistant that could answer questions about personal notes stored in CSV format. After researching various approaches, LangGraph emerged as the perfect framework for orchestrating the agent workflow. The biggest challenge was passing the VectorStoreRetriever object to the tool parameters - solved by using a custom BaseTool class with Pydantic Field exclusion. The agent uses a think-action-observation loop inspired by ReAct, continuously reasoning until it can provide a final answer in structured JSON format.

## 🎞️ Running the Project

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env`:
   - `GOOGLE_API_KEY` for Gemini
4. Prepare your notes in `data/sample.csv` with columns: day, month, year, time, tasknote
5. Run the application:
   ```bash
   python app.py
   ```

## Preview

The agent will load your notes, create embeddings, and respond to queries like:
- "At what time was the gym session?"
- "What meetings do I have on December 5th?"

Output is returned as JSON with the assistant's response and confidence level.
