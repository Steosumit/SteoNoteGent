# Overview
Contains notes about learnings during the building

### How to pass the VectorStoreRetriever object throught the llm to tool parameters?

There are many approaches to handle this like using partial tools madule.
I used BaseTool approach to create custom tool class from stratch:

```from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import Field

class CustomRetrieverTool(BaseTool):
    name: str = "retriever_tool"
    description: str = "Retrieves relevant information from user's notes"
    retriever: VectorStoreRetriever = Field(exclude=True)  # Exclude from serialization
    
    def _run(self, query: str) -> str:
        """Execute the retrieval"""
        results = []
        try:
            response = self.retriever.invoke(query)
            if not response:
                return "No relevant information found."
            for i, result in enumerate(response):
                results.append(f"Document {i+1}:\n{result.page_content}\n")
            return "\n\n".join(results)
        except Exception as e:
            return f"[Error]: {str(e)}"
```

### Why are we using common storage self.storage in retriving process?
As the embedding created are memory base. Meaning they are not stored 
anywhere just persist in the memory while the program is running. So, keeping 
a common memory helps save storage

### On what factors the quality of performance depends?
The quality of the performance depends largely:
- on prompt engineering between the tool node and llm node
- the tool code being used