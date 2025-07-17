import requests
import json
from typing import List, Dict, Any, Optional
import logging
from vectorization.vectorizer import DataVectorizer

class OllamaRAGSystem:
    def __init__(self, ollama_base_url: str, model_name: str, vectorizer: DataVectorizer):
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
        self.vectorizer = vectorizer
        self.logger = logging.getLogger(__name__)
    
    def query_ollama(self, prompt: str, context: str = "") -> str:
        """Query Ollama with context"""
        full_prompt = f"""
        Context from vector database:
        {context}
        
        Question: {prompt}
        
        Please answer based on the provided context. If the context doesn't contain 
        relevant information, please say so rather than making up information.
        """
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            self.logger.error(f"Error querying Ollama: {e}")
            return f"Error: Could not get response from Ollama - {str(e)}"
    
    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant context from Pinecone"""
        try:
            similar_docs = self.vectorizer.query_similar(query, top_k=top_k)
            
            context_parts = []
            for doc in similar_docs:
                metadata = doc.get('metadata', {})
                text = metadata.get('text', '')
                score = doc.get('score', 0)
                
                context_parts.append(f"[Relevance: {score:.3f}] {text}")
            
            return "\n\n".join(context_parts)
        except Exception as e:
            self.logger.error(f"Error retrieving context: {e}")
            return ""
    
    def rag_query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Perform RAG query with context retrieval and LLM generation"""
        # Retrieve relevant context
        context = self.retrieve_context(question, top_k=top_k)
        
        # Generate response with context
        response = self.query_ollama(question, context)
        
        return {
            "question": question,
            "context": context,
            "response": response,
            "context_sources": top_k
        }
    
    def validate_gnn_output(self, gnn_result: str, original_data: str) -> Dict[str, Any]:
        """Use RAG to validate GNN outputs against original data"""
        validation_prompt = f"""
        Original data: {original_data}
        GNN analysis result: {gnn_result}
        
        Please validate if the GNN analysis result is consistent with the original data.
        Are there any potential hallucinations or inconsistencies?
        """
        
        return self.rag_query(validation_prompt)
    
    def explain_gnn_decision(self, node_data: str, gnn_decision: str) -> Dict[str, Any]:
        """Use RAG to explain GNN decisions based on stored context"""
        explanation_prompt = f"""
        Node data: {node_data}
        GNN decision/output: {gnn_decision}
        
        Based on the available context, please explain why the GNN might have made this decision.
        What patterns in the data could lead to this result?
        """
        
        return self.rag_query(explanation_prompt)