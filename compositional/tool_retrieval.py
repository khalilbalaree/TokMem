#!/usr/bin/env python3
"""
Tool Retrieval Module using ToolLLM-inspired Embedding-based Approach

This module implements semantic similarity search for retrieving relevant tools
based on user queries, reducing prompt length and improving accuracy.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import json


class ToolRetriever:
    """
    Retrieves relevant tools using semantic similarity search.
    Based on ToolLLM's embedding-based retrieval approach.
    """
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize with a pre-trained sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        print(f"Initializing ToolRetriever with model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.tool_embeddings = None
        self.tool_descriptions = None
        self.tool_names = []
        self.tool_texts = []  # Store for debugging
    
    def index_tools(self, tool_descriptions: Dict[str, Any]):
        """
        Pre-compute embeddings for all tools.
        
        Args:
            tool_descriptions: Dictionary mapping tool names to their descriptions
        """
        # Combine tool info into searchable text
        tool_texts = []
        self.tool_names = []
        self.tool_descriptions = tool_descriptions
        
        for name, info in tool_descriptions.items():
            # Combine name, description, and parameter info
            text_parts = [f"{name}"]
            
            # Add description if available
            if isinstance(info, dict) and 'description' in info:
                text_parts.append(info['description'])
            
            # Add parameter information if available
            if isinstance(info, dict) and 'parameters' in info:
                params = info['parameters']
                if isinstance(params, dict):
                    param_names = list(params.keys())
                    if param_names:
                        text_parts.append(f"Parameters: {', '.join(param_names)}")
                    
                    # Add parameter descriptions if available
                    for param_name, param_info in params.items():
                        if isinstance(param_info, dict) and 'description' in param_info:
                            text_parts.append(f"{param_name}: {param_info['description']}")
            
            # Combine all parts
            text = " ".join(text_parts)
            tool_texts.append(text)
            self.tool_names.append(name)
        
        self.tool_texts = tool_texts  # Store for debugging
        
        # Encode all tools at once
        print(f"Encoding {len(tool_texts)} tools...")
        self.tool_embeddings = self.encoder.encode(tool_texts, convert_to_numpy=True)
        print(f"Tool indexing complete. Indexed {len(self.tool_names)} tools.")
    
    def retrieve(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Retrieve top-k relevant tools for a query.
        
        Args:
            query: User query to search for relevant tools
            top_k: Number of tools to retrieve
            
        Returns:
            Dictionary of retrieved tools with their descriptions
        """
        if self.tool_embeddings is None:
            raise ValueError("Tools not indexed. Call index_tools() first.")
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        
        # Compute cosine similarity
        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        tools_norm = self.tool_embeddings / np.linalg.norm(self.tool_embeddings, axis=1, keepdims=True)
        
        similarities = np.dot(tools_norm, query_norm.T).squeeze()
        
        # Handle case where we have fewer tools than requested
        actual_k = min(top_k, len(self.tool_names))
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-actual_k:][::-1]
        
        # Return selected tools with their similarity scores for debugging
        retrieved_tools = {}
        for idx in top_indices:
            tool_name = self.tool_names[idx]
            retrieved_tools[tool_name] = self.tool_descriptions[tool_name]
        
        return retrieved_tools
    
    def retrieve_with_scores(self, query: str, top_k: int = 10) -> List[tuple]:
        """
        Retrieve top-k relevant tools with similarity scores.
        
        Args:
            query: User query to search for relevant tools
            top_k: Number of tools to retrieve
            
        Returns:
            List of (tool_name, tool_description, similarity_score) tuples
        """
        if self.tool_embeddings is None:
            raise ValueError("Tools not indexed. Call index_tools() first.")
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        
        # Compute cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        tools_norm = self.tool_embeddings / np.linalg.norm(self.tool_embeddings, axis=1, keepdims=True)
        
        similarities = np.dot(tools_norm, query_norm.T).squeeze()
        
        # Handle case where we have fewer tools than requested
        actual_k = min(top_k, len(self.tool_names))
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-actual_k:][::-1]
        
        # Return selected tools with scores
        results = []
        for idx in top_indices:
            tool_name = self.tool_names[idx]
            score = float(similarities[idx])
            results.append((tool_name, self.tool_descriptions[tool_name], score))
        
        return results


def test_retriever():
    """Test the retriever with sample data"""
    # Sample tool descriptions
    sample_tools = {
        "calculate_mean": {
            "description": "Calculate the mean of a list of numbers",
            "parameters": {
                "numbers": {"description": "List of numbers to calculate mean"}
            }
        },
        "sort_list": {
            "description": "Sort a list in ascending or descending order",
            "parameters": {
                "items": {"description": "List to sort"},
                "reverse": {"description": "Sort in descending order if True"}
            }
        },
        "find_max": {
            "description": "Find the maximum value in a list",
            "parameters": {
                "values": {"description": "List of values"}
            }
        }
    }
    
    # Initialize retriever
    retriever = ToolRetriever()
    retriever.index_tools(sample_tools)
    
    # Test queries
    test_queries = [
        "I need to find the average of some numbers",
        "Sort these items in reverse order",
        "What's the largest value?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve_with_scores(query, top_k=2)
        for tool_name, _, score in results:
            print(f"  - {tool_name}: {score:.3f}")


if __name__ == "__main__":
    test_retriever()