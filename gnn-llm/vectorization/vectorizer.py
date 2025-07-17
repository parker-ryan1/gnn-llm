import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import pinecone
from pinecone import Pinecone, ServerlessSpec
import logging

class DataVectorizer:
    def __init__(self, model_name: str, pinecone_api_key: str, 
                 pinecone_environment: str, index_name: str):
        self.model = SentenceTransformer(model_name)
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.logger = logging.getLogger(__name__)
        
        # Create index if it doesn't exist
        self._setup_pinecone_index()
    
    def _setup_pinecone_index(self):
        """Setup Pinecone index"""
        try:
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.model.get_sentence_embedding_dimension(),
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            self.logger.error(f"Error setting up Pinecone: {e}")
            raise
    
    def prepare_text_data(self, df: pd.DataFrame, text_columns: List[str]) -> List[str]:
        """Combine text columns into single text for each row"""
        texts = []
        for _, row in df.iterrows():
            combined_text = " ".join([
                f"{col}: {str(row[col])}" for col in text_columns 
                if pd.notna(row[col])
            ])
            texts.append(combined_text)
        return texts
    
    def vectorize_and_store(self, df: pd.DataFrame, text_columns: List[str], 
                           id_column: str = None) -> Dict[str, np.ndarray]:
        """Vectorize data and store in Pinecone"""
        texts = self.prepare_text_data(df, text_columns)
        embeddings = self.model.encode(texts)
        
        # Prepare vectors for Pinecone
        vectors_to_upsert = []
        embedding_dict = {}
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vector_id = str(df.iloc[i][id_column]) if id_column else str(i)
            
            # Store embedding in dict for graph building
            embedding_dict[vector_id] = embedding
            
            # Prepare metadata
            metadata = {
                'text': text,
                'original_data': df.iloc[i].to_dict()
            }
            
            vectors_to_upsert.append({
                'id': vector_id,
                'values': embedding.tolist(),
                'metadata': metadata
            })
        
        # Upsert to Pinecone in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        self.logger.info(f"Stored {len(vectors_to_upsert)} vectors in Pinecone")
        return embedding_dict
    
    def query_similar(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """Query similar vectors from Pinecone"""
        query_embedding = self.model.encode([query_text])
        results = self.index.query(
            vector=query_embedding[0].tolist(),
            top_k=top_k,
            include_metadata=True
        )
        return results['matches']