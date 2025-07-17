import logging
import pandas as pd
import numpy as np
from config import Config
from database.db_connector import DatabaseConnector
from vectorization.vectorizer import DataVectorizer
from graph.graph_builder import GraphBuilder
from graph.gnn_model import DistanceMinimizingGNN, GNNTrainer
from rag.rag_system import OllamaRAGSystem
from web_automation.intelligent_ordering import IntelligentOrderingSystem, OrderingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphGNNRAGPipeline:
    def __init__(self):
        self.config = Config()
        self.db_connector = None
        self.vectorizer = None
        self.graph_builder = None
        self.gnn_model = None
        self.gnn_trainer = None
        self.rag_system = None
        self.ordering_system = None
        
    def setup_components(self):
        """Initialize all pipeline components"""
        logger.info("Setting up pipeline components...")
        
        # Database connector
        self.db_connector = DatabaseConnector(self.config.DATABASE_URL)
        
        # Vectorizer with Pinecone
        self.vectorizer = DataVectorizer(
            model_name=self.config.EMBEDDING_MODEL,
            pinecone_api_key=self.config.PINECONE_API_KEY,
            pinecone_environment=self.config.PINECONE_ENVIRONMENT,
            index_name=self.config.PINECONE_INDEX_NAME
        )
        
        # Graph builder
        self.graph_builder = GraphBuilder(
            similarity_threshold=self.config.SIMILARITY_THRESHOLD,
            max_edges_per_node=self.config.MAX_EDGES_PER_NODE
        )
        
        # RAG system
        self.rag_system = OllamaRAGSystem(
            ollama_base_url=self.config.OLLAMA_BASE_URL,
            model_name=self.config.OLLAMA_MODEL,
            vectorizer=self.vectorizer
        )
        
        # Intelligent ordering system
        ordering_config = OrderingConfig(
            max_budget=500.0,  # $500 budget
            max_items=3,       # Max 3 items
            min_confidence=0.6, # 60% confidence threshold
            auto_order=False   # Demo mode - don't actually order
        )
        self.ordering_system = IntelligentOrderingSystem(
            rag_system=self.rag_system,
            ordering_config=ordering_config
        )
        
        logger.info("All components initialized successfully")
    
    def load_and_process_data(self, query: str = None, table_name: str = None, 
                            text_columns: list = None, id_column: str = None):
        """Load data from database and process it"""
        logger.info("Loading data from database...")
        
        # Fetch data
        df = self.db_connector.fetch_data(query=query, table_name=table_name)
        logger.info(f"Loaded {len(df)} records from database")
        
        # Auto-detect text columns if not provided
        if text_columns is None:
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            logger.info(f"Auto-detected text columns: {text_columns}")
        
        # Vectorize and store in Pinecone
        logger.info("Vectorizing data and storing in Pinecone...")
        embeddings = self.vectorizer.vectorize_and_store(
            df, text_columns, id_column
        )
        
        return df, embeddings
    
    def build_and_train_graph(self, embeddings: dict):
        """Build graph and train GNN"""
        logger.info("Building graph from embeddings...")
        
        # Build NetworkX graph
        nx_graph = self.graph_builder.build_graph_from_embeddings(embeddings)
        
        # Convert to PyTorch Geometric format
        pyg_data = self.graph_builder.networkx_to_pytorch_geometric(nx_graph)
        
        # Calculate graph metrics
        metrics = self.graph_builder.calculate_graph_metrics(nx_graph)
        logger.info(f"Graph metrics: {metrics}")
        
        # Initialize GNN model
        input_dim = pyg_data.x.shape[1]
        self.gnn_model = DistanceMinimizingGNN(
            input_dim=input_dim,
            hidden_dim=self.config.GNN_HIDDEN_DIM,
            num_layers=self.config.GNN_NUM_LAYERS
        )
        
        # Initialize trainer
        self.gnn_trainer = GNNTrainer(
            model=self.gnn_model,
            learning_rate=self.config.LEARNING_RATE
        )
        
        # Train the model
        logger.info("Training GNN model...")
        losses = self.gnn_trainer.train(pyg_data, num_epochs=100)
        
        # Get optimized embeddings
        optimized_embeddings = self.gnn_trainer.get_optimized_embeddings(pyg_data)
        
        return nx_graph, pyg_data, optimized_embeddings, losses
    
    def run_rag_analysis(self, sample_questions: list = None):
        """Run RAG analysis with sample questions"""
        if sample_questions is None:
            sample_questions = [
                "What patterns do you see in the data?",
                "What are the main clusters or groups?",
                "How are the data points connected?",
                "What insights can you provide about the relationships?"
            ]
        
        logger.info("Running RAG analysis...")
        results = []
        
        for question in sample_questions:
            result = self.rag_system.rag_query(question)
            results.append(result)
            logger.info(f"Q: {question}")
            logger.info(f"A: {result['response'][:200]}...")
        
        return results
    
    def run_intelligent_ordering(self, pipeline_results: dict) -> dict:
        """Run intelligent ordering based on GNN and RAG results"""
        logger.info("Running intelligent ordering analysis...")
        
        try:
            # Extract GNN results for ordering analysis
            gnn_results = {
                'graph': pipeline_results.get('graph'),
                'training_losses': pipeline_results.get('training_losses'),
                'optimized_embeddings': pipeline_results.get('optimized_embeddings')
            }
            
            # Run intelligent ordering
            ordering_results = self.ordering_system.execute_intelligent_ordering(
                gnn_results=gnn_results,
                original_data=pipeline_results.get('data')
            )
            
            return ordering_results
            
        except Exception as e:
            logger.error(f"Intelligent ordering failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'product_suggestions': [],
                'recommendations': []
            }
    
    def run_full_pipeline(self, query: str = None, table_name: str = None,
                         text_columns: list = None, id_column: str = None):
        """Run the complete pipeline"""
        try:
            # Setup components
            self.setup_components()
            
            # Load and process data
            df, embeddings = self.load_and_process_data(
                query=query, table_name=table_name,
                text_columns=text_columns, id_column=id_column
            )
            
            # Build graph and train GNN
            nx_graph, pyg_data, optimized_embeddings, losses = self.build_and_train_graph(embeddings)
            
            # Run RAG analysis
            rag_results = self.run_rag_analysis()
            
            # Prepare pipeline results
            pipeline_results = {
                'data': df,
                'original_embeddings': embeddings,
                'optimized_embeddings': optimized_embeddings,
                'graph': nx_graph,
                'training_losses': losses,
                'rag_results': rag_results
            }
            
            # Run intelligent ordering
            ordering_results = self.run_intelligent_ordering(pipeline_results)
            pipeline_results['ordering_results'] = ordering_results
            
            logger.info("Pipeline completed successfully!")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Example usage with sample data"""
    pipeline = GraphGNNRAGPipeline()
    
    try:
        # Run with sample documents table
        results = pipeline.run_full_pipeline(
            table_name="documents",
            text_columns=["title", "content", "category"],
            id_column="id"
        )
        
        print("Pipeline completed successfully!")
        print(f"Processed {len(results['data'])} records")
        print(f"Final training loss: {results['training_losses'][-1]:.4f}")
        
        # Print some sample RAG results
        print("\nSample RAG Results:")
        for i, result in enumerate(results['rag_results'][:2]):
            print(f"\nQ{i+1}: {result['question']}")
            print(f"A{i+1}: {result['response'][:200]}...")
        
        # Print ordering results
        ordering_results = results.get('ordering_results', {})
        if ordering_results.get('success'):
            print(f"\nIntelligent Ordering Results:")
            print(f"Product suggestions: {len(ordering_results.get('product_suggestions', []))}")
            print(f"Recommendations found: {len(ordering_results.get('recommendations', []))}")
            print(f"Total estimated cost: ${ordering_results.get('total_cost', 0):.2f}")
            
            # Show top recommendations
            recommendations = ordering_results.get('recommendations', [])[:3]
            for i, rec in enumerate(recommendations):
                print(f"\nRecommendation {i+1}:")
                print(f"  Product: {rec['product_name']}")
                print(f"  Price: ${rec['price']:.2f}")
                print(f"  Confidence: {rec['confidence_score']:.2f}")
                print(f"  Reasoning: {rec['reasoning'][:100]}...")
        else:
            print(f"\nOrdering analysis completed with warnings:")
            error_messages = ordering_results.get('error_messages', [])
            if error_messages:
                for error in error_messages:
                    print(f"  - {error}")
            else:
                print("  - No specific errors reported")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()