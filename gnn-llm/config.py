import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database settings
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/dbname')
    
    # Pinecone settings
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'graph-embeddings')
    
    # Ollama settings
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral')
    
    # Model settings
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '384'))
    
    # GNN settings
    GNN_HIDDEN_DIM = int(os.getenv('GNN_HIDDEN_DIM', '128'))
    GNN_NUM_LAYERS = int(os.getenv('GNN_NUM_LAYERS', '3'))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.001'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
    
    # Graph settings
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
    MAX_EDGES_PER_NODE = int(os.getenv('MAX_EDGES_PER_NODE', '10'))
    
    # Ordering settings
    MAX_BUDGET = float(os.getenv('MAX_BUDGET', '500.0'))
    MAX_ITEMS = int(os.getenv('MAX_ITEMS', '3'))
    MIN_CONFIDENCE = float(os.getenv('MIN_CONFIDENCE', '0.6'))
    AUTO_ORDER = os.getenv('AUTO_ORDER', 'false').lower() == 'true'
    
    # API settings
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', '8000'))
    
    # Monitoring settings
    ENABLE_MONITORING = os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'
    MONITORING_INTERVAL = int(os.getenv('MONITORING_INTERVAL', '30'))
    
    # Security settings
    API_KEY = os.getenv('API_KEY', '')  # Optional API key for endpoints
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))
    
    # Logging settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/pipeline.log')
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        errors = []
        
        if not cls.PINECONE_API_KEY:
            errors.append("PINECONE_API_KEY is required")
        
        if not cls.DATABASE_URL:
            errors.append("DATABASE_URL is required")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True