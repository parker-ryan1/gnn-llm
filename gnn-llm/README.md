# Graph-GNN-RAG Pipeline with Intelligent Ordering

A complete AI pipeline that combines Graph Neural Networks, Vector Databases, Retrieval-Augmented Generation, and Intelligent Web Automation for data analysis and automated purchasing decisions.

## 🚀 Features

- **Database Integration**: Fetch and analyze data from PostgreSQL
- **Vector Storage**: Store embeddings in Pinecone for similarity search
- **Graph Construction**: Build graphs from data relationships using cosine similarity
- **GNN Training**: Train models to minimize node distances and optimize connectivity
- **RAG System**: Use local Ollama + Mistral for grounded, hallucination-free responses
- **Intelligent Ordering**: Automatically analyze results and suggest/order relevant products
- **Web Automation**: Scrape and interact with e-commerce sites (Amazon, eBay)
- **Complete Dockerization**: Fully containerized deployment with all dependencies

## 🏗️ Architecture

```
Database → Vectorization → Pinecone → Graph Building → GNN Training → RAG Analysis → Intelligent Ordering → Web Automation
```

## 🐳 Quick Start with Docker

1. **Setup environment:**

```bash
# Copy and edit the environment file
cp .env .env.local
# Edit .env.local with your Pinecone API key
```

2. **Start all services:**

```bash
docker-compose up -d
```

3. **Monitor the pipeline:**

```bash
docker-compose logs -f graph-gnn-app
```

4. **Access services:**

- Main Application: Running in container
- PostgreSQL: `localhost:5432` (graphuser/graphpass)
- Ollama API: `localhost:11434`
- Sample data automatically loaded

## 📋 Prerequisites

- Docker and Docker Compose
- Pinecone API account (free tier available)
- 4GB+ RAM recommended
- Optional: NVIDIA GPU for faster processing

## ⚙️ Configuration

Edit `.env` file with your settings:

```env
# Required: Get from Pinecone dashboard
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=graph-embeddings

# Database (auto-configured in Docker)
DATABASE_URL=postgresql://graphuser:graphpass@postgres:5432/graphdb

# Ollama (auto-configured in Docker)
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=mistral

# Ordering Configuration
MAX_BUDGET=500.0
MAX_ITEMS=3
MIN_CONFIDENCE=0.6
AUTO_ORDER=false  # Set to true for actual ordering (DEMO MODE)
```

## 🧠 Pipeline Components

### 1. Data Vectorization (`vectorization/`)

- Uses SentenceTransformers (all-MiniLM-L6-v2) for embeddings
- Stores vectors in Pinecone with rich metadata
- Supports batch processing and similarity queries

### 2. Graph Building (`graph/graph_builder.py`)

- Creates graphs from embedding similarity
- Configurable similarity thresholds and edge limits
- Converts between NetworkX and PyTorch Geometric formats

### 3. GNN Model (`graph/gnn_model.py`)

- Custom architecture designed for distance minimization
- Specialized loss function rewards shorter node paths
- Trains to optimize graph connectivity and relationships

### 4. RAG System (`rag/rag_system.py`)

- Retrieves relevant context from Pinecone
- Uses local Ollama + Mistral for generation
- Validates GNN outputs to prevent hallucination

### 5. Intelligent Ordering (`web_automation/`)

- Analyzes GNN/RAG results to suggest products
- Scrapes e-commerce sites for relevant items
- Uses LLM to evaluate product relevance and quality
- **DEMO MODE**: Shows recommendations without actual purchasing

## 🛒 Intelligent Ordering Process

1. **Analysis**: GNN results analyzed to identify product needs
2. **Search**: Web scraping of Amazon/eBay for relevant products
3. **Evaluation**: LLM evaluates each product for relevance and quality
4. **Filtering**: Budget and confidence-based filtering
5. **Recommendation**: Ranked list of purchase recommendations
6. **Ordering**: Optional automated purchasing (DEMO MODE ONLY)

## 📊 Sample Output

```
Pipeline completed successfully!
Processed 8 records
Final training loss: 0.1234

Sample RAG Results:
Q1: What patterns do you see in the data?
A1: Based on the graph analysis, I can see clear clustering patterns around AI and database topics...

Intelligent Ordering Results:
Product suggestions: 5
Recommendations found: 3
Total estimated cost: $284.97

Recommendation 1:
  Product: Wireless Programming Keyboard
  Price: $149.99
  Confidence: 0.85
  Reasoning: Highly relevant for developers working with databases and AI projects...
```

## 🔧 Manual Setup (Alternative)

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Install Chrome/ChromeDriver:**

```bash
# Ubuntu/Debian
sudo apt-get install google-chrome-stable
# Download ChromeDriver from https://chromedriver.chromium.org/
```

3. **Start services:**

```bash
# PostgreSQL
docker run -d --name postgres -e POSTGRES_DB=graphdb -e POSTGRES_USER=graphuser -e POSTGRES_PASSWORD=graphpass -p 5432:5432 postgres:15

# Ollama
docker run -d --name ollama -p 11434:11434 ollama/ollama:latest
docker exec ollama ollama pull mistral
```

4. **Run pipeline:**

```bash
python main.py
```

## 🐛 Troubleshooting

### Common Issues:

1. **Pinecone connection fails**

   - Verify API key in `.env`
   - Check internet connection
   - Ensure correct environment region

2. **Ollama not responding**

   - Wait for model download to complete
   - Check `docker-compose logs ollama`
   - Verify model is pulled: `docker exec ollama ollama list`

3. **Chrome/Selenium issues**

   - Ensure Chrome is installed in container
   - Check ChromeDriver compatibility
   - Try headless mode for server environments

4. **Memory issues**
   - Reduce batch sizes in `config.py`
   - Limit number of products searched
   - Use smaller embedding models

### Debug Commands:

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs [service-name]

# Access database
docker exec -it graph_postgres psql -U graphuser -d graphdb

# Test Ollama
curl http://localhost:11434/api/tags
```

## 🚨 Important Notes

- **DEMO MODE**: Web automation is in demo mode by default
- **No Real Purchases**: Set `AUTO_ORDER=false` to prevent actual orders
- **Rate Limiting**: Respect website rate limits and terms of service
- **Security**: Never commit real credentials to version control
- **Legal**: Ensure compliance with website terms of service

## 🔮 Advanced Usage

### Custom Data Sources:

```python
# Use your own database table
results = pipeline.run_full_pipeline(
    table_name="your_products",
    text_columns=["name", "description", "features"],
    id_column="product_id"
)
```

### Custom Ordering Logic:

```python
# Modify ordering configuration
ordering_config = OrderingConfig(
    max_budget=1000.0,
    max_items=5,
    min_confidence=0.8,
    preferred_sites=["amazon"],
    auto_order=False
)
```

## 📈 Performance Optimization

- Use GPU for faster GNN training
- Implement caching for repeated queries
- Batch process large datasets
- Use async operations for web scraping

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch Geometric for GNN implementations
- Sentence Transformers for embeddings
- Ollama for local LLM hosting
- Pinecone for vector database services
