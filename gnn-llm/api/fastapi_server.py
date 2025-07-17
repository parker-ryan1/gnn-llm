from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import asyncio
from datetime import datetime
import json

from main import GraphGNNRAGPipeline
from web_automation.intelligent_ordering import OrderingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Graph-GNN-RAG Pipeline API",
    description="API for running graph neural network analysis with RAG and intelligent ordering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None
current_job = None

class PipelineRequest(BaseModel):
    table_name: str
    text_columns: List[str]
    id_column: str = "id"
    query: Optional[str] = None

class OrderingConfigRequest(BaseModel):
    max_budget: float = 500.0
    max_items: int = 3
    min_confidence: float = 0.6
    preferred_sites: List[str] = ["amazon", "ebay"]
    auto_order: bool = False

class JobStatus(BaseModel):
    job_id: str
    status: str  # "running", "completed", "failed"
    progress: float
    message: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup"""
    global pipeline
    try:
        pipeline = GraphGNNRAGPipeline()
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Graph-GNN-RAG Pipeline API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    health_status = {
        "api": "healthy",
        "pipeline": "unknown",
        "database": "unknown",
        "ollama": "unknown",
        "pinecone": "unknown"
    }
    
    if pipeline:
        health_status["pipeline"] = "initialized"
        
        # Check database connection
        try:
            if pipeline.db_connector:
                tables = pipeline.db_connector.get_all_tables()
                health_status["database"] = f"connected ({len(tables)} tables)"
        except:
            health_status["database"] = "disconnected"
        
        # Check Ollama
        try:
            if pipeline.rag_system:
                response = pipeline.rag_system.query_ollama("test", "")
                health_status["ollama"] = "connected"
        except:
            health_status["ollama"] = "disconnected"
    
    return health_status

@app.post("/pipeline/run")
async def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Run the complete pipeline"""
    global current_job
    
    if current_job and current_job["status"] == "running":
        raise HTTPException(status_code=409, detail="Pipeline is already running")
    
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    current_job = {
        "job_id": job_id,
        "status": "running",
        "progress": 0.0,
        "message": "Starting pipeline...",
        "started_at": datetime.now(),
        "completed_at": None,
        "results": None
    }
    
    # Run pipeline in background
    background_tasks.add_task(
        run_pipeline_background,
        job_id,
        request.table_name,
        request.text_columns,
        request.id_column,
        request.query
    )
    
    return {"job_id": job_id, "status": "started"}

async def run_pipeline_background(job_id: str, table_name: str, text_columns: List[str], 
                                id_column: str, query: Optional[str]):
    """Run pipeline in background"""
    global current_job
    
    try:
        current_job["message"] = "Setting up components..."
        current_job["progress"] = 10.0
        
        pipeline.setup_components()
        
        current_job["message"] = "Loading and processing data..."
        current_job["progress"] = 30.0
        
        df, embeddings = pipeline.load_and_process_data(
            query=query,
            table_name=table_name,
            text_columns=text_columns,
            id_column=id_column
        )
        
        current_job["message"] = "Building graph and training GNN..."
        current_job["progress"] = 60.0
        
        nx_graph, pyg_data, optimized_embeddings, losses = pipeline.build_and_train_graph(embeddings)
        
        current_job["message"] = "Running RAG analysis..."
        current_job["progress"] = 80.0
        
        rag_results = pipeline.run_rag_analysis()
        
        current_job["message"] = "Running intelligent ordering..."
        current_job["progress"] = 90.0
        
        pipeline_results = {
            'data': df.to_dict('records'),
            'graph_metrics': {
                'nodes': nx_graph.number_of_nodes(),
                'edges': nx_graph.number_of_edges()
            },
            'training_losses': losses,
            'rag_results': rag_results
        }
        
        ordering_results = pipeline.run_intelligent_ordering(pipeline_results)
        pipeline_results['ordering_results'] = ordering_results
        
        current_job["status"] = "completed"
        current_job["progress"] = 100.0
        current_job["message"] = "Pipeline completed successfully"
        current_job["completed_at"] = datetime.now()
        current_job["results"] = pipeline_results
        
    except Exception as e:
        current_job["status"] = "failed"
        current_job["message"] = f"Pipeline failed: {str(e)}"
        current_job["completed_at"] = datetime.now()
        logger.error(f"Pipeline failed: {e}")

@app.get("/pipeline/status/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if not current_job or current_job["job_id"] != job_id:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return current_job

@app.get("/pipeline/results/{job_id}")
async def get_job_results(job_id: str):
    """Get job results"""
    if not current_job or current_job["job_id"] != job_id:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if current_job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    return current_job["results"]

@app.post("/rag/query")
async def rag_query(query: str):
    """Direct RAG query"""
    if not pipeline or not pipeline.rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        result = pipeline.rag_system.rag_query(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@app.get("/database/tables")
async def list_tables():
    """List available database tables"""
    if not pipeline or not pipeline.db_connector:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        tables = pipeline.db_connector.get_all_tables()
        return {"tables": tables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tables: {str(e)}")

@app.get("/database/schema/{table_name}")
async def get_table_schema(table_name: str):
    """Get table schema"""
    if not pipeline or not pipeline.db_connector:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        schema = pipeline.db_connector.get_table_schema(table_name)
        return {"table": table_name, "schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schema: {str(e)}")

@app.post("/ordering/search")
async def search_products(query: str, site: str = "amazon"):
    """Search for products"""
    if not pipeline or not pipeline.ordering_system:
        raise HTTPException(status_code=503, detail="Ordering system not available")
    
    try:
        products = pipeline.ordering_system.web_scraper.search_products(query, site)
        return {
            "query": query,
            "site": site,
            "products": [
                {
                    "name": p.name,
                    "price": p.price,
                    "url": p.url,
                    "rating": p.rating,
                    "availability": p.availability
                }
                for p in products
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Product search failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)