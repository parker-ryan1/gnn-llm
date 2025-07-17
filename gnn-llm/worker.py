"""
Background worker for processing heavy pipeline tasks
"""
import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any
import redis
from main import GraphGNNRAGPipeline
from utils.logging_config import setup_logging
from monitoring.metrics import metrics_collector

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class PipelineWorker:
    def __init__(self):
        self.redis_client = redis.from_url("redis://localhost:6379/0")
        self.pipeline = GraphGNNRAGPipeline()
        self.running = True
        
    async def start(self):
        """Start the worker"""
        logger.info("Starting pipeline worker...")
        
        # Start monitoring
        metrics_collector.start_monitoring()
        
        # Initialize pipeline
        try:
            self.pipeline.setup_components()
            logger.info("Pipeline components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return
        
        # Main worker loop
        while self.running:
            try:
                await self.process_jobs()
                await asyncio.sleep(5)  # Check for new jobs every 5 seconds
            except KeyboardInterrupt:
                logger.info("Worker shutdown requested")
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(10)  # Wait before retrying
        
        # Cleanup
        metrics_collector.stop_monitoring()
        logger.info("Worker stopped")
    
    async def process_jobs(self):
        """Process jobs from Redis queue"""
        try:
            # Get job from queue (blocking pop with timeout)
            job_data = self.redis_client.blpop("pipeline_jobs", timeout=5)
            
            if job_data:
                queue_name, job_json = job_data
                job = json.loads(job_json.decode('utf-8'))
                
                logger.info(f"Processing job: {job['job_id']}")
                await self.execute_pipeline_job(job)
                
        except redis.RedisError as e:
            logger.error(f"Redis error: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid job data: {e}")
    
    async def execute_pipeline_job(self, job: Dict[str, Any]):
        """Execute a pipeline job"""
        job_id = job['job_id']
        
        try:
            # Update job status
            self.update_job_status(job_id, "running", "Processing pipeline...")
            
            start_time = time.time()
            
            # Run pipeline
            results = self.pipeline.run_full_pipeline(
                table_name=job['table_name'],
                text_columns=job['text_columns'],
                id_column=job.get('id_column', 'id'),
                query=job.get('query')
            )
            
            duration = time.time() - start_time
            
            # Store results
            self.store_job_results(job_id, results)
            
            # Update job status
            self.update_job_status(
                job_id, 
                "completed", 
                f"Pipeline completed in {duration:.2f}s",
                results
            )
            
            logger.info(f"Job {job_id} completed successfully in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            self.update_job_status(job_id, "failed", f"Error: {str(e)}")
    
    def update_job_status(self, job_id: str, status: str, message: str, results: Dict = None):
        """Update job status in Redis"""
        job_status = {
            "job_id": job_id,
            "status": status,
            "message": message,
            "updated_at": datetime.now().isoformat()
        }
        
        if results:
            job_status["results"] = results
        
        # Store with expiration (24 hours)
        self.redis_client.setex(
            f"job_status:{job_id}",
            86400,  # 24 hours
            json.dumps(job_status)
        )
    
    def store_job_results(self, job_id: str, results: Dict[str, Any]):
        """Store job results separately (they might be large)"""
        # Store with longer expiration (7 days)
        self.redis_client.setex(
            f"job_results:{job_id}",
            604800,  # 7 days
            json.dumps(results, default=str)  # Convert non-serializable objects
        )

async def main():
    """Main worker function"""
    worker = PipelineWorker()
    await worker.start()

if __name__ == "__main__":
    asyncio.run(main())