""" 
FastAPI Wrapper for Domain Comparable Search Agent
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn
from datetime import datetime
import logging
import sys

from src.agent.graph import create_agent_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title= "Domain Comparable Search API",
    description =" AI-powered domain comparable search using LangGraph",
    version = "1.0.0"
)

# Enable CORS(adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize the LangGraph agent
agent_graph = None

class DomainSearchRequest(BaseModel):
    domain: str = Field(..., description="Domain name to search for comparables for", example="onepay.ai")
    
    class Config:
        json_schema_extra = {
            "example": {
                "domain": "onepay.ai"
            }
        }

class DomainSearchResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup"""
    global agent_graph
    try:
        logger.info("=" * 60)
        logger.info("🚀 STARTING DOMAIN COMPARABLE AGENT API")
        logger.info("=" * 60)
        
        # Verify ChromaDB files exist
        import os
        chroma_path = "./chroma_db"
        sqlite_path = f"{chroma_path}/chroma.sqlite3"
        
        logger.info(f"Checking ChromaDB at: {chroma_path}")
        
        if not os.path.exists(chroma_path):
            logger.error(f"❌ ChromaDB directory not found: {chroma_path}")
            raise FileNotFoundError(f"ChromaDB directory missing")
        
        if not os.path.exists(sqlite_path):
            logger.error(f"❌ ChromaDB database not found: {sqlite_path}")
            logger.error("💡 Hint: Check if .dockerignore excludes *.sqlite3 files")
            raise FileNotFoundError(f"ChromaDB database missing")
        
        db_size = os.path.getsize(sqlite_path) / (1024 * 1024)  # MB
        logger.info(f"✅ ChromaDB database found ({db_size:.2f} MB)")
        
        logger.info("Step 1: Creating agent graph...")
        agent_graph = create_agent_graph()
        
        logger.info("Step 2: Validating ChromaDB contents...")
        from src.enrichment.retrieval.chroma_client import ChromaClient
        chroma = ChromaClient()
        doc_count = chroma.collection.count()
        
        logger.info(f"✅ ChromaDB loaded successfully")
        logger.info(f"✅ Total documents: {doc_count:,}")
        
        if doc_count == 0:
            logger.error("❌ ChromaDB is empty (0 documents)")
            raise ValueError("ChromaDB contains no documents - run build_and_export_embeddings.py first")
        
        logger.info("=" * 60)
        logger.info("🎉 API IS READY TO ACCEPT REQUESTS")
        logger.info(f"🔗 Access at: http://localhost:8000")
        logger.info(f"📚 Docs at: http://localhost:8000/docs")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ AGENT INITIALIZATION FAILED!")
        logger.error(f"❌ Error Type: {type(e).__name__}")
        logger.error(f"❌ Error Message: {str(e)}")
        logger.error("=" * 60)
        import traceback
        logger.error(traceback.format_exc())
        raise
# API Endpoints



@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Domain Comparable Search API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "agent_loaded": agent_graph is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/search", response_model=DomainSearchResponse)
async def search_comparables(request: DomainSearchRequest):
    """
    Search for comparable domains
    
    - **domain**: The domain name to find comparables for (e.g., "onepay.ai")
    
    Returns comparable domains with pricing, categories, and similarity scores.
    """
    try:
        if agent_graph is None:
            logger.error("❌ Agent not initialized - cannot process request")
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        # Validate input
        if not request.domain or len(request.domain.strip()) == 0:
            raise HTTPException(status_code=400, detail="Domain cannot be empty")
        
        logger.info(f"📥 Processing request for domain: {request.domain}")
        
        # Run the agent
        initial_state = {
            "input_domain": request.domain.strip()
        }
        
        result = agent_graph.invoke(initial_state)
        
        # Check for errors in result
        if result.get("error"):
            logger.warning(f"⚠️ Agent returned error for {request.domain}: {result.get('error')}")
            return DomainSearchResponse(
                success=False,
                data=None,
                error=result["error"],
                timestamp=datetime.utcnow().isoformat()
            )
        
        # Extract the result
        output = result.get("result", {})
        
        logger.info(f"✅ Successfully processed {request.domain} - found {output.get('total_comparables', 0)} comparables")
        
        return DomainSearchResponse(
            success=True,
            data=output,
            error=None,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ API ERROR for {request.domain}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/batch-search")
async def batch_search_comparables(domains: list[str], background_tasks: BackgroundTasks):
    """
    Batch search for multiple domains (async processing)
    
    - **domains**: List of domain names to search
    
    Returns a job ID to track the batch processing.
    """
    if not domains or len(domains) == 0:
        raise HTTPException(status_code=400, detail="Domains list cannot be empty")
    
    # TODO: Implement async batch processing with job queue
    return {
        "message": "Batch processing not yet implemented",
        "domains_count": len(domains),
        "status": "pending"
    }

if __name__ == "__main__":
    # Run the API Server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level = "info"
    )