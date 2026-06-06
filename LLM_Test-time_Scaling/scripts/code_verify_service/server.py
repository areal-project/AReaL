"""HTTP API server for C++ code verification service."""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root and src to path to import evaluation modules
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Error: fastapi and uvicorn are required. Install with: pip install fastapi uvicorn")
    sys.exit(1)


from src.evaluation.cpp_runner import run_cpp_tests_with_checker


app = FastAPI(title="Code Verify Service", version="1.0.0")


class VerifyRequest(BaseModel):
    """Request model for code verification."""
    
    code: str
    problem_id: str
    data_dir: Optional[str] = None
    compile_timeout: int = 30


class VerifyResponse(BaseModel):
    """Response model for code verification."""
    
    success: bool
    results: list[bool]
    passed: int
    total: int
    is_correct: bool
    score: float
    feedback: str
    metadata: Dict[str, Any]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "code_verify"}


@app.post("/verify", response_model=VerifyResponse)
async def verify_code(request: VerifyRequest) -> VerifyResponse:
    """Verify C++ code against test cases.
    
    This endpoint accepts requests and processes them. If the server is busy,
    requests will be queued and processed when workers become available.
    The connection will be kept alive during queuing and processing.
    
    Args:
        request: VerifyRequest containing code, problem_id, and optional data_dir
        
    Returns:
        VerifyResponse with evaluation results
    """
    try:
        # Get data directory
        default_locations = [
            project_root / "data" / "local_data" / "lcb_testcases" / "data",
            project_root / "data" / "benchmarks" / "lcb_testcases" / "data",
            Path("path-to-results/llm_test_time_scaling/data/local_data/lcb_testcases/data"),
            Path("/storage/openpsi/users/yl/AReaL/LLM_Test-time_Scaling/data/lcb_testcases"),
        ]
        
        if request.data_dir is None:
            # Try default locations
            data_dir = None
            for default_path in default_locations:
                if default_path.exists():
                    data_dir = str(default_path)
                    break
        else:
            data_dir = request.data_dir
        
        if data_dir is None or not Path(data_dir).exists():
            error_msg = f"Data directory not found."
            if request.data_dir:
                error_msg += f" Provided path: {request.data_dir} does not exist."
            else:
                error_msg += " No data_dir provided and no default location found."
                error_msg += f" Checked locations: {[str(loc) for loc in default_locations]}"
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
        
        # Run evaluation in executor to avoid blocking
        loop = asyncio.get_event_loop()
        results, metadata = await loop.run_in_executor(
            None,
            run_cpp_tests_with_checker,
            request.code,
            data_dir,
            request.problem_id,
            request.compile_timeout,
        )
        
        if not results:
            return VerifyResponse(
                success=False,
                results=[],
                passed=0,
                total=0,
                is_correct=False,
                score=0.0,
                feedback=metadata.get("error_message", "Evaluation failed"),
                metadata=metadata,
            )
        
        # Calculate metrics
        passed = sum(1 for r in results if r is True)
        total = len(results)
        is_correct = all(r is True for r in results)
        score = passed / total if total > 0 else 0.0
        
        feedback = f"Passed {passed}/{total} test cases"
        if metadata:
            if "error_message" in metadata:
                feedback += f"\nError: {metadata['error_message']}"
            if "error" in metadata:
                feedback += f"\nError details: {metadata['error']}"
        
        return VerifyResponse(
            success=True,
            results=results,
            passed=passed,
            total=total,
            is_correct=is_correct,
            score=score,
            feedback=feedback,
            metadata=metadata,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error in verify_code: {error_detail}")
        print(traceback_str)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {error_detail}"
        )


def main():
    """Main entry point for the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Code Verify Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument(
        "--timeout-keep-alive",
        type=int,
        default=3600,
        help="Keep-alive timeout in seconds. Increase this to allow longer queuing times. Default: 3600 (1 hour)"
    )
    parser.add_argument(
        "--timeout-graceful-shutdown",
        type=int,
        default=120,
        help="Graceful shutdown timeout in seconds. Default: 120 (2 minutes)"
    )
    parser.add_argument(
        "--limit-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent connections. Default: None (unlimited)"
    )
    
    args = parser.parse_args()
    
    print(f"Starting Code Verify Service on {args.host}:{args.port}")
    print(f"Workers: {args.workers}")
    print(f"Keep-alive timeout: {args.timeout_keep_alive} seconds (allows long queuing)")
    print(f"Backlog: 2048 (allows up to 2048 queued connections)")
    if args.limit_concurrency:
        print(f"Concurrency limit: {args.limit_concurrency}")
    print("\nNote: The server will keep connections alive during queuing.")
    print("      Clients can wait indefinitely for their requests to be processed.")
    
    # Configure uvicorn with settings to handle queuing
    uvicorn_config = {
        "app": app,
        "host": args.host,
        "port": args.port,
        "workers": args.workers,
        "timeout_keep_alive": args.timeout_keep_alive,
        "timeout_graceful_shutdown": args.timeout_graceful_shutdown,
        "limit_concurrency": args.limit_concurrency,
        "backlog": 2048,  # Increase backlog to allow more queued connections
        "log_level": "info",
    }
    
    # Set timeout_keep_alive to allow long queuing times
    # This prevents connections from being closed while requests are queued
    # The server will keep connections alive even during long processing/queuing
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
