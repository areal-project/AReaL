import os
import requests
import threading
import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException

from src.typings import LoggerConfig
from src.utils import SingletonLogger


logger_config = LoggerConfig(
    level="DEBUG",
    log_file_path="./outputs/forwarding_app.log",
    logger_name="forwarding_app",
)
logger = SingletonLogger.get_instance(logger_config)

app = FastAPI()

# Configuration: Destination server base URL
DESTINATION_SERVER = os.getenv("DESTINATION_SERVER", "http://destination-server")


@app.api_route("/{path:path}", methods=["POST"])
async def forward_any_post(path: str, request: Request) -> Response:
    """
    Forward any POST request to the Destination Server based on the incoming port.
    """
    # Read the raw request body
    body = await request.body()

    # Copy the original request headers and remove "host" to avoid conflicts
    original_headers = dict(request.headers)
    original_headers.pop("host", None)

    # Determine the incoming port from the request URL
    incoming_port = request.url.port

    # Log the incoming request
    logger.info(f"Received POST request on port {incoming_port}: /{path}")

    # Use the incoming port as the destination port
    destination_port = incoming_port

    # Construct the forwarding URL
    destination_url = f"{DESTINATION_SERVER}:{destination_port}/{path}"
    logger.info(f"Forwarding to {destination_url}")

    try:
        # Forward the request to the Destination Server with a timeout
        response_from_destination = requests.post(
            destination_url, data=body, headers=original_headers, timeout=60  # seconds
        )
        logger.info(
            f"Received response from Destination Server: {response_from_destination.status_code}"
        )
    except requests.exceptions.Timeout:
        logger.error("Destination server timed out")
        raise HTTPException(status_code=504, detail="Destination server timed out")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error reaching destination server: {e}")
        raise HTTPException(status_code=503, detail="Destination server not reachable")

    # Extract response details from the Destination Server
    content = response_from_destination.content
    status_code = response_from_destination.status_code

    # Forward relevant headers, excluding those that can cause conflicts
    response_headers = dict(response_from_destination.headers)
    for h in ["content-encoding", "transfer-encoding", "content-length", "connection"]:
        response_headers.pop(h, None)

    # Log the response status
    logger.info(f"Responding with status {status_code} to client")

    # Return the response exactly as received from the Destination Server
    return Response(content=content, status_code=status_code, headers=response_headers)


def run_forwarding_app(host: str, port: int) -> None:
    """
    Function to run an Uvicorn application on a specified host and port.
    """
    logger.info(f"Starting forwarding_app on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    # Define the host and ports to listen on
    HOST = "0.0.0.0"

    PORTS_ENV = os.getenv(
        "PORTS", "8000,8001,8002,8003"
    )  # Default ports if PORTS not set
    try:
        # Parse the PORTS_ENV string into a list of integers
        PORTS = [
            int(port.strip()) for port in PORTS_ENV.split(",") if port.strip().isdigit()
        ]
        if not PORTS:
            raise ValueError("No valid ports found in PORTS environment variable.")
    except ValueError as e_:
        logger.error(f"Invalid PORTS environment variable: {e_}")
        PORTS = [8000, 8001, 8002, 8003]  # Fallback to default ports

    # Create and start a thread for each Uvicorn application instance
    threads = []
    for port in PORTS:
        thread = threading.Thread(
            target=run_forwarding_app, args=(HOST, port), daemon=True
        )
        thread.start()
        threads.append(thread)
        logger.info(f"Application thread started for port {port}")

    # Keep the main thread alive to allow application threads to run
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        logger.info("Shutting down applications...")
