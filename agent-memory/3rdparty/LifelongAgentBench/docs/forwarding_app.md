# FastAPI Request Forwarding Service

This directory contains a FastAPI application that forwards any POST request to a specified destination server.

## Files

- `./src/forwarding_app/main.py`: The main FastAPI application that handles incoming POST requests and forwards them to the destination server.

## Setup

1. Install the required dependencies:
    ```sh
    pip install fastapi
    pip install uvicorn
    pip install requests
    ```

2. Set up the n2n network:
    ```sh
    # This machine is the supernode
    sudo supernode -p 7654
    # This machine is also an edge with IP 192.168.100.2
    sudo edge -a 192.168.100.2 -c community-zjh -k mypassword-zjh -l 39.105.154.46:7654
    ```

3. Start the FastAPI application:
    ```sh
    bash ./scripts/shell/forwarding_app.sh
    ```

## Usage

Send any POST request to the FastAPI application, and it will forward the request to the destination server specified in .