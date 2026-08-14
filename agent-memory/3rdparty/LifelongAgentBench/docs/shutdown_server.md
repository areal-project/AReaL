## Shutting Down the Server

To gracefully shut down the server and remove Docker containers created by this repository, use the provided shutdown script:

```bash
python ./src/distributed_deployment_utils/shutdown_server.py --process_name start_server.py --auto_confirm
```

This script will find and terminate the processes related to `start_server.py`.