import os
import subprocess
import argparse
from typing import Optional

# Define color codes for enhanced output
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
PURPLE = "\033[0;35m"
CYAN = "\033[0;36m"
NC = "\033[0m"  # No Color


def get_current_pid() -> int:
    """Retrieve the current process ID."""
    return os.getpid()


def get_parent_pid(pid: int) -> Optional[int]:
    """Retrieve the parent process ID (PPID) for a given process ID."""
    try:
        process_info = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "ppid="], text=True
        ).strip()
        return int(process_info) if process_info else None
    except subprocess.CalledProcessError:
        return None


def terminate_processes(process_name: str, auto_confirm: bool) -> None:
    print(f"{CYAN}🔧 Starting process termination...{NC}")

    current_pid = get_current_pid()

    # Step 1: List the processes related to the given process name
    try:
        # Use 'pgrep' for more accurate process matching
        processes = subprocess.check_output(["pgrep", "-f", process_name], text=True)
        # Split the output into individual PIDs and convert to integers
        pids = [int(pid) for pid in processes.strip().split("\n") if pid.isdigit()]
    except subprocess.CalledProcessError:
        pids = []

    # Remove the current script's PID and its parent PID from the list to prevent self-termination
    parent_pid = get_parent_pid(current_pid)
    pids = [pid for pid in pids if pid != current_pid and pid != parent_pid]

    # Step 2: Check if any processes were found
    if not pids:
        print(f"{YELLOW}No processes related to '{process_name}' found.{NC}")
    else:
        # Step 3: Display the processes to the user for confirmation (unless auto_confirm is True)
        print(f"{GREEN}Found the following '{process_name}' processes:{NC}")
        for pid in pids:
            try:
                # Retrieve detailed process information
                process_info = subprocess.check_output(
                    ["ps", "-p", str(pid), "-o", "pid,comm,args"], text=True
                ).strip()
                print(process_info)
            except subprocess.CalledProcessError:
                print(f"{YELLOW}Unable to retrieve information for PID {pid}.{NC}")
        print()

        if not auto_confirm:
            # Step 4: Ask for user confirmation if not in auto-confirm mode
            confirmation = input(
                f"{PURPLE}Do you want to kill these processes? (y/n): {NC}"
            )
        else:
            confirmation = "y"  # Automatically confirm if in auto-confirm mode

        # Step 5: If confirmed, kill the processes
        if confirmation.lower() == "y":
            print(f"{BLUE}Terminating processes...{NC}")
            for pid in pids:
                try:
                    subprocess.run(["kill", str(pid)], check=True)
                    print(f"{GREEN}Process with PID {pid} killed successfully.{NC}")
                except subprocess.CalledProcessError:
                    print(f"{RED}Failed to kill process with PID {pid}.{NC}")
            print(f"{GREEN}Processes terminated successfully.{NC}")
        else:
            print(f"{YELLOW}Operation canceled by the user.{NC}")
    print(f"{CYAN}🔧 Process termination completed.{NC}")


def terminate_docker_containers(
    container_id_list: list[str], auto_confirm: bool
) -> None:
    print(f"{CYAN}🐳 Starting Docker container termination...{NC}")

    # Step 1: List Docker containers using '|' as delimiter
    try:
        docker_containers: list[str] = []
        for container_id in container_id_list:
            docker_container = subprocess.check_output(
                [
                    "docker",
                    "ps",
                    "--filter",
                    f"id={container_id}",
                    "--format",
                    "{{.ID}}|{{.Names}}|{{.CreatedAt}}|{{.Status}}|{{.Labels}}",
                ],
                text=True,
            )
            if docker_container.strip():
                docker_containers.append(docker_container.strip())
    except subprocess.CalledProcessError:
        docker_containers = []

    # Step 2: Check if any Docker containers were found
    if not docker_containers:
        print(f"🥵 {YELLOW}No Docker containers found. ")
        return

    # Step 3: Display the Docker containers to the user for confirmation (unless auto_confirm is True)
    print(f"{GREEN}Found Docker:{NC}")
    for container in docker_containers:
        id, name, created, status, labels = container.split("|")
        print(f"  🆔 ID       : {BLUE}{id}{NC}")
        print(f"  🏷️  Name     : {BLUE}{name}{NC}")
        print(f"  📅 Created  : {BLUE}{created}{NC}")
        print(f"  🟢 Status   : {BLUE}{status}{NC}")
        print(f"  🏷️  Labels   : {BLUE}{labels}{NC}")
        print("----------------------------------------")
    print()

    # Step 4: Iterate through each container
    for container in docker_containers:
        id, name, created, status, labels = container.split("|")
        id = (
            id.strip()
        )  # Clean the container ID by removing any whitespace or hidden characters

        print(f"{PURPLE}📦 Container Details:{NC}")
        print(f"  🆔 ID       : {BLUE}{id}{NC}")
        print(f"  🏷️  Name     : {BLUE}{name}{NC}")
        print(f"  📅 Created  : {BLUE}{created}{NC}")
        print(f"  🟢 Status   : {BLUE}{status}{NC}")
        print(f"  🏷️  Labels   : {BLUE}{labels}{NC}")

        # Prompt user for confirmation if not in auto-confirm mode
        if not auto_confirm:
            docker_confirmation = input(
                f"{PURPLE}Do you want to stop and remove this Docker container? (y/n): {NC}"
            )
        else:
            docker_confirmation = "y"  # Automatically confirm if in auto-confirm mode

        if docker_confirmation.lower() == "y":
            print(f"{YELLOW}🔄 Attempting to stop container with ID: '{id}'{NC}")

            # Step 5: Stop and remove the container
            try:
                subprocess.run(
                    ["docker", "stop", id],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"{GREEN}✅ Container {id} stopped successfully.{NC}")

                # Check if the container was automatically removed
                container_exists = subprocess.run(
                    [
                        "docker",
                        "ps",
                        "-a",
                        "--filter",
                        f"id={id}",
                        "--format",
                        "{{.ID}}",
                    ],
                    text=True,
                    capture_output=True,
                )

                if container_exists.stdout.strip():  # Container still exists
                    try:
                        subprocess.run(
                            ["docker", "rm", id],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        print(
                            f"{GREEN}✅ Docker container {id} removed successfully.{NC}"
                        )
                    except subprocess.CalledProcessError:
                        print(f"{RED}❌ Failed to remove Docker container {id}.{NC}")
                else:
                    print(
                        f"{GREEN}✅ Docker container {id} was automatically removed upon stopping.{NC}"
                    )

            except subprocess.CalledProcessError:
                print(f"{RED}❌ Failed to stop Docker container {id}.{NC}")
        else:
            print(f"{YELLOW}⚠️ Operation canceled for container {id}.{NC}")
        print()

    print(f"{CYAN}🐳 Docker container termination completed.{NC}")


def main() -> None:
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Shutdown script for stopping processes and Docker containers."
    )
    parser.add_argument(
        "--process_name",
        type=str,
        default="start_server.py",
        help="The name of the process to terminate (default: 'start_server.py').",
    )
    parser.add_argument(
        "--docker_container_id_list",
        type=str,
        default="",
        help="List of docker container IDs to terminate. Separated by '_'",
    )
    parser.add_argument(
        "--auto_confirm",
        action="store_true",
        help="Automatically confirm the termination of processes and containers without user input.",
    )

    args = parser.parse_args()

    print(f"{PURPLE}🚀 Starting shutdown_server.py script...{NC}")
    print()

    print(f"{PURPLE}🔄 Shutting down processes...{NC}")
    terminate_processes(args.process_name, args.auto_confirm)
    print()

    print(f"{PURPLE}🐳 Shutting down Docker containers...{NC}")
    terminate_docker_containers(
        args.docker_container_id_list.split("_"), args.auto_confirm
    )
    print()

    print(f"{GREEN}✅ shutdown_server.py script completed successfully.{NC}")


if __name__ == "__main__":
    main()
