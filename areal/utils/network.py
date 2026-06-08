# SPDX-License-Identifier: Apache-2.0

import random
import socket
from ipaddress import ip_address


def gethostname():
    return socket.gethostname()


def gethostip(probe_host: str = "8.8.8.8", probe_port: int = 80) -> str:
    """
    Find the local IP address for outbound route to `probe_host:probe_port`.

    Args:
        probe_host: Remote address used to trigger route selection.
        probe_port: Remote port used for the UDP probe.

    Returns:
        The selected local IP address as a string. Supports both IPv4 and IPv6.

    Raises:
        RuntimeError: If no suitable address can be determined
    """
    try:
        hostname = socket.gethostname()
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_DGRAM)
        for family, _, _, _, sockaddr in infos:
            if family == socket.AF_INET:
                ip = sockaddr[0]
                if ip and not ip.startswith("127."):
                    return ip
            elif family == socket.AF_INET6:
                ip = sockaddr[0]
                if ip and ip != "::1":
                    return ip
    except socket.gaierror:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((probe_host, probe_port))
            return sock.getsockname()[0]
    except OSError as e:
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as sock:
                sock.connect(("2001:4860:4860::8888", probe_port))
                ip6 = sock.getsockname()[0]
                if ip6 and ip6 != "::1":
                    return ip6
        except OSError:
            raise RuntimeError("Could not determine host IP") from e


def get_loopback_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
        return "127.0.0.1"
    except OSError:
        pass
    if socket.has_ipv6:
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
                sock.bind(("::1", 0))
            return "::1"
        except OSError:
            pass
    raise RuntimeError("Could not determine loopback IP")


def format_host_for_url(host: str) -> str:
    if host.startswith("[") and host.endswith("]"):
        return host
    if ":" in host:
        return f"[{host}]"
    return host


def format_hostport(host: str, port: int) -> str:
    return f"{format_host_for_url(host)}:{port}"


def split_hostport(addr: str) -> tuple[str, int]:
    if addr.startswith("["):
        end = addr.find("]")
        if end == -1:
            raise ValueError(f"Invalid bracketed address: {addr}")
        host = addr[1:end]
        rest = addr[end + 1 :]
        if not rest.startswith(":"):
            raise ValueError(f"Invalid bracketed address: {addr}")
        host_for_validate = host.split("%", 1)[0]
        try:
            ip_address(host_for_validate)
        except ValueError as e:
            raise ValueError(f"Invalid bracketed address: {addr}") from e
        return host, int(rest[1:])

    if addr.count(":") == 1:
        host, port_s = addr.split(":", 1)
        return host, int(port_s)

    host, port_s = addr.rsplit(":", 1)
    host_for_validate = host.split("%", 1)[0]
    try:
        ip_address(host_for_validate)
    except ValueError as e:
        raise ValueError(f"Invalid host:port address: {addr}") from e
    return host, int(port_s)


def find_free_ports(
    count: int,
    port_range: tuple = (10000, 32767),
    exclude_ports: set[int] | None = None,
) -> list[int]:
    """
    Find multiple free ports within a specified range.

    Args:
        count: Number of free ports to find
        port_range: Tuple of (min_port, max_port) to search within.
            Defaults to (10000, 32767) to avoid the OS ephemeral range
            (typically 32768-60999 on Linux) which causes TOCTOU collisions.
        exclude_ports: Set of ports to exclude from search

    Returns:
        List of free port numbers

    Raises:
        ValueError: If unable to find requested number of free ports
    """
    if exclude_ports is None:
        exclude_ports = set()

    min_port, max_port = port_range
    free_ports = []
    attempted_ports = set()

    # Calculate available port range
    available_range = max_port - min_port + 1 - len(exclude_ports)

    if count > available_range:
        raise ValueError(
            f"Cannot find {count} ports in range {port_range}. "
            f"Only {available_range} ports available."
        )

    max_attempts = count * 10  # Reasonable limit to avoid infinite loops
    attempts = 0

    while len(free_ports) < count and attempts < max_attempts:
        # Generate random port within range
        port = random.randint(min_port, max_port)

        # Skip if port already attempted or excluded
        if port in attempted_ports or port in exclude_ports:
            attempts += 1
            continue

        attempted_ports.add(port)

        if is_port_free(port):
            free_ports.append(port)

        attempts += 1

    if len(free_ports) < count:
        raise ValueError(
            f"Could only find {len(free_ports)} free ports "
            f"out of {count} requested after {max_attempts} attempts"
        )

    return sorted(free_ports)


def is_port_free(port: int) -> bool:
    """
    Check if a port is free by attempting to bind to it.

    Args:
        port: Port number to check

    Returns:
        True if port is free, False otherwise
    """
    # Check TCP
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", port))
    except OSError:
        return False
    finally:
        sock.close()

    # Check UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()
