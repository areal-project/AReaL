from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


class ColorMessage:
    """Utility class for coloring messages with ANSI codes and adding emojis."""

    @staticmethod
    def red(msg: str) -> str:
        return f"{Fore.RED}{msg}{Style.RESET_ALL}"

    @staticmethod
    def green(msg: str) -> str:
        return f"{Fore.GREEN}{msg}{Style.RESET_ALL}"

    @staticmethod
    def cyan(msg: str) -> str:
        return f"{Fore.CYAN}{msg}{Style.RESET_ALL}"

    @staticmethod
    def yellow(msg: str) -> str:
        return f"{Fore.YELLOW}{msg}{Style.RESET_ALL}"

    @staticmethod
    def blue(msg: str) -> str:
        return f"{Fore.BLUE}{msg}{Style.RESET_ALL}"

    @staticmethod
    def magenta(msg: str) -> str:
        return f"{Fore.MAGENTA}{msg}{Style.RESET_ALL}"

    @staticmethod
    def bold(msg: str) -> str:
        return f"{Style.BRIGHT}{msg}{Style.RESET_ALL}"
