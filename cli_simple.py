# General modules
import sys
import cli

from rich.console import Console
console = Console()

# Main method
if __name__ == "__main__":
    try:
        cli.simple(
            selection=False
        )
    except KeyboardInterrupt:
        print("")
        console.print("\nI hope you found what you were looking for. 👀✨ See you next time! 👋🌟", style="#eb6134")
        print("")
        sys.exit(1)
