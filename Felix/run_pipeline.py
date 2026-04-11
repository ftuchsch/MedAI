"""Pipeline entrypoint for Felix."""

from src.config import ensure_directories

def main() -> None:
    ensure_directories()
    print("Project directories are ready.")
    print("Pipeline scaffold initialized.")

if __name__ == "__main__":
    main()