try:
    # normal import when the wheel exposes the 'ollama' namespace
    import ollama
except ModuleNotFoundError:
    # fallback: some PyPI wheels install as 'ollama_python'
    import importlib
    ollama = importlib.import_module("ollama_python")


def main():
    print("LAN Transcriber placeholder")

if __name__ == "__main__":
    main()
