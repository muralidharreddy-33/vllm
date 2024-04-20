import re
from pathlib import Path


def fix_case(text: str) -> str:
    subs = [
        ("api", "API"),
        ("llm", "LLM"),
        ("vllm", "vLLM"),
        ("openai", "OpenAI"),
    ]
    for sub in subs:
        text = re.sub(*sub, text, flags=re.IGNORECASE)
    return text


def underline(title: str, character: str = "=") -> str:
    return f"{title}\n{character * len(title)}"


def generate_title(filename: str) -> str:
    # Turn filename into a title
    title = filename.replace("_", " ").title()
    # Handle acronyms and names
    title = fix_case(title)
    # Underline title
    title = underline(title)
    return title


def generate_examples():
    # Source paths
    script_dir = Path("../examples")
    script_paths = sorted(script_dir.glob("*.py"))

    # Destination paths
    doc_dir = Path("source/getting_started/examples")
    doc_paths = [doc_dir / f"{path.stem}.rst" for path in script_paths]

    # Generate the example docs for each example script
    for script_path, doc_path in zip(script_paths, doc_paths):
        # Make script_path relative to doc_path and call it include_path
        include_path = '../../..' / script_path
        content = (
            f"{generate_title(doc_path.stem)}\n\n"
            f".. literalinclude:: {include_path}\n"
            "    :language: python\n"
            "    :linenos:\n"
        ) 
        with open(doc_path, "w+") as f:
            f.write(content)

    # Generate the toctree for the example scripts
    with open(doc_dir / "examples_index.template.rst") as f:
        examples_index = f.read()
    with open(doc_dir / "examples_index.rst", "w+") as f:
        example_docs = "\n   ".join(path.stem for path in script_paths)
        f.write(examples_index.replace(r"%EXAMPLE_DOCS%", example_docs))

