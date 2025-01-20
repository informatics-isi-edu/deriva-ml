import ast
import argparse

def format_docstring(docstring):
    """
    Formats the extracted docstring into Markdown, preserving indentation and sections like params and return.

    :param docstring: The raw docstring text.
    :return: A formatted Markdown string.
    """
    lines = docstring.strip().split("\n")
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith(":param"):
            # Format parameter documentation
            param_parts = line.split(":", 2)
            if len(param_parts) == 3:
                param_name = param_parts[1].strip()
                param_desc = param_parts[2].strip()
                formatted_lines.append(f"- **{param_name}**: {param_desc}")
        elif line.startswith(":return"):
            # Format return documentation
            return_desc = line.split(":", 1)[1].strip()
            formatted_lines.append(f"**Returns**: {return_desc}")
        else:
            # Add regular text lines
            formatted_lines.append(line)

    return "\n".join(formatted_lines)


def extract_docstrings(file_path):
    with open(file_path, "r") as file:
        tree = ast.parse(file.read())

    docstrings = []

    # Extract module-level docstring
    if ast.get_docstring(tree):
        docstrings.append(f"# Module Documentation\n\n{format_docstring(ast.get_docstring(tree))}\n")

    # Extract class and function docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            name = node.name
            docstring = ast.get_docstring(node)
            if docstring:
                header = f"## Class: {name}\n" if isinstance(node, ast.ClassDef) else f"### Function: {name}\n"
                docstrings.append(f"{header}\n{format_docstring(docstring)}\n")

    return "\n".join(docstrings)


def save_to_markdown(docstrings, output_path):
    with open(output_path, "w") as file:
        file.write(docstrings)


if __name__ == "__main__":
    # Input Python file path
    parser = argparse.ArgumentParser(description="Extract docstrings from a Python file and save them as Markdown documentation.")
    parser.add_argument("input_file", help="Path to the input Python file.")
    parser.add_argument("output_file", help="Path to save the output Markdown file.")
    args = parser.parse_args()

    # Extract and save docstrings
    documentation = extract_docstrings(args.input_file)
    save_to_markdown(documentation, args.output_file)
    print(f"Documentation saved to {args.output_file}")


# python doc_gen.py ../src/deriva_ml/deriva_ml_base.py deriva_ml_base.md
# python doc_gen.py ../src/deriva_ml/dataset_bag.py datasetbag.md