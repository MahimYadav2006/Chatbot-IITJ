import os
import ast

def clean_signature(sig_text):
    """Strip trailing comment lines, empty lines, or docstring starts from a signature block."""
    lines = sig_text.splitlines()
    while lines:
        last_line = lines[-1].strip()
        if not last_line or last_line.startswith('#') or last_line.startswith('"""') or last_line.startswith("'''"):
            lines.pop()
        else:
            break
    return "\n".join(lines)

def extract_signature(lines, node):
    """Extract signature lines of a class or function from source lines."""
    start_idx = node.lineno - 1
    if not node.body:
        return lines[start_idx].strip()
        
    end_idx = node.body[0].lineno - 1
    
    sig_lines = lines[start_idx:end_idx]
    if not sig_lines:
        sig_lines = [lines[start_idx]]
        
    sig_text = "".join(sig_lines)
    return clean_signature(sig_text)

def parse_python_file(filepath, rel_path):
    """Parse a python file and return structured class/function/docstring info."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        lines = source.splitlines(keepends=True)
        tree = ast.parse(source)
    except Exception as e:
        return f"### {rel_path}\n*Error parsing file: {e}*\n\n"

    # Get module docstring
    module_doc = ast.get_docstring(tree)
    module_doc_str = f"> {module_doc.strip().replace(chr(10), chr(10) + '> ')}\n\n" if module_doc else ""

    content = f"### [{rel_path}](file://{filepath})\n"
    if module_doc_str:
        content += module_doc_str
        
    # Track top level classes and functions
    classes = []
    functions = []
    
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_sig = extract_signature(lines, node)
            class_info = {
                "sig": class_sig,
                "docstring": ast.get_docstring(node),
                "methods": []
            }
            for subnode in node.body:
                if isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_sig = extract_signature(lines, subnode)
                    method_doc = ast.get_docstring(subnode)
                    class_info["methods"].append((method_sig, method_doc))
            classes.append(class_info)
            
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_sig = extract_signature(lines, node)
            func_doc = ast.get_docstring(node)
            functions.append((func_sig, func_doc))
            
    if not classes and not functions:
        content += "*No classes or functions defined.*\n\n"
        return content

    if classes:
        for c in classes:
            content += f"#### `{c['sig'].splitlines()[0]}`\n"
            if len(c['sig'].splitlines()) > 1:
                content += "```python\n" + c['sig'] + "\n```\n"
            if c['docstring']:
                content += f"  > {c['docstring'].strip().replace(chr(10), chr(10) + '  > ')}\n"
            if c['methods']:
                content += "  ```python\n"
                for m_sig, m_doc in c['methods']:
                    # Indent method signature
                    indented_sig = "\n".join("  " + l for l in m_sig.splitlines())
                    content += f"{indented_sig}\n"
                    if m_doc:
                        doc_lines = m_doc.strip().split('\n')
                        if len(doc_lines) == 1:
                            content += f'      """{doc_lines[0]}"""\n'
                        else:
                            content += '      """\n'
                            for line in doc_lines:
                                content += f'      {line}\n'
                            content += '      """\n'
                content += "  ```\n"
            content += "\n"
            
    if functions:
        content += "#### Top-level Functions\n"
        content += "```python\n"
        for f_sig, f_doc in functions:
            content += f"{f_sig}\n"
            if f_doc:
                doc_lines = f_doc.strip().split('\n')
                if len(doc_lines) == 1:
                    content += f'    """{doc_lines[0]}"""\n'
                else:
                    content += '    """\n'
                    for line in doc_lines:
                        content += f'    {line}\n'
                    content += '    """\n'
        content += "```\n\n"
        
    return content

def generate_directory_tree(startpath, exclude_dirs):
    """Generate a clean text-based directory tree."""
    tree_lines = []
    
    def walk(directory, prefix=""):
        try:
            items = sorted(os.listdir(directory))
        except OSError:
            return
            
        # Filter items
        filtered_items = []
        for item in items:
            if item in exclude_dirs:
                continue
            if item.startswith('.') or item == '__pycache__' or item == '.pytest_cache':
                if item != '.env.example' and item != '.env':
                    continue
            filtered_items.append(item)
            
        for i, item in enumerate(filtered_items):
            path = os.path.join(directory, item)
            is_last = (i == len(filtered_items) - 1)
            connector = "└── " if is_last else "├── "
            
            tree_lines.append(f"{prefix}{connector}{item}")
            
            if os.path.isdir(path):
                extension_prefix = "    " if is_last else "│   "
                walk(path, prefix + extension_prefix)
                
    tree_lines.append(os.path.basename(startpath.rstrip('/')))
    walk(startpath)
    return "\n".join(tree_lines)

def main():
    workspace_dir = "/home/c3i/chatbot"
    output_file = "/home/c3i/chatbot/repo_map.md"
    
    exclude_dirs = {
        'venv', '.venv', '.git', '.pytest_cache', '__pycache__', 
        'scraped_data', 'data', '.gemini', '.codex', '.agents', 'static', 'templates'
    }
    
    # 1. Generate Directory Tree
    tree_str = generate_directory_tree(workspace_dir, exclude_dirs)
    
    # 2. Walk and parse files
    files_info = []
    
    # Walk the directories
    all_files = []
    for root, dirs, files in os.walk(workspace_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, workspace_dir)
                all_files.append((rel_path, filepath))
                
    # Sort files by path depth and name
    all_files.sort(key=lambda x: (x[0].count(os.sep), x[0]))
    
    for rel_path, filepath in all_files:
        print(f"Parsing: {rel_path}")
        files_info.append(parse_python_file(filepath, rel_path))
        
    # 3. Assemble full Markdown content
    md_content = f"""# IIT Jammu Chatbot Repository Map

This is a detailed repository map of the chatbot codebase, designed for LLM agents to quickly understand the structure, classes, and function signatures without having to read the full contents of every file.

## 📁 Repository Directory Structure
```text
{tree_str}
```

---

## 🛠️ Modules & Signatures

"""
    md_content += "\n".join(files_info)
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    print(f"Successfully generated repo map at {output_file}")

if __name__ == "__main__":
    main()
