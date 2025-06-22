#!/usr/bin/env python3
"""
Script to convert notebook content text file into a proper Jupyter notebook.
"""

import json
import re
import nbformat

def create_notebook_from_content(content_file, output_file):
    """Convert text content to Jupyter notebook format."""
    
    # Read the content file
    with open(content_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content by cell markers
    cells_content = content.split('---')
    
    # Initialize notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Process each cell
    for cell_content in cells_content:
        cell_content = cell_content.strip()
        if not cell_content:
            continue
        
        # Check if it's markdown (starts with # and doesn't contain import or print)
        if cell_content.startswith('#') and not any(keyword in cell_content for keyword in ['import ', 'print(', 'plt.', 'sns.', 'df.', 'X_', 'y_', 'model', 'results', 'cv_', 'tuned_', 'advanced_', 'os.makedirs', 'with open', 'pickle.dump', 'json.dump']):
            # Markdown cell
            cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": cell_content
            }
        else:
            # Code cell
            cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell_content
            }
        
        notebook["cells"].append(cell)
    
    # Write the notebook
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"Notebook created successfully: {output_file}")
    print(f"Total cells: {len(notebook['cells'])}")

if __name__ == "__main__":
    # Read the content file
    with open('notebooks/05_rag_agent_content.txt', 'r') as f:
        content = f.read()

    cells = []
    current_cell = []
    cell_type = 'markdown'
    for line in content.splitlines():
        if line.strip().startswith('# %% [markdown]'):
            if current_cell:
                cells.append(nbformat.v4.new_markdown_cell('\n'.join(current_cell)) if cell_type == 'markdown' else nbformat.v4.new_code_cell('\n'.join(current_cell)))
                current_cell = []
            cell_type = 'markdown'
        elif line.strip().startswith('# %%'):
            if current_cell:
                cells.append(nbformat.v4.new_markdown_cell('\n'.join(current_cell)) if cell_type == 'markdown' else nbformat.v4.new_code_cell('\n'.join(current_cell)))
                current_cell = []
            cell_type = 'code'
        else:
            current_cell.append(line)
    if current_cell:
        cells.append(nbformat.v4.new_markdown_cell('\n'.join(current_cell)) if cell_type == 'markdown' else nbformat.v4.new_code_cell('\n'.join(current_cell)))

    nb = nbformat.v4.new_notebook()
    nb['cells'] = cells
    nb['metadata'] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12"
        }
    }

    with open('notebooks/05_rag_agent.ipynb', 'w') as f:
        nbformat.write(nb, f)

    print('Notebook created: notebooks/05_rag_agent.ipynb') 