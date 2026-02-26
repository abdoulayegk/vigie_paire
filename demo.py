import os
import json
import re

import opendataloader_pdf


def remove_footnotes(text: str) -> str:
    """
    Remove footnotes and normalize text.
    Removes superscript numbers, asterisks, and other footnote markers.

    :param text: Text to normalize
    :return: Normalized text
    """
    if not text:
        return ""

    # Remove superscript numbers (common footnote markers)
    text = re.sub(r'\d+$', '', text)

    # Remove asterisks and other common footnote markers
    text = re.sub(r'[*†‡§¶#]+', '', text)

    # Remove parenthetical numbers like (1), (2), etc.
    text = re.sub(r'\(\d+\)', '', text)

    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_first_column_from_tables(json_data: dict, page_ranges: list | None = None) -> list:
    """
    Extract the first column from all tables in the JSON data.

    :param json_data: Parsed JSON data from opendataloader_pdf
    :param page_ranges: List of tuples with (start_page, end_page) ranges to extract.
                        If None, extracts from all pages.
    :return: List of tables with only first column data
    """
    tables_first_column = []

    for item in json_data.get('kids', []):
        if item.get('type') == 'table':
            page_number = item.get('page number')

            # Check if table is within specified page ranges
            if page_ranges:
                in_range = False
                for start_page, end_page in page_ranges:
                    if start_page <= page_number <= end_page:
                        in_range = True
                        break

                # Skip tables outside the specified ranges
                if not in_range:
                    continue

            table_info = {
                'table_id': item.get('id'),
                'page_number': page_number,
                'number_of_rows': item.get('number of rows'),
                'first_column_data': []
            }

            # Extract first column from each row
            for row in item.get('rows', []):
                for cell in row.get('cells', []):
                    # Only process cells in column 1
                    if cell.get('column number') == 1:
                        cell_content = []

                        # Extract text from all kids in the cell
                        for kid in cell.get('kids', []):
                            if kid.get('type') in ['paragraph', 'heading']:
                                content = kid.get('content', '')
                                # Normalize and remove footnotes
                                normalized_content = remove_footnotes(content)
                                if normalized_content:
                                    cell_content.append(normalized_content)

                        if cell_content:
                            table_info['first_column_data'].append({
                                'row_number': cell.get('row number'),
                                'content': ' '.join(cell_content)
                            })

            if table_info['first_column_data']:
                tables_first_column.append(table_info)

    return tables_first_column


def save_first_columns_to_json(tables_data: list, output_path: str):
    """
    Save extracted first column data to JSON.

    :param tables_data: List of tables with first column data
    :param output_path: Path to save the JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tables_data, f, ensure_ascii=False, indent=2)

    print(f"First columns JSON saved to: {output_path}")


def save_first_columns_to_markdown(tables_data: list, output_path: str):
    """
    Save extracted first column data to Markdown.

    :param tables_data: List of tables with first column data
    :param output_path: Path to save the Markdown file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Extracted First Columns from Tables\n\n")

        for table in tables_data:
            f.write(f"## Table {table['table_id']} (Page {table['page_number']})\n\n")
            f.write(f"**Number of rows:** {table['number_of_rows']}\n\n")

            for row_data in table['first_column_data']:
                f.write(f"- **Row {row_data['row_number']}:** {row_data['content']}\n")

            f.write("\n---\n\n")

    print(f"First columns Markdown saved to: {output_path}")


def convert_report_to_json(input_pdf_path: str, output_dir: str):
    """
    Convert a PDF report to JSON using opendataloader_pdf.

    :param input_pdf_path: Path to the input PDF file
    :param output_dir: Directory where outputs will be saved
    """

    # Vérification que le fichier existe
    if not os.path.exists(input_pdf_path):
        raise FileNotFoundError(f"Input file not found: {input_pdf_path}")

    # Création du dossier de sortie si nécessaire
    os.makedirs(output_dir, exist_ok=True)

    # Conversion
    opendataloader_pdf.convert(
        input_path=[input_pdf_path],
        output_dir=output_dir,
        format="json,markdown",
    )

    print(f"Conversion completed. JSON saved in: {output_dir}")


if __name__ == "__main__":
    input_pdf = "/Users/balde/Desktop/vigie_paire/data/bmo/t1_report_q1.pdf"
    output_directory = "/Users/balde/Desktop/vigie_paire/data/bmo/"

    # Define page ranges to extract (start_page, end_page)
    page_ranges = [
        (20, 23),  # Pages 20-23
        (39, 53)   # Pages 39-53
    ]

    # Step 1: Convert PDF to JSON and Markdown
    convert_report_to_json(input_pdf, output_directory)

    # Step 2: Extract first column from tables in specific page ranges
    base_name = os.path.splitext(os.path.basename(input_pdf))[0]
    json_path = os.path.join(output_directory, f"{base_name}.json")

    # Load the generated JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Extract first columns from tables within specified page ranges
    tables_first_column = extract_first_column_from_tables(json_data, page_ranges=page_ranges)

    # Step 3: Save first columns to JSON and Markdown
    first_col_json_path = os.path.join(output_directory, f"{base_name}_first_columns.json")
    first_col_md_path = os.path.join(output_directory, f"{base_name}_first_columns.md")

    save_first_columns_to_json(tables_first_column, first_col_json_path)
    save_first_columns_to_markdown(tables_first_column, first_col_md_path)

    print("\n✓ Processing complete!")
    print(f"  - Page ranges: {page_ranges}")
    print(f"  - Tables found: {len(tables_first_column)}")
    print(f"  - Full JSON: {json_path}")
    print(f"  - First columns JSON: {first_col_json_path}")
    print(f"  - First columns Markdown: {first_col_md_path}")
