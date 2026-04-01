# Python Imports
import os
import re
import json
import shutil
import torch
from bs4 import BeautifulSoup
from transformers import AutoModel, AutoTokenizer
from pdf2image import convert_from_path

# -- Environment Config --
# Hide all GPUs except device 0 to control which GPU is used
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Name of the OCR model to load from HuggingFace Hub
model_name = 'deepseek-ai/DeepSeek-OCR'

# Base path to the input and output folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# INPUT AND OUTPUT PATH
input_dir  = os.path.join(BASE_DIR, "uploads")
output_dir = os.path.join(BASE_DIR, "result")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


# Funciton to cleanup the folder
def cleanup_output_folder(folder_path):

    # Print a message indicating which folder is being cleaned up
    print(f"--- Limpando arquivos residuais em: {folder_path} ---")

    # Iterate over every item inside the output folder
    for item in os.listdir(folder_path):

        # Build the full absolute path for the current item
        item_path = os.path.join(folder_path, item)

        try:

            # Check whether the current item 
            if os.path.isfile(item_path):
 
                os.remove(item_path)

            # Check whether the current item is a directory
            elif os.path.isdir(item_path):

                # Recursively delete the entire subdirectory and its contents
                shutil.rmtree(item_path)

        except Exception as e:

            # Print the error and continue cleaning the remaining items
            print(f"Erro ao deletar {item_path}: {e}")


def get_files_queue(input_folder):
    
    # Define the file extensions that are accepted as valid input
    valid_extensions = ('.pdf', '.png', '.jpeg', '.jpg')

    # Build a list of absolute paths for every valid file in the input folder
    queue = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(valid_extensions)
    ]

    # Return the list sorted alphabetically for deterministic processing order
    return sorted(queue)


def prepare_pages(file_path, temp_folder):

    # Initialize an empty list to collect the image paths for this file
    image_paths = []

    # Check if the file is a PDF and needs to be converted page-by-page
    if file_path.lower().endswith('.pdf'):

        print(f" -> Convertendo PDF: {os.path.basename(file_path)}")

        # Render every PDF page as a PIL image at 300 DPI for high quality
        pdf_pages = convert_from_path(file_path, dpi=300)

        # Strip the .pdf extension to use as a base name for temp image files
        base_name = os.path.basename(file_path).replace('.pdf', '')

        # Save each rendered page as a PNG file in the temp folder
        for i, page_img in enumerate(pdf_pages):

            # Build the output path using 1-based page numbering
            temp_path = os.path.join(temp_folder, f"{base_name}_page_{i + 1}.png")

            # Write the PIL image to disk as PNG
            page_img.save(temp_path, "PNG")

            # Add the saved page path to the results list
            image_paths.append(temp_path)

    else:

        # For image files, use the original path directly without conversion
        image_paths.append(file_path)

    # Return the list of image paths (one per page for PDFs, one for images)
    return image_paths


def create_master_list(input_folder, temp_folder):

    # Initialize the master list that will hold all page image paths
    master_list = []

    # Iterate over every file returned by the queue builder
    for f in get_files_queue(input_folder):

        # Expand each file into its individual page images and append them all
        master_list.extend(prepare_pages(f, temp_folder))

    # Return the flat list of all page image paths ready for OCR
    return master_list


# ──────────────────────────────────────────────
# HTML table parser → structured dict
# ──────────────────────────────────────────────

def parse_html_table(table_tag):

    """
    Converts an HTML <table> into a key→value dictionary.

    Automatically detects two opposite patterns found in the documents:

    ┌─ INLINE PATTERN (keys ending with ':') ────────────────────────────────┐
    │ Each row contains side-by-side pairs. Keys end with ':'.               │
    │ Ex: Alvará Sanitário                                                   │
    │   Tipo de Tributação: | TRIBUTÁVEL | Informação: | INICIAL             │
    │   Razão Social:       | DM AMBIENTAL LTDA                              │
    │   CPF/CNPJ:           | 18.628…    | Insc. Municipal: | 203239648      │
    │                                                                         │
    │ Discriminator: ≥50% of rows start with a cell ending in ':'           │
    └────────────────────────────────────────────────────────────────────────┘

    ┌─ ALTERNATING PATTERN (headers without ':') ────────────────────────────┐
    │ Even rows = labels, odd rows = values.                                 │
    │ Ex: Alvará de Funcionamento                                            │
    │   CPF/CNPJ | Área  | Porte              | Horário     ← labels        │
    │   18.628…  | 171m² | Microempresa (ME…) | 07:00-17:00 ← data         │
    │                                                                         │
    │ Discriminator: has_explicit_keys=False AND ≥1 label/data pair found   │
    └────────────────────────────────────────────────────────────────────────┘
    """

    # Collect all <tr> rows from the table element
    rows = table_tag.find_all('tr')

    # Build a 2-D list of stripped cell texts, skipping completely empty rows
    all_cells = []
    for row in rows:

        cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]

        if cells:

            all_cells.append(cells)

    # Return an empty dict immediately if the table contains no usable cells
    if not all_cells:
        return {}

    # Pre-compiled regex patterns that positively identify concrete data values
    VALUE_PATTERNS = [

        # Brazilian CPF / CNPJ format  e.g. 18.628.123/0001-45
        re.compile(r'^\d{2,3}\.\d{3}\.\d{3}[\/\-][\d\-]+$'),

        # Time values like 07:00 or 17:30
        re.compile(r'^\d{1,2}:\d{2}'),

        # Portuguese month names used in date strings
        re.compile(
            r'\b(janeiro|fevereiro|março|abril|maio|junho|julho'
            r'|agosto|setembro|outubro|novembro|dezembro)\b', re.I),

        # Hexadecimal strings (e.g. document hash codes)
        re.compile(r'^[A-F0-9]{8,}$'),

        # Area measurements like 171m² or 50m2
        re.compile(r'^\d+\s*m[²2]$', re.I),

        # Long numeric sequences such as registration or protocol numbers
        re.compile(r'^\d{5,}$'),
    ]

    def is_concrete_value(text):

        # Strip whitespace before testing
        t = text.strip()

        # A concrete value must be non-empty and match at least one pattern
        return bool(t) and any(pat.search(t) for pat in VALUE_PATTERNS)

    def is_label(text):

        # Remove surrounding whitespace and trailing colon before evaluating
        t = text.strip().rstrip(':')

        # Empty strings are never labels
        if not t:
            return False
        
        # Concrete data values are not labels
        if is_concrete_value(t):
            return False
        
        # Cells that contain only digits, dots, commas, or spaces are not labels
        if re.match(r'^[\d.,\s]+$', t):
            return False
        
        # Cells that start with a long numeric sequence are not labels
        if re.match(r'^\d{6,}', t):
            return False
        
        # Very long strings are unlikely to be column/field labels
        if len(t) > 60:
            return False
        
        # Everything that passes the above filters is treated as a label
        return True

    def is_label_row(row):

        # Ignore blank cells and check that all remaining cells qualify as labels
        non_empty = [c for c in row if c.strip()]
        return bool(non_empty) and all(is_label(c) for c in non_empty)

    def has_concrete_value(row):

        # Return True if at least one non-blank cell in the row is a concrete value
        return any(is_concrete_value(c) for c in row if c.strip())

    def has_explicit_keys(cells_list):

        # Count how many non-empty rows start with a cell that ends in ':'
        colon_count = 0
        total = 0
        for row in cells_list:
            non_empty = [c.strip() for c in row if c.strip()]
            if non_empty:
                total += 1
                if non_empty[0].endswith(':'):
                    colon_count += 1

        # Consider the table to have explicit keys if ≥50% of rows qualify
        return total > 0 and (colon_count / total) >= 0.5

    def count_header_data_pairs(cells_list):

        # Walk through the rows looking for consecutive label-row / data-row pairs
        count, i = 0, 0
        while i < len(cells_list) - 1:

            if is_label_row(cells_list[i]) and has_concrete_value(cells_list[i + 1]):
                count += 1

                # Skip both rows since they form a complete pair
                i += 2

            else:
                i += 1

        return count

    # Determine whether the table uses explicit ':'-terminated key cells
    explicit_keys  = has_explicit_keys(all_cells)

    # Count alternating header/data row pairs for the other detection strategy
    total_pairs    = count_header_data_pairs(all_cells)

    # Flag for inline pattern: keys are present in the same row as their values
    is_inline      = explicit_keys

    # Flag for alternating pattern: one row of labels followed by one row of data
    is_alternating = (total_pairs >= 1) and not explicit_keys

    # Dictionary that will accumulate all extracted key-value pairs
    structured_data = {}

    if is_inline:

        # Process each row scanning left-to-right for ':'-terminated key cells
        for row in all_cells:

            # Strip whitespace from every cell in this row
            row = [c.strip() for c in row]

            # Remove trailing empty cells to avoid off-by-one index errors
            while row and not row[-1]:
                row.pop()

            j = 0

            while j < len(row):

                cell      = row[j]

                # Safely peek at the next cell; use empty string if out of bounds
                next_cell = row[j + 1].strip() if j + 1 < len(row) else ''

                # A key ends with ':' and is followed by a non-key value cell
                if cell.endswith(':') and next_cell and not next_cell.endswith(':'):

                    # Store the pair, removing the trailing colon from the key
                    structured_data[cell.rstrip(':').strip()] = next_cell

                    # Advance by two to skip both the key and value cells
                    j += 2

                    continue

                j += 1

    elif is_alternating:
        i = 0
        while i < len(all_cells):

            # Case 1: label row immediately followed by a row with concrete values
            if (i + 1 < len(all_cells)
                    and is_label_row(all_cells[i])
                    and has_concrete_value(all_cells[i + 1])):
                headers = all_cells[i]
                values  = all_cells[i + 1]

                # Zip headers and values by column position
                for j in range(min(len(headers), len(values))):
                    key = headers[j].strip().rstrip(':')
                    val = values[j].strip()

                    # Only store the pair when both key and value are meaningful
                    if key and val and is_label(key):
                        structured_data[key] = val

                # Both rows consumed; advance by two
                i += 2

            # Case 2: two consecutive label rows (treat second row as values)
            elif (i + 1 < len(all_cells)
                    and is_label_row(all_cells[i])
                    and is_label_row(all_cells[i + 1])):
                headers = all_cells[i]
                values  = all_cells[i + 1]

                # Zip headers and values by column position
                for j in range(min(len(headers), len(values))):
                    key = headers[j].strip().rstrip(':')
                    val = values[j].strip()

                    # Only store the pair when both key and value are meaningful
                    if key and val and is_label(key):
                        structured_data[key] = val

                # Both rows consumed; advance by two
                i += 2
            else:
                # Row does not match any known pattern; skip it
                i += 1

    return structured_data


# ──────────────────────────────────────────────
# MMD → structured JSON conversion
# ──────────────────────────────────────────────

def convert_mmd_to_structured_json(file_path):
    print(f"Convertendo {file_path} para JSON...")

    # Abort early if the source .mmd file does not exist on disk
    if not os.path.exists(file_path):
        print(f"Erro: Arquivo {file_path} não encontrado.")
        return None

    # Read the full MMD (Markdown + embedded HTML) content into memory
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse the content with BeautifulSoup to locate HTML table elements
    soup = BeautifulSoup(content, 'html.parser')

    # Iterate over every <table> tag found in the document
    extracted_tables = []

    for table in soup.find_all('table'):

        # Convert the HTML table into a structured dictionary
        parsed = parse_html_table(table)

        # Only keep the table if at least one key-value pair was extracted
        if parsed:
            extracted_tables.append(parsed)

        # Remove the table from the soup tree so it does not appear in text_flow
        table.decompose()

    # Extract the remaining plain text after all tables have been removed
    remaining_text = soup.get_text()

    # Split into individual lines and discard blank ones
    lines = [line.strip() for line in remaining_text.split('\n') if line.strip()]

    # Assemble the top-level document structure that will be serialised to JSON
    document_structure = {
        # List of structured dicts extracted from HTML tables
        'tables': extracted_tables,

        # Remaining text lines that were not part of any table
        'text_flow': lines,

        # Inline key:value fields detected in plain-text lines
        'fields_detected': {}
    }

    # Scan every text line for "Key: Value" patterns not captured by tables
    for line in lines:

        # Match lines like "Razão Social: DM Ambiental LTDA" (key up to 80 chars)
        match = re.match(r'^([^#\n]{1,80}):\s*(.+)$', line)

        if match:

            key   = match.group(1).strip()
            value = match.group(2).strip()

            # Store the detected field in the dedicated section
            document_structure['fields_detected'][key] = value

    return document_structure
# Main class to be used by the endpoint
class DeepSeekOCRProcessor:

    # Static main method
    @staticmethod
    def main(model, tokenizer):

        # Ensure input and output directories exist before processing
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Escaneando pasta: {input_dir}")

        # Build the flat list of all page images to be processed
        all_pages = create_master_list(input_dir, output_dir)

        # Store the total number of pages for progress logging
        total = len(all_pages)

        # Dictionary that will accumulate results for every processed page
        all_results = {}

        print(f"Iniciando processamento de {total} página(s).")

        # Iterate over each page image with its 1-based index for logging
        for index, current_img_path in enumerate(all_pages):

            # Extract the filename without extension to use as the page identifier
            page_id = os.path.basename(current_img_path).rsplit('.', 1)[0]

            # Default path where the model always writes its output file
            mmd_default = os.path.join(output_dir, 'result.mmd')

            # Final path for the .mmd file renamed with the page identifier
            mmd_final = os.path.join(output_dir, f"{page_id}.mmd")

            print(f"\n[{index + 1}/{total}] Processando: {os.path.basename(current_img_path)}")

            try:
                # Run the OCR model on the current page image
                model.infer(

                    tokenizer,
                    # Prompt instructs the model to convert the image to markdown
                    prompt="<image>\n<|grounding|>Convert the document to markdown.",

                    # Path to the current page image
                    image_file=current_img_path,

                    # Directory where the model will write result.mmd
                    output_path=output_dir,

                    # Tell the model to persist its output to disk
                    save_results=True
                )

                # Check if the model produced the expected output file
                if os.path.exists(mmd_default):

                    # Remove a stale .mmd from a previous run for this page
                    if os.path.exists(mmd_final):
                        os.remove(mmd_final)

                    # Rename the generic output file to the page-specific name
                    os.rename(mmd_default, mmd_final)

                    # Parse the .mmd file into a structured dictionary
                    result = convert_mmd_to_structured_json(mmd_final)

                    # Only store the result if the conversion returned valid data
                    if result:
                        all_results[page_id] = result

                else:

                    # Warn if the model did not generate any output for this page
                    print(f"  !! Aviso: result.mmd não encontrado para {page_id}.")
                    all_results[page_id] = {"error": "result.mmd não gerado pelo modelo"}

            except torch.cuda.OutOfMemoryError:

                # GPU ran out of memory: free the cache and record the error
                print(f"  !! Erro: VRAM insuficiente ao processar {page_id}.")
                torch.cuda.empty_cache()
                all_results[page_id] = {"error": "VRAM insuficiente"}

            except Exception as e:

                # Catch any other unexpected error and continue to the next page
                print(f"  !! Erro inesperado em {page_id}: {e}")
                all_results[page_id] = {"error": str(e)}

        # Delete all temporary files from the output folder after processing
        cleanup_output_folder(output_dir)

        print("\n--- OCR de todos os arquivos concluído ---")

        # Return the dictionary with results for all processed pages
        return all_results