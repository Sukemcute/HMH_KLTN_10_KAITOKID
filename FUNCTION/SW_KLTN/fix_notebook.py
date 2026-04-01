import json

notebook_path = 'd:\\SW_KLTN\\Quantized_model.ipynb'
try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_data = json.load(f)

    # Modify paths and commands in the notebook
    for cell in nb_data.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            for line in source:
                # Remove Colab specific setup
                if '!git clone' in line or '%cd' in line or '!pip install' in line or 'username =' in line or 'token =' in line:
                    new_source.append('# ' + line)
                # Fix quant_state_dict path
                elif '/content/Ultralytics-dev/ultralytics/quant/quant_state_dict/qat_sttd.pt' in line:
                    new_source.append(line.replace('/content/Ultralytics-dev/ultralytics/quant/quant_state_dict/qat_sttd.pt', './Ultralytics-dev/ultralytics/quant/quant_state_dict/qat_sttd.pt'))
                else:
                    new_source.append(line)
            cell['source'] = new_source

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=2)
    print("Notebook patched successfully.")
except Exception as e:
    print(f"Error: {e}")
