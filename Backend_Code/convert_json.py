import json

def convert_txt_to_json(input_file='output_gemini.txt', output_file='data.json'):
    """
    Convert a text file to JSON format by parsing specific keywords.
    Extracts content after "Diagnosis:" and "Explanation:" keywords.
    
    Args:
        input_file (str): Path to the input text file
        output_file (str): Path to the output JSON file
    """
    try:
        # Read the content from the text file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Initialize the data dictionary
        data = {}
        
        # Find "Diagnosis:" and extract its content
        if "Diagnosis:" in content:
            diagnosis_start = content.find("Diagnosis:") + len("Diagnosis:")
            # Find where Explanation starts (or end of file)
            explanation_pos = content.find("Explanation:", diagnosis_start)
            if explanation_pos != -1:
                diagnosis_content = content[diagnosis_start:explanation_pos].strip()
            else:
                diagnosis_content = content[diagnosis_start:].strip()
            data["Diagnosis"] = diagnosis_content
        
        # Find "Explanation:" and extract its content
        if "Explanation:" in content:
            explanation_start = content.find("Explanation:") + len("Explanation:")
            explanation_content = content[explanation_start:].strip()
            data["Explanation"] = explanation_content
        
        # Write to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully converted {input_file} to {output_file}")
        return data
        
    except FileNotFoundError:
        print(f"Error: {input_file} not found")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    convert_txt_to_json('FinalDiagnosis.txt', 'data.json')
