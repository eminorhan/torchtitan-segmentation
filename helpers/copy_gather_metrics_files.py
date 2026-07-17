import zipfile
from pathlib import Path

def collect_experiment_metrics(base_dir="outputs", output_zip="all_experiments.zip"):
    base_path = Path(base_dir)
    
    # Check if the outputs directory exists
    if not base_path.is_dir():
        print(f"Error: Directory '{base_dir}' not found.")
        return

    # Open a new zip archive with standard deflation (compression)
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        # Recursively find all matching metrics.jsonl files
        # Glob pattern targets: outputs/<expt_name>/logs/metrics.jsonl
        matched_files = list(base_path.glob("*/logs/metrics.jsonl"))
        
        if not matched_files:
            print("No metrics.jsonl files found matching the expected structure.")
            return

        for file_path in matched_files:
            # Extract the experiment name. 
            # file_path.parents[0] is 'logs', file_path.parents[1] is '<expt_name>'
            expt_name = file_path.parents[1].name
            
            # Define what the file will be called inside the zip archive
            archive_name = f"{expt_name}.jsonl"
            
            # Add to archive, using arcname to rename it
            zipf.write(file_path, arcname=archive_name)
            
        print(f"Successfully archived {len(matched_files)} files into '{output_zip}'.")

if __name__ == "__main__":
    
    collect_experiment_metrics()