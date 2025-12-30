import arxiv
import tarfile
import os
import shutil

def fetch_source(arxiv_id, output_dir="arxiv_source_cache"):
    """
    Downloads and extracts the source (LaTeX) for a given ArXiv ID.
    Returns the path to the extracted directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    target_dir = os.path.join(output_dir, arxiv_id)
    if os.path.exists(target_dir):
        print(f"Source for {arxiv_id} already exists at {target_dir}")
        return target_dir

    print(f"Fetching source for {arxiv_id}...")
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    
    paper = next(client.results(search), None)
    if not paper:
        print(f"Error: Paper {arxiv_id} not found.")
        return None

    try:
        # Download tarball
        tar_path = paper.download_source(dirpath=output_dir, filename=f"{arxiv_id}.tar.gz")
        print(f"Downloaded source to {tar_path}")
        
        # Extract
        print(f"Extracting to {target_dir}...")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=target_dir)
        except tarfile.ReadError:
             # Sometimes it's not a tar (e.g. PDF only or single file). 
             # Check if it is a PDF masked as tar? No, download_source gets source.
             # If source is not available, ArXiv might return PDF.
             print("Warning: Could not extract tar. Is the source available?")
             return None
             
        # Cleanup tar
        os.remove(tar_path)
        return target_dir
        
    except Exception as e:
        print(f"Failed to fetch source: {e}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        fetch_source(sys.argv[1])
    else:
        print("Usage: python fetch_source.py <ARXIV_ID>")
