"""
Helper script to download additional datasets for benchmarking.
"""

import os
import requests
import json
from pathlib import Path
import subprocess
import shutil


def download_spdx_samples(output_dir: Path):
    """Download SPDX license samples."""
    print("Downloading SPDX license samples...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clone SPDX license list
    temp_dir = output_dir.parent / "temp_spdx"
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    try:
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/spdx/license-list-data.git",
            str(temp_dir)
        ], check=True)
        
        # Copy JSON files
        json_dir = temp_dir / "json" / "details"
        if json_dir.exists():
            shutil.copytree(json_dir, output_dir, dirs_exist_ok=True)
            print(f"✓ Downloaded SPDX samples to {output_dir}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"✗ Error downloading SPDX samples: {e}")


def download_github_licenses(output_dir: Path):
    """Download common licenses from GitHub API."""
    print("Downloading GitHub common licenses...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of licenses
    api_url = "https://api.github.com/licenses"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        licenses = response.json()
        
        print(f"Found {len(licenses)} licenses")
        
        # Download each license
        for license_info in licenses:
            license_key = license_info['key']
            license_url = license_info['url']
            
            print(f"  Downloading {license_key}...")
            
            license_response = requests.get(license_url)
            license_response.raise_for_status()
            license_data = license_response.json()
            
            # Save to file
            output_file = output_dir / f"{license_key}.json"
            with open(output_file, 'w') as f:
                json.dump(license_data, f, indent=2)
        
        print(f"✓ Downloaded GitHub licenses to {output_dir}")
        
    except Exception as e:
        print(f"✗ Error downloading GitHub licenses: {e}")


def create_real_world_template(output_dir: Path):
    """Create template for real-world license collection."""
    print("Creating real-world dataset template...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create README
    readme_content = """# Real-World License Dataset

This directory should contain license files collected from real open-source projects.

## Structure

1. Place license files in subdirectories (e.g., by project name)
2. Create a `labels.csv` file with columns: `file_path`, `license_id`

Example `labels.csv`:
```csv
file_path,license_id
project1/LICENSE,MIT
project2/COPYING,GPL-3.0-only
project3/LICENSE.txt,Apache-2.0
```

## Collection Tips

1. Clone popular open-source projects
2. Extract their LICENSE files
3. Manually verify the license type
4. Add entries to labels.csv

## Suggested Projects

- tensorflow (Apache-2.0)
- pytorch (BSD-3-Clause)
- linux (GPL-2.0-only)
- react (MIT)
- django (BSD-3-Clause)
- flask (BSD-3-Clause)
- redis (BSD-3-Clause)
- docker (Apache-2.0)
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    # Create empty labels.csv
    labels_path = output_dir / "labels.csv"
    with open(labels_path, 'w') as f:
        f.write("file_path,license_id\n")
    
    print(f"✓ Created template at {output_dir}")
    print(f"  → Edit {labels_path} and add license files")


def main():
    """Download all additional datasets."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    print("="*80)
    print("DOWNLOADING ADDITIONAL DATASETS")
    print("="*80 + "\n")
    
    # SPDX samples
    spdx_dir = data_dir / "spdx_samples"
    download_spdx_samples(spdx_dir)
    print()
    
    # GitHub licenses
    github_dir = data_dir / "github_licenses"
    download_github_licenses(github_dir)
    print()
    
    # Real-world template
    real_world_dir = data_dir / "real_world_licenses"
    create_real_world_template(real_world_dir)
    print()
    
    print("="*80)
    print("DONE")
    print("="*80)
    print("\nTo enable these datasets:")
    print("1. Edit benchmarks/config.py")
    print("2. Set 'enabled': True for the datasets you want to use")
    print("3. For real_world dataset, collect license files and update labels.csv")


if __name__ == "__main__":
    main()
