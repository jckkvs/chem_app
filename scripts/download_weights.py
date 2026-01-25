#!/usr/bin/env python
"""
Model Weights Downloader for ChemML Platform

Downloads pre-trained weights for SSL and GNN models from official sources.
"""
import argparse
import hashlib
import logging
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Optional
import urllib.request
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("downloader")

# Default weights directory: ~/.chem_ml/weights
DEFAULT_WEIGHTS_DIR = Path.home() / ".chem_ml" / "weights"

# Model definitions
# Format: key -> {url, filename, md5, description}
MODELS = {
    "grover_base": {
        "url": "https://grover.readthedocs.io/en/latest/_downloads/grover_base.pt",
        "filename": "grover_base.pt",
        "description": "GROVER Base (100M params, 10M molecules)",
        # "md5": "..." # Optional verification
    },
    "grover_large": {
        "url": "https://grover.readthedocs.io/en/latest/_downloads/grover_large.pt",
        "filename": "grover_large.pt",
        "description": "GROVER Large (100M params, 10M molecules)",
    },
    "molclr_gin": {
        "url": "https://github.com/yuyangw/MolCLR/raw/master/ckpt/pretrained_gin/checkpoints/model.pth", 
        "filename": "molclr_gin.pth",
        "description": "MolCLR with GIN encoder (10M molecules)",
    },
    # SchNet/PaiNN usually download via schnetpack internal tools, 
    # but we can provide specific fine-tuned weights if hosted.
    # For now, we focus on the ones that require manual download.
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url: str, output_path: Path, force: bool = False) -> bool:
    """Download file from URL with progress bar"""
    if output_path.exists() and not force:
        logger.info(f"File already exists: {output_path.name}")
        return True
    
    logger.info(f"Downloading {url} to {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True,
                               miniters=1, desc=output_path.name) as t:
            urllib.request.urlretrieve(
                url, filename=output_path, reporthook=t.update_to
            )
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if output_path.exists():
            os.remove(output_path)
        return False

def main():
    parser = argparse.ArgumentParser(description="Download pre-trained model weights")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Model to download (default: all)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=str(DEFAULT_WEIGHTS_DIR),
        help=f"Output directory (default: {DEFAULT_WEIGHTS_DIR})"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force download even if file exists"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    # Select models
    if args.model == "all":
        targets = MODELS.keys()
    else:
        targets = [args.model]
    
    logger.info(f"Output directory: {output_dir}")
    
    success_count = 0
    for key in targets:
        info = MODELS[key]
        logger.info(f"Target: {info['description']} ({key})")
        
        outfile = output_dir / info['filename']
        if download_url(info['url'], outfile, args.force):
            success_count += 1
            
    logger.info(f"Completed: {success_count}/{len(targets)} downloads successful")

if __name__ == "__main__":
    main()
