#!/usr/bin/env python3
import os
import shutil
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("flatten_curriculum")

def main():
    src_root = "/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/academics-specialisation-and-courses"
    dest_dir = "/home/c3i/chatbot/scraped_data/sections/academics"

    if not os.path.exists(src_root):
        logger.error(f"Source directory does not exist: {src_root}")
        return

    os.makedirs(dest_dir, exist_ok=True)
    copied_count = 0

    for root, dirs, files in os.walk(src_root):
        for file in files:
            if not file.endswith(".md"):
                continue

            filepath = os.path.join(root, file)
            # Get relative path under src_root
            rel_path = os.path.relpath(filepath, src_root)
            
            # Extract hierarchy parts (e.g. ['PG', 'MTech', 'file.md'])
            parts = rel_path.split(os.sep)
            
            # Lowercase the directories and construct the clean prefix
            prefix_parts = [p.lower().replace(" ", "_").replace("-", "_") for p in parts[:-1]]
            clean_filename = parts[-1]
            
            # Flatten name: academics_curriculum_<dir1>_<dir2>_<filename>
            new_filename = "academics_curriculum_" + "_".join(prefix_parts) + "_" + clean_filename
            # Clean up double underscores or weird characters
            new_filename = re.sub(r'_+', '_', new_filename)
            
            dest_path = os.path.join(dest_dir, new_filename)
            logger.info(f"Copying: {rel_path} -> {new_filename}")
            shutil.copy2(filepath, dest_path)
            copied_count += 1

    logger.info(f"Successfully copied {copied_count} curriculum markdown files to {dest_dir}")

if __name__ == "__main__":
    main()
