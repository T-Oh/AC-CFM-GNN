import os
import shutil

def copy_and_rename_scenarios(processed_path, renamed_path, source_id):
    """
    Copies scenario folders and their data files to a new location with renamed folders and files.

    Args:
        processed_path (str): Path to the directory containing the original scenario folders.
        renamed_path (str): Path to the directory where renamed files and folders will be copied.
        source_id (int): Single-digit integer to prefix the folder and file names.
    """
    os.makedirs(renamed_path, exist_ok=True)

    for root, dirs, files in os.walk(processed_path):
        # Determine the relative path
        rel_path = os.path.relpath(root, processed_path)

        # Rename scenario folders correctly
        if rel_path != ".":
            base_folder = os.path.basename(root)
            if base_folder.startswith("scenario_"):
                new_folder_name = f"scenario_{source_id}{base_folder[9:]}"  # Add prefix to scenario ID only
            else:
                new_folder_name = base_folder
            new_folder_path = os.path.join(renamed_path, os.path.dirname(rel_path), new_folder_name)
        else:
            new_folder_path = renamed_path

        # Create the new folder path
        os.makedirs(new_folder_path, exist_ok=True)

        # Copy and rename files
        for file in files:
            if file.endswith(".pt"):
                old_file_path = os.path.join(root, file)
                new_file_name = f"data_{source_id}{file[5:]}"  # Add prefix to file name
                new_file_path = os.path.join(new_folder_path, new_file_name)
                print(f"Copying and renaming file: {old_file_path} -> {new_file_path}")
                shutil.copy(old_file_path, new_file_path)

        # Print folder copying action
        if rel_path != ".":
            print(f"Copying folder: {root} -> {new_folder_path}")

if __name__ == "__main__":
    processed_path = "processed/"
    renamed_path = "renamed/"
    source_id = 6  # Set the desired source ID (single-digit integer)

    print("Copying and renaming scenario folders and files...")
    copy_and_rename_scenarios(processed_path, renamed_path, source_id)
    print("Process completed.")
