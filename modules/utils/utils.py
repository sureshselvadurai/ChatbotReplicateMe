import os

def merge_txt_files(directory, output_file="merged_file.txt"):
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory.")
        return

    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Filter out only the txt files
    txt_files = [f for f in files if f.endswith(".txt")]

    if not txt_files:
        print("No .txt files found in the directory.")
        return

    # Extract the directory path from the output file
    output_directory = os.path.dirname(output_file)

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Open the output file for writing
    with open(output_file, "w") as outfile:
        # Loop through each txt file and append its content to the output file
        for txt_file in txt_files:
            with open(os.path.join(directory, txt_file), "r") as infile:
                outfile.write(infile.read())
                # Add a newline after each file's content for separation
                outfile.write("\n")

    print(f"Merged {len(txt_files)} .txt files into {output_file}.")
