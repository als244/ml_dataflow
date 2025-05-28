import os
import re
import pathlib

# Define the list of architectures to process
ARCHS = [80, 86, 89, 90, 100, 120]

# Get the current working directory (where the script is expected to be run,
# and where the .cu files are located)
current_dir = pathlib.Path(".")

print(f"Starting Python script to modify .cu files for different architectures...")
print(f"Architectures to process: {ARCHS}")
print(f"Looking for .cu files in the current directory: {current_dir.resolve()}")

def modify_template_line(line, arch_value):
    """
    Modifies a line of CUDA code to add the architecture as the first template argument.
    Handles the outer template specialization (void run_mha_...) and 
    the splitkv dispatch template (template void run_mha_fwd_splitkv_dispatch...).
    """
    # Pattern for the outer template function signature line
    # (e.g., "void run_mha_fwd_<T, U>(params...)")
    # Captures:
    # 1. The prefix up to and including '<': (^\s*void\s+run_mha_(?:fwd|bwd)_<)
    # 2. Existing arguments (if any): ([^>]*)
    # 3. The closing angle bracket: (\>)
    outer_template_pattern = r"(^\s*void\s+run_mha_(?:fwd|bwd)_<)([^>]*)(\>)"

    # Pattern for the "template void run_mha_fwd_splitkv_dispatch" line
    # (e.g., "template void run_mha_fwd_splitkv_dispatch<T, U>(params...);")
    # Captures:
    # 1. The prefix up to and including '<': (^\s*template\s+void\s+run_mha_fwd_splitkv_dispatch<)
    # 2. Existing arguments (if any): ([^>]*)
    # 3. The closing angle bracket: (\>)
    splitkv_template_pattern = r"(^\s*template\s+void\s+run_mha_fwd_splitkv_dispatch<)([^>]*)(\>)"


    def replacer(match):
        prefix = match.group(1)
        existing_args = match.group(2)
        closing_bracket = match.group(3) # This will be ">"

        if existing_args:
            # If there are existing arguments, add the new arch, a comma, and then the existing args
            new_args_section = f"{arch_value}, {existing_args}"
        else:
            # If there are no existing arguments, just add the new arch
            new_args_section = str(arch_value)
        
        # re.sub replaces only the matched part. The rest of the line after the closing_bracket
        # (e.g., "(Flash_bwd_params &params, cudaStream_t stream)") is preserved.
        return f"{prefix}{new_args_section}{closing_bracket}"

    # Apply modifications sequentially.
    # The patterns are distinct enough not to interfere with each other.
    current_line_state = line
    current_line_state = re.sub(outer_template_pattern, replacer, current_line_state)
    
    # If the line was not modified by the first pattern, this second sub will try.
    # If it was modified, this second pattern (being distinct) should not match the already modified line.
    if line == current_line_state: # Only try the second pattern if the first didn't change the line
        current_line_state = re.sub(splitkv_template_pattern, replacer, current_line_state)
    
    return current_line_state

# Iterate over all .cu files in the current directory
for cu_file_path in current_dir.glob("*.cu"):
    if cu_file_path.is_file():
        original_name_no_ext = cu_file_path.stem
        print("-----------------------------------------------------")
        print(f"Processing original file: {cu_file_path.name}")

        for arch in ARCHS:
            print(f"  Processing for Arch: {arch}")

            # Create subdirectory sm<Arch> if it doesn't exist
            output_dir = current_dir / f"sm{arch}"
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"    Output directory: {output_dir.resolve()}")


            # Define the new filename: originalname_sm<Arch>.cu
            new_cu_filename = f"{original_name_no_ext}_sm{arch}.cu"
            new_cu_filepath = output_dir / new_cu_filename

            print(f"    Attempting to create new file: {new_cu_filepath.resolve()}")

            modified_content = [] # Store all lines for the new file
            try:
                with open(cu_file_path, 'r', encoding='utf-8') as infile:
                    for line_content in infile:
                        modified_content.append(modify_template_line(line_content.rstrip('\n'), arch))
                
                with open(new_cu_filepath, 'w', encoding='utf-8') as outfile:
                    for processed_line in modified_content:
                        outfile.write(processed_line + '\n')
                print(f"    Successfully created and modified: {new_cu_filepath.resolve()}")

            except Exception as e:
                print(f"    ERROR: Failed to process or create {new_cu_filepath.resolve()} for Arch {arch} from {cu_file_path.name}.")
                print(f"    Error details: {e}")

print("-----------------------------------------------------")
print("Python script finished.")
print("Please check the sm<Arch> subdirectories for the new files.")