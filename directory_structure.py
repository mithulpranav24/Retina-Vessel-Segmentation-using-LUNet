import os

def write_directory_tree(start_path, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(start_path):
            level = root.replace(start_path, '').count(os.sep)
            indent = ' ' * 4 * level
            f.write(f'{indent}{os.path.basename(root)}/\n')
            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                f.write(f'{sub_indent}{file}\n')

# Example usage:
start_directory = r'C:\Users\Mithul\Documents\JetBrains Projects\PyCharm Professional\PythonProject\DRIVE'  # change to your directory
output_txt = 'directory_structure.txt'
write_directory_tree(start_directory, output_txt)

print(f'Directory structure written to: {output_txt}')
