import subprocess

# List of command lines to execute
commands = [
    "python scripts/generate_fidelity_levels.py --im_name ballerina --layer_opt 2 --object_or_background 'background'",
    "python scripts/generate_fidelity_levels.py --im_name ballerina --layer_opt 8 --object_or_background 'background'",
    "python scripts/generate_fidelity_levels.py --im_name ballerina --layer_opt 11 --object_or_background 'background'",
    "python scripts/generate_fidelity_levels.py --im_name ballerina --layer_opt 2 --object_or_background 'object' --resize_obj 1",
    "python scripts/generate_fidelity_levels.py --im_name ballerina --layer_opt 8 --object_or_background 'object' --resize_obj 1",
    "python scripts/generate_fidelity_levels.py --im_name ballerina --layer_opt 11 --object_or_background 'object' --resize_obj 1",
]

# Iterate through the commands and execute them one by one
for cmd in commands:
    try:
        # Execute the command and capture the output
        output = subprocess.check_output(cmd, shell=True, universal_newlines=True)
        
        # Print the command and its output
        print(f"Command: {cmd}")
        print(f"Output:\n{output}")
    
    except subprocess.CalledProcessError as e:
        # Handle any errors that occur during command execution
        print(f"Error executing command: {cmd}")
        print(f"Error message: {e.output}")

print("All commands have been executed.")
