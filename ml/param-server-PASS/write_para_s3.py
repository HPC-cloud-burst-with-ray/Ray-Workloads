import sys
def change_bandwidth(file_path, new_value):
    # Read the contents of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the line containing HPC_MAX_BANDWITH=1000 and replace it
    for i, line in enumerate(lines):
        
        if 'MAX_BANDWIDTH_HPC=' in line:
            lines[i] = f'MAX_BANDWIDTH_HPC={new_value}\n'
            break

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

# Example usage:
if len(sys.argv) > 1:
    new_bandwidth_value =sys.argv[1] 
else:
    print(f"Unknown value. Exiting")
    sys.exit(1)
file_path = '/home/ec2-user/share/Ray-Workloads/ml/param-server-PASS/const.py'
file_path_sch='/home/ec2-user/ray/python/ray/scheduler/scheduler_constant.py'
# Change this to the desired value
change_bandwidth(file_path, new_bandwidth_value)

change_bandwidth(file_path_sch,new_bandwidth_value)
