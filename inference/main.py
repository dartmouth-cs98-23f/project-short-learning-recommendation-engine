import subprocess
import json
import datetime
import os

RC_SUCESS = 0
RC_MEMORY_LIMIT = 1

def run_inference_subprocess(script_path, output_dir):
    """Runs inference script as a subprocess. Will continue running until return code is 0"""
    ## If inference metadata folder doesnt exist, create it 
    if not os.path.exists(f'{output_dir}/inference_metadata'):
        os.makedirs(f'{output_dir}/inference_metadata')

    output_file_no = len(os.listdir(f'{output_dir}/inference_metadata'))
    output_file_path = f'{output_dir}/inference_metadata/{output_file_no}.txt'
    metadata_file = open(output_file_path, 'w')
    metadata_file.write(f'################# BEGINNING RUN: TIME: {datetime.datetime.now()} #################\n')

    result = subprocess.run(['python3', script_path] + [output_dir], capture_output=True, text=True)
    return_code = result.returncode

    metadata_file.write(f"stdout: {result.stdout}\nsterr: {result.stderr}\n")

    if return_code == RC_SUCESS:
        print('All Documents Completed')
    elif return_code == RC_MEMORY_LIMIT:
        print('Memory limit exceeded')
        run_inference_subprocess(script_path, output_dir)
    else:
        print('Unknown error')
        print(result.stderr)
        print(result.stdout)
        run_inference_subprocess(script_path, output_dir)
    
    return result

if __name__ == "__main__":
    
    output = run_inference_subprocess('inference/pipeline/inference.py', 'inference/pipeline/data')
