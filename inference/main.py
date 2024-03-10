import subprocess
import json, dotenv
import datetime, sys, os
from pymongo import MongoClient
from pymongo.server_api import ServerApi

dotenv.load_dotenv(dotenv_path="/home/ubuntu/InferenceTest1/rec/inference/.env")

RC_SUCESS = 0
RC_MEMORY_LIMIT = 1
OUTPUT_DIR = "/home/ubuntu/InferenceTest1/rec/inference/pipeline/data"
DB_NAME = "preTechnigalaClean_db"
COLLECTION_NAME = "video_metadata"
MONGO_DB_CLIENT = MongoClient(os.getenv("MONGODB_URI"), server_api=ServerApi('1'))

def run_inference_subprocess(script_path, output_dir):
    """Runs inference script as a subprocess. Will continue running until return code is 0"""
    ## If inference metadata folder doesnt exist, create it 
    if not os.path.exists(f'{output_dir}/inference_metadata'):
        os.makedirs(f'{output_dir}/inference_metadata')
    ## if no documents have missing isVectorized or vectorizeFailed fields, exit
    if MONGO_DB_CLIENT[DB_NAME][COLLECTION_NAME].count_documents({"$or": [{"isVectorized": {"$exists": False}}, {"vectorizeFailed": {"$exists": False}}]}) == 0:
        print('All Documents Completed')
        sys.exit(0)

    output_file_no = len(os.listdir(f'{output_dir}/inference_metadata'))
    output_file_path = f'{output_dir}/inference_metadata/{output_file_no}.txt'
    with open(output_file_path, 'w') as metadata_file:
        metadata_file.write(f'################# BEGINNING RUN: TIME: {datetime.datetime.now()} #################\n')
        with subprocess.Popen(['python3', '-u', script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            # Read the subprocess output line by line and write to the file in real-time
            for line in proc.stdout:
                print(line, end='')  # Optionally print to the console in real-time
                metadata_file.write(line)
                metadata_file.flush()  # Ensure each line is written to the file immediately
            # Wait for the subprocess to finish and get the return code
            proc.wait()

    return_code = proc.returncode

    if return_code == RC_SUCESS:
        print('All Documents Completed')
        sys.exit(0)
    elif return_code == RC_MEMORY_LIMIT:
        print('Memory limit exceeded')
        run_inference_subprocess(script_path, output_dir)
    else:
        print('Unknown error')
        print(proc.stderr)
        print(proc.stdout)
        run_inference_subprocess(script_path, output_dir)
    
    return proc

if __name__ == "__main__":
    
    output = run_inference_subprocess('/home/ubuntu/InferenceTest1/rec/inference/pipeline/inference.py', '/home/ubuntu/InferenceTest1/rec/inference/pipeline/data')
