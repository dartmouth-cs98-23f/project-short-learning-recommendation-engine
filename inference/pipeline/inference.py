import sys, os, datetime, dotenv, time, re, json
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pinecone import Pinecone
from validator import validate_inference_output
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import string
import nltk
from nltk.corpus import stopwords
from utils.transcript import download_transcript

dotenv.load_dotenv(dotenv_path="/home/ubuntu/InferenceTest1/rec/inference/.env")
# print current os location

########################
## GLOBAL VARIABLES ###
######################
current_doc = None

#################
## CONSTANTS ###
################
OUTPUT_DIR = "/home/ubuntu/InferenceTest1/rec/inference/pipeline/data"
DB_NAME = "preTechnigalaClean_db"
COLLECTION_NAME = "video_metadata"
MONGO_DB_CLIENT = MongoClient(os.getenv("MONGODB_URI"), server_api=ServerApi('1'))
PINECONE_INDEX_NAME = "pretechnigala"
PINECONE_CLIENT = Pinecone(api_key=os.getenv("PINECONE_KEY"))

MAX_INFERENCE_RUNS = 3
SHOTS = ["mixtral8x7b", "full-stack"]
SHOTS_CHAR_LIMIT = 5550
INPUT_CHAR_LIMIT = 5550
TEMPERATURE = 0.5
TOP_K = 35
TOP_P = 0.96

def main():
    """
    Main function to run the pipeline. If an error occurs in inference or embedding generation, 
    this subprocess is restarted. If validation fails up to MAX_INFERENCE_RUNS times, 
    or any other error occurs, the document is marked as failed and the next document is processed.
    """
    global current_doc
    start_time = time.time()
    doc = get_next_document()
    if doc is None:
        print("No more documents to process")
        sys.exit(0)
    current_doc = doc
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(device)
    print(f"Model loaded on {device}, Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    while current_doc:
        try:
            print(f'\n########################################\n# ID: {current_doc["_id"]} #\n###########################################\n')
            print(f'Runtime so far: {(time.time() - start_time)/60:.2f} minutes')
            print(f'Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB')
            clean_transcript(current_doc)
            ## Try to run inference and validate output up to MAX_INFERENCE_RUNS times
            final_path = None
            did_fail = False
            for run in range(MAX_INFERENCE_RUNS):
                print(f"Run {run+1}/{MAX_INFERENCE_RUNS}")
                path = f'{OUTPUT_DIR}/outputs/{current_doc["_id"]}/inference_output_{run}.json'
                final_path = path
                run_inference(device, tokenizer, model, path)
                if validate_inference_output(path):
                    did_fail = False
                    break
                else:
                    did_fail = True
            generate_embedding(device, tokenizer, model, path)
            upload_to_database(did_fail, final_path)

            current_doc = get_next_document()
        except Exception as e:
            ## If an error occurs, mark the document as failed
            collection = MONGO_DB_CLIENT[DB_NAME][COLLECTION_NAME]
            collection.update_one({"_id": current_doc["_id"]}, {"$set": {"vectorizeFailed": True}})
            print(f"Error processing document: {e}")
            current_doc = get_next_document()
    
    return 0

def clean_transcript(doc):
    """
    Clean the transcript of any unwanted characters
    Args:
        doc (dict): The document to clean
    """
    yt_id = doc["youtubeURL"].split("v=")[1]
    name = doc["_id"]
    print(f"Cleaning transcript yt_id: {yt_id}")
    os.makedirs(f'{OUTPUT_DIR}/transcripts/raw', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/transcripts/clean', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/transcripts/processed', exist_ok=True)

    download_transcript(yt_id, f'{OUTPUT_DIR}/transcripts/raw/{name}.txt')
    with open(f'{OUTPUT_DIR}/transcripts/raw/{name}.txt', 'r') as file:
        with open(f'{OUTPUT_DIR}/transcripts/clean/{name}.txt', 'w') as file_clean:
            for i,line in enumerate(file):
                if i == 0:
                    continue
                ## Remove Timestamp
                pattern = r'\d+\.\d+\,\d+\.\d+\,'
                line = re.sub(pattern, '', line)
                line = line.replace("'", "")
                line = line.replace('"', "")
                ## if last char is a space, remove
                if line[-1] == " ":
                    line = line[:-1]
                line = line.replace("  ", " ")
                line = line.replace('\n', ' ')
                file_clean.write(line)

    nltk.download('stopwords')
    with open(f'{OUTPUT_DIR}/transcripts/clean/{name}.txt', 'r') as file:
        with open(f'{OUTPUT_DIR}/transcripts/processed/{name}.txt', 'w') as file_processed:
            text = file.read()
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            stop_words = set(stopwords.words('english'))
            filtered_text = [word for word in text.split() if word not in stop_words]
            filtered_text = ' '.join(filtered_text)
            file_processed.write(text)

def abridge_transcript(transcript: str, chars: int) -> str:
    """ Abridge a transcript to a certain number of characters
    Args:
        transcript (str): The transcript to abridge
        chars (int): The number of characters to abridge to
    Returns:
        str: The abridged transcript
    """
    try:
        if len(transcript) <= chars:
            return transcript
        
        output_transcript = ""
        ## Use 20% of chars for the start
        output_transcript += transcript[:int(chars*0.2)]
        output_transcript += " "

        # evenly divide the characters in the middle of the transcript and extract in 6 10% chunks
        middle = transcript[int(chars*0.2):-int(chars*0.2)]
        transcript_chunk_size = len(middle) // 6
        abridged_chunk_size = int(chars * 0.1)
        for i in range(6):
            target_chunk_index = transcript_chunk_size * i + (transcript_chunk_size // 2)
            output_transcript += middle[target_chunk_index:target_chunk_index+abridged_chunk_size]
            output_transcript += " "

        ## Use 20% of chars for the end
        output_transcript += transcript[-int(chars*0.2):]
        return output_transcript
    except Exception as e:
        print(f"Error abridging transcript: {e}")
        return transcript

def build_message(shots: list, transcript: str, prompt: str) -> list:
    """ Build a message for the model to generate from.
    Args:
        shots (list): List of shots to include in the message
        target (str): the open file of the target transcript
        start (int): The start of the transcript to include
        end (int): The end of the transcript to include
    Returns:
        list: A list of messages to send to the model
    """
    messages = []
    oneshots_dir = f'{OUTPUT_DIR}/inputs/shots'

    for shot in shots:
        f_oneshot_transcript = open(f'{oneshots_dir}/transcript_{shot}.txt', 'r', encoding='utf-8')
        f_oneshot = open(f'{oneshots_dir}/output_{shot}.txt', 'r', encoding='utf-8')
        
        messages.append({"role": "user", "content": f'Transcript:\n{abridge_transcript(f_oneshot_transcript.read(), SHOTS_CHAR_LIMIT)}\nInstruction:\n{prompt}'})
        messages.append({"role": "assistant", "content": f_oneshot.read()})

        f_oneshot.close()
        f_oneshot_transcript.close()

    ## Append final message
    messages.append({"role": "user", "content": f'Transcript:\n{abridge_transcript(transcript, INPUT_CHAR_LIMIT)}\nInstruction:\n{prompt}'})
    
    return messages

def run_inference(device, tokenizer, model, path):
    """
    Run inference on the model
    Args:
        device (str): The device to run the model on
        tokenizer (AutoTokenizer): The tokenizer to use
        model (AutoModelForCausalLM): The model to use
    Exits:
        1: If an error occurs (Mem error)
    """
    try:
        global current_doc
        os.makedirs(f'{OUTPUT_DIR}/outputs/{current_doc["_id"]}', exist_ok=True)

        f_prompt = open(f'{OUTPUT_DIR}/inputs/prompt.txt', 'r', encoding='utf-8')
        f_transcript = open(f'{OUTPUT_DIR}/transcripts/processed/{current_doc["_id"]}.txt', 'r', encoding='utf-8')
        transcript = f_transcript.read()
        prompt = f_prompt.read()
        f_prompt.close()
        f_transcript.close()
        
        start_time = time.time()
        messages = build_message(SHOTS, transcript, prompt)
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

        print(f"Tokens: {len(inputs[0])}")

        with torch.no_grad():
            text = model.generate(inputs, max_new_tokens=3500, do_sample=True, pad_token_id=tokenizer.eos_token_id, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P)
            decoded = tokenizer.batch_decode(text)

        print(f"Inference time taken: {time.time() - start_time:.2f} seconds")
        del text
        del inputs
        torch.cuda.empty_cache()

        f_output = open(path, 'w', encoding='utf-8')
        f_output.write(decoded[0].split('[/INST]')[-1][1:-4])
        f_output.close()
    except Exception as e:
        print(f"Error running inference: {e}")
        sys.exit(1)

def generate_embedding(device, tokenizer, model, summary_path):
    """
    Generate an embedding for the transcript.
    Args:
        device (str): The device to run the model on
        tokenizer (AutoTokenizer): The tokenizer to use
        model (AutoModelForCausalLM): The model to use
    Returns:
        Tuple(torch.Tensor, torch.Tensor): The max and avg pool embeddings
    Exits:
        1: If an error occurs (Mem error)
    """
    global current_doc
    try:

        f_summary = open(summary_path, 'r', encoding='utf-8')
        summary = f_summary.read()
        f_summary.close()

        start_time = time.time()
        inputs = tokenizer(summary, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**inputs, output_hidden_states=True)
        
        print(f"Embedding generation time taken: {time.time() - start_time:.2f} seconds")

        ## Write max and avg tensors
        tensor_t = hidden.hidden_states[-1].transpose(1,2)
        max_pool = torch.nn.functional.max_pool1d(tensor_t, tensor_t.shape[2]).transpose(1, 2).squeeze()
        avg_pool = torch.nn.functional.avg_pool1d(tensor_t, tensor_t.shape[2]).transpose(1, 2).squeeze()
        torch.save(max_pool, f'{OUTPUT_DIR}/outputs/{current_doc["_id"]}/max_pool.pt')
        torch.save(avg_pool, f'{OUTPUT_DIR}/outputs/{current_doc["_id"]}/avg_pool.pt')

        del hidden
        del inputs
        torch.cuda.empty_cache()

        return max_pool, avg_pool
    except Exception as e:
        print(f"Error generating embedding: {e}")
        sys.exit(1)
    
def upload_to_database(did_fail, final_path):
    """
    Upload an embedding to Pinecone and the document to MongoDB
    """
    global current_doc
    ## MongoDB Update
    collection = MONGO_DB_CLIENT[DB_NAME][COLLECTION_NAME]
    inference_collection = MONGO_DB_CLIENT[DB_NAME]["inference_summary"]
    
    ## upload with topicIds and complexities
    topics = []
    complexities = []
    with open(final_path, 'r') as file:
        output = file.read()
        json_summary = json.loads(output)
        generalTopics = json_summary["generalTopics"]
        for obj in generalTopics:
            topics.append(obj["name"])
            complexities.append(obj["complexity"])

    collection.update_one({"_id": current_doc["_id"]}, {"$set": {"isVectorized": True, "inferenceTopics": topics, "inferenceComplexities": complexities, "inferenceMisshape": did_fail, "vectorizeFailed": False}})
    inference_collection.insert_one({"_id": current_doc["_id"], "inferenceSummary": json_summary})

    print(f"Document updated in MongoDB: {current_doc['_id']} and inference summary uploaded")

    ## Pinecone Upload
    index = PINECONE_CLIENT.Index(PINECONE_INDEX_NAME)

    max_pool = torch.load(f'{OUTPUT_DIR}/outputs/{current_doc["_id"]}/max_pool.pt').to('cpu').numpy().tolist()
    avg_pool = torch.load(f'{OUTPUT_DIR}/outputs/{current_doc["_id"]}/avg_pool.pt').to('cpu').numpy().tolist()

    metadata_max = {"type": "max_pool", "title": current_doc["title"], "topics": [str(topic) for topic in current_doc["topicId"]], "videoID": str(current_doc["_id"]), "inferenceTopics": [str(topic) for topic in topics], "inferenceComplexities": [str(complexity) for complexity in complexities]}
    metadata_avg = {"type": "avg_pool", "title": current_doc["title"], "topics": [str(topic) for topic in current_doc["topicId"]], "videoID": str(current_doc["_id"]), "inferenceTopics": [str(topic) for topic in topics], "inferenceComplexities": [str(complexity) for complexity in complexities]}

    vectors = [
        {"id": str(current_doc["_id"]), "values": max_pool, "metadata": metadata_max},
        {"id": str(current_doc["_id"]), "values": avg_pool, "metadata": metadata_avg}
    ]
    index.upsert(vectors=vectors)
    print(f"Embedding uploaded for document: {current_doc['_id']}")

def get_next_document():
    """
    Get the next document to process. Sets global doc
    Returns:
        dict: The next document to process
        None: If there are no more documents to process
    """
    global current_doc
    db = MONGO_DB_CLIENT[DB_NAME]
    collection = db[COLLECTION_NAME]

    ## Find a document with no isVectorized field or isVectorized is False
    ## Ensure that this document has not failed in the past
    current_doc = collection.find_one({
        "$and": [ 
            {
                "$or": [
                    {"isVectorized": False},      
                    {"isVectorized": {"$exists": False}}
                ],
            },
            {
                "$or": [
                    {"vectorizeFailed": False},      
                    {"vectorizeFailed": {"$exists": False}}
                ],
            }
        ]
    })

    return current_doc


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        sys.exit(1)