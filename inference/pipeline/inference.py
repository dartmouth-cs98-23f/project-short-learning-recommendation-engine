import sys, os, datetime, dotenv, time
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pinecone import Pinecone
from validator import validate_inference_output
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

dotenv.load_dotenv(dotenv_path="../.env")

########################
## GLOBAL VARIABLES ###
######################
current_doc = None

#################
## CONSTANTS ###
################
OUTPUT_DIR = sys.argv[1]
DB_NAME = "preTechnigala_db"
COLLECTION_NAME = "video_metadata"
MONGO_DB_CLIENT = MongoClient(os.getenv("MONGODB_URI"), server_api=ServerApi('1'))
PINECONE_INDEX_NAME = "preTechnigala"
PINECONE_CLIENT = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

MAX_INFERENCE_RUNS = 3
SHOTS = ["mixtral8x7b", "full-stack"]
SHOTS_CHAR_LIMIT = 6500
INPUT_CHAR_LIMIT = 6500
TEMPERATURE = 0.5
TOP_K = 35
TOP_P = 0.96

def main(output_dir):
    """
    Main function to run the pipeline. If an error occurs in inference or embedding generation, 
    this subprocess is restarted. If validation fails up to MAX_INFERENCE_RUNS times, 
    or any other error occurs, the document is marked as failed and the next document is processed.
    """
    global current_doc

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
            ## Try to run inference and validate output up to MAX_INFERENCE_RUNS times
            for run in range(MAX_INFERENCE_RUNS):
                print(f"Run {run+1}/{MAX_INFERENCE_RUNS}")
                run_inference(device, tokenizer, model)
                try:
                    validate_inference_output(f'{OUTPUT_DIR}/outputs/{current_doc["_id"]}/inference_output.json')
                except Exception as e:
                    print(f"Validation failed: {e}")
                    continue
            generate_embedding(device, tokenizer, model)
            upload_embeddings()
        except Exception as e:
            ## If an error occurs, mark the document as failed
            collection = MONGO_DB_CLIENT[DB_NAME][COLLECTION_NAME]
            collection.update_one({"_id": current_doc["_id"]}, {"$set": {"vectorizeFailed": True}})
    
    return 0


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

def run_inference(device, tokenizer, model):
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

        f_prompt = open(f'{OUTPUT_DIR}/inputs/prompt.txt', 'r', encoding='utf-8')
        f_transcript = open(f'{OUTPUT_DIR}/outputs/{current_doc["_id"]}/transcript.txt', 'r', encoding='utf-8')
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

        f_output = open(f'{OUTPUT_DIR}/outputs/{current_doc["_id"]}/inference_output.json', 'w', encoding='utf-8')
        f_output.write(decoded[0])
        f_output.close()
    except Exception as e:
        print(f"Error running inference: {e}")
        sys.exit(1)

def generate_embedding(device, tokenizer, model):
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

        f_summary = open(f'{OUTPUT_DIR}/outputs/{current_doc["_id"]}/inference_output.json', 'r', encoding='utf-8')
        summary = f_summary.read()
        f_summary.close()

        start_time = time.time()
        inputs = tokenizer(summary, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**inputs, output_hidden_states=True)
            hidden_states = hidden.hidden_states
        
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
    
def upload_embeddings():
    """
    Upload an embedding to Pinecone
    Args:
        embedding (torch.Tensor): The embedding to upload
    """
    global current_doc
    try:
        index = PINECONE_CLIENT.Index(PINECONE_INDEX_NAME)

        max_pool = torch.load(f'{OUTPUT_DIR}/outputs/{current_doc["_id"]}/max_pool.pt').to('cpu').numpy().tolist()
        avg_pool = torch.load(f'{OUTPUT_DIR}/outputs/{current_doc["_id"]}/avg_pool.pt').to('cpu').numpy().tolist()

        metadata_max = {"type": "max_pool", "title": current_doc["title"], "topics": current_doc["topics"]}
        metadata_avg = {"type": "avg_pool", "title": current_doc["title"], "topics": current_doc["topics"]}
        vectors = [
            {"id": current_doc["_id"], "values": max_pool, "metadata": metadata_max},
            {"id": current_doc["_id"], "values": avg_pool, "metadata": metadata_avg}
        ]
        index.upsert(vectors=vectors)
        print(f"Embedding uploaded for document: {current_doc['_id']}")
    except Exception as e:
        print(f"Error uploading embedding: {e}")

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

    print(f"Processing document: {current_doc['_id']}")

    return current_doc


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        sys.exit(1)