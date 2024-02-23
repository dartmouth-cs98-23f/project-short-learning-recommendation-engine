import torch
import os
import time
import json
import dotenv

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from transformers import AutoTokenizer, AutoModelForCausalLM
from pinecone import Pinecone


shots_transcripts = "../data/transcripts/processed"
transcripts_dir = "../data/transcripts/processed"
prompt_path = "../data/prompts/prompt.txt"
oneshots_dir = "../data/oneshots"
dotenv.load_dotenv(dotenv_path="../env")

def abridge_transcript(transcript: str, chars: int) -> str:
    """ Abridge a transcript to a certain number of characters
    Args:
        transcript (str): The transcript to abridge
        chars (int): The number of characters to abridge to
    Returns:
        str: The abridged transcript
    """

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


def build_message(shots: list, target, prompt, shots_char_limit: int, input_char_limit: int) -> list:
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

    for shot in shots:
        f_oneshot_transcript = open(f'{shots_transcripts}/{shot}.txt', 'r', encoding='utf-8')
        f_oneshot = open(f'{oneshots_dir}/{shot}.txt', 'r', encoding='utf-8')
        
        messages.append({"role": "user", "content": f'Transcript:\n{abridge_transcript(f_oneshot_transcript.read(), shots_char_limit)}\nInstruction:\n{prompt}'})
        messages.append({"role": "assistant", "content": f_oneshot.read()})

        f_oneshot.close()
        f_oneshot_transcript.close()

    ## Append final message
    messages.append({"role": "user", "content": f'Transcript:\n{abridge_transcript(target, input_char_limit)}\nInstruction:\n{prompt}'})
 
    return messages

def vectorize_pipeline(doc):
    ## Setup Config
    with open('config/videos.json') as config_file:
        videos = json.load(config_file)
    with open('config/name_to_url.json') as config_file:
        name_to_url = json.load(config_file)
    
    folder_no = doc["_id"]
    print(f"Folder Number: {folder_no}")
    if os.path.exists(f'../data/outputs/{folder_no}'):
        print(f"Folder {folder_no} already exists. Skipping...")
        return
    os.makedirs(f'../data/technigala/{folder_no}/text', exist_ok=True)
    os.makedirs(f'../data/technigala/{folder_no}/embeddings', exist_ok=True)
    os.makedirs(f'../data/technigala/{folder_no}/results', exist_ok=True)
    outputs_dir = f'../data/technigala/{folder_no}'
    prompt = open(prompt_path, "r").read()
    shots = ["mixtral8x7b", "full-stack"]
    
    # Load models
    torch.cuda.empty_cache()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(device)

    # Write some metadata
    with open(f'{outputs_dir}/metadata.txt', 'a') as f:
        if os.stat(f'{outputs_dir}/metadata.txt').st_size == 0:
            f.write(f"#####################\n###### METADATA ######\n#####################\n\n")
            f.write(f"Model: mistralai/Mistral-7B-Instruct-v0.1\n")
            f.write(f"Time: {time.time()}\n")
            f.write(f"Videos: {videos}\n")
            f.write(f"Shots: {shots}\n")
            f.write(f"Prompt: {prompt}\n")
    
    target = doc["title"]
    # Run Inference
    run_inference(model, outputs_dir, transcripts_dir, shots, tokenizer, device, prompt, target)
    create_embeddings(outputs_dir, tokenizer, model, device, target)
    upload_to_pinecone(outputs_dir, name_to_url, doc, target)
    

def run_inference(model, outputs_dir, shots, tokenizer, device, prompt, target):
    """ Run inference on the model
    Args:
        model (AutoModelForCausalLM): The model to run inference on
        outputs_dir (str): The directory to save the outputs to
        transcripts_dir (str): The directory containing the transcripts
        shots (list): The shots to include in the message
    """

    metadata = open(f'{outputs_dir}/metadata.txt', 'a')
    metadata.write(f"\n\n###################\n###### INFERENCE ######\n###################\n")
    f_message = open(f'{outputs_dir}/messages.txt', 'w', encoding='utf-8')
    temperature = 0.5
    top_k = 35
    top_p = 0.96

    metadata.write(f"Temperature: {temperature}\nTop_k: {top_k}\nTop_p: {top_p}\n")
    metadata.close()
    metadata = open(f'{outputs_dir}/metadata.txt', 'a')

        ## open all files in data/transcripts/processed. For each file, create a message using as many shots as given above. 
    global_start = time.time()
    metadata.write(f"Processing: {target}\n")
    with open(f'{outputs_dir}/transcript.txt', 'r', encoding='utf-8') as f_target_transcript:
        messages = build_message(shots, target, prompt, 8000, 14500)
        f_message.write(f'{target}: {messages}\n\n')
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)    
    metadata.write(f"    Tokens: {len(inputs[0])}\n")
    print(f"    Tokens: {len(inputs[0])}")
    ## An idea for later, do not include the oneshot prompt if we are only directly using transcript embedding
    # start_token = len(inputs[0]) 

    with torch.no_grad():
        start_time = time.time()
        text = model.generate(inputs, max_new_tokens=3500, do_sample=True, pad_token_id=tokenizer.eos_token_id, temperature=temperature, top_k=top_k, top_p=top_p)
        decoded = tokenizer.batch_decode(text)
        print(f'Decoding finished: {target} in {round(time.time() - start_time, 3)} seconds')   
        metadata.write(f"    Decoding took: {round(time.time() - start_time, 3)} seconds\n")
    ## obtain all tokens after the second "[/INST]" and remove the </s> token. Write this as our output.
    with open(f'{outputs_dir}/text/{target}.json', 'w', encoding='utf-8') as f:
        f.write(decoded[0].split('[/INST]')[-1][1:-4])

    ## Free up memory
    del text
    del decoded
    torch.cuda.empty_cache()
    print(f'cuda memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB', f'cuda memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB')
    metadata.write(f'    cuda memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB, cuda memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB\n')

    metadata.write(f'Global runtime: {round(time.time() - global_start, 3)} seconds')
    metadata.close()
    f_message.close()

def create_embeddings(outputs_dir, tokenizer, model, device, target):
    """ Create embeddings from the outputs of the model
    Args:
        outputs_dir (str): The directory containing the outputs
    """
    
    metadata = open(f'{outputs_dir}/metadata.txt', 'a')

    print(f'cuda memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB', f'cuda memory cached: {torch.cuda.memory_cached()/1024**3:.2f} GB')

    start_time = time.time()
    with open(f'{outputs_dir}/text/{target}.json', 'r', encoding='utf-8') as f:
        inputs = tokenizer(f.read(), return_tensors="pt").to(device)     
    with torch.no_grad():
        hidden = model(**inputs, output_hidden_states=True)
    print(f'Hidden states finished: {target} in {round(time.time() - start_time, 3)} seconds')
    metadata.write(f"Target: {target}, Hidden states took: {round(time.time() - start_time, 3)} seconds\n")

    ## Write hidden states
    tensor_t = hidden.hidden_states[-1].transpose(1,2)
    max_pool = torch.nn.functional.max_pool1d(tensor_t, tensor_t.shape[2]).transpose(1, 2).squeeze()
    avg_pool = torch.nn.functional.avg_pool1d(tensor_t, tensor_t.shape[2]).transpose(1, 2).squeeze()
    print(max_pool.shape)
    torch.save(max_pool, f'{outputs_dir}/embeddings/max_{target}.pt')
    torch.save(avg_pool, f'{outputs_dir}/embeddings/avg_{target}.pt')

    print(f'Target: {target} finished. Wrote to file.')
    metadata.write(f'    Target: {target} tensor finished. Wrote to file.\n')

    del inputs
    del hidden
    del tensor_t
    del max_pool
    del avg_pool
    print(f'cuda memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB', f'cuda memory cached: {torch.cuda.memory_cached()/1024**3:.2f} GB')
    torch.cuda.empty_cache()

    metadata.close()

def upload_to_pinecone(outputs_dir, name_to_url, doc, target):
    pc = Pinecone(api_key=os.environ["PINECONE_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX"])
    
    # Prepare embeddings for upload
    path_to_embeddings = f'{outputs_dir}/embeddings'
    mode = "max"
    embedding_file = f'{path_to_embeddings}/{mode}_{target}.pt'
    vectors = []
    if not embedding_file.endswith('.pt'):
        return
    if mode not in embedding_file:
        return

    name = embedding_file[4:-3]
    metadata = {"name": name, "url": doc["youtubeURL"], "title": doc["title"]}
    tensor = torch.load(f'{path_to_embeddings}/{embedding_file}').to('cpu').numpy().tolist()
    vectors.append({"values": tensor, "id": name, "metadata": metadata})
    print(f'Loaded {name}, metadata: {metadata}')
    del tensor
    index.upsert(vectors=vectors)

    torch.cuda.empty_cache()