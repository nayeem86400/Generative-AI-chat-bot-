import chainlit as cl
# from transformers import BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
# import torch
# from langchain import HuggingFacePipeline
import os
from create_vector_db import create_vector_db
import glob
from llm_config import load_llm, final_result
import time
# import json


def check_for_pdfs(directory):
    # Use glob to search for PDF files in the directory
    pdf_files = glob.glob(os.path.join(directory, '*.pdf'))

    # Check if any PDF files were found
    if pdf_files:
        return True
    else:
        return False


success_db = -1
rag_flag = False
DB_FAISS_PATH = 'vectorstore/db_faiss'
data_path = 'data/pdf/'
model_id = 'nuk091/Llama-2-7b-chat-finetune_OLScience-guanaco-format'

llm = load_llm(model_id)


@cl.on_message
async def main(message: cl.Message):
    global success_db, rag_flag

    # Extract the content of the message
    print(f"Success Db value {success_db}")
    if not rag_flag:
        print("Im Notttt RAG.....................e")

        user_input = message.content
        response = llm(user_input)

        # print(response)

        print(f"Model response: {response}")
        generated_text = response[0]['generated_text'] if isinstance(
            response, list) and 'generated_text' in response[0] else response

        await cl.Message(content=generated_text, author="Model").send()
    elif rag_flag:
        if success_db == 1:
            print("Im here.....................e")
            user_input = message.content
            # response = llm(user_input)
            # qa_chain =  retrieval_qa_chain(llm, prompt, db)
            response = final_result(user_input, llm)
            print(response['source_documents'][0].metadata)
            # response = json.loads(response)
            metadata_list = response['source_documents'][0]
            # for key, value in temp.items():
            #     print(f"{key}: {value}")
            await cl.Message(content=f"{response['result']} \n\n{metadata_list}", author="Model").send()

            # print(f"Model response: {model_response}")


@cl.action_callback("upload_button")
async def on_action(action):
    global success_db, rag_flag

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=" ",
            accept=["*/*"],
            max_size_mb=1024,
            max_files=200,
            timeout=1800
        ).send()

        if check_for_pdfs:
            # os.makedirs(local_path)
            rag_flag = True
            for pdf_file in files:
                print(pdf_file.name)
                pdf_file_path = os.path.join(data_path, pdf_file.name)
                with open(pdf_file_path, 'wb') as buffer:
                    with open(pdf_file.path, 'rb') as f:
                        buffer.write(f.read())

            time.sleep(2)
            success_db = create_vector_db(data_path, DB_FAISS_PATH)

            # files = glob.glob(os.path.join(local_path, '*'))
            # # Remove each file
            # for file in files:
            #     try:
            #         os.remove(file)
            #         print(f"Removed file: {file}")
            #     except Exception as e:
            #         print(f"Error removing file: {file}, {e}")


@cl.on_chat_start
async def start():
    # Sending an action button within a chatbot message
    actions = [
        cl.Action(name="upload_button", value="start_upload",
                  description="Upload a File")
    ]

    await cl.Message(content="Click the button to upload a file:", actions=actions).send()
