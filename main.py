import os
from llmware.library import Library
from llmware.setup import Setup
from llmware.configs import LLMWareConfig
from llmware.parsers import Parser
import chainlit as cl
from chainlit.types import AskFileResponse 
from llmware.retrieval import Query
from llmware.prompts import Prompt

config = LLMWareConfig()
LLMWareConfig().set_active_db("sqlite")
LLMWareConfig().set_vector_db("chromadb")
library_name = "test"  # This should be defined outside the function if it's global
library = Library().create_new_library(library_name)  # This should also be defined outside if it's global
async def chunk(file):
    file_path = file.path  # Use the 'path' attribute that contains the file path
    library.add_file(file_path)
    library.install_new_embedding(from_hf=True,embedding_model_name="bge-base-en-v1.5", vector_db="chromadb", batch_size=200)


welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""


llm_model_name = "llmware/bling-phi-3-gguf"
prompter = Prompt().load_model(llm_model_name)


@cl.on_chat_start
async def start():
    await cl.Avatar(
        name="Chatbot",
        url="https://avatars.githubusercontent.com/u/128686189?s=400&u=a1d1553023f8ea0921fba0debbe92a8c5f840dd9&v=4",
    ).send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=500,
            timeout=180,
        ).send()
    msg = cl.Message(content=f"Processing `{files[0].name}`...", disable_feedback=True)
    await msg.send()

 
    await chunk(files[0])



    cl.user_session.set("counter", 0)

@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the counter from the user session
    counter = cl.user_session.get("counter")
    counter += 1
    cl.user_session.set("counter", counter)

    # Your existing model query and response logic
    user_query_text = message.content
    results = Query(library=library).semantic_query(user_query_text, result_count=200, embedding_distance_threshold=1.0)
    source = prompter.add_source_query_results(query_results=results)
    response = prompter.prompt_with_source(user_query_text, temperature=0.3, prompt_name="default_with_context")
    
    llm_response = response[0]['llm_response']
    response_message = cl.Message(content=llm_response)
    
    # Add source information as elements
    elements = [
        cl.Text(content=f"Source {i}: {res['file_source']} (Distance: {res['distance']})\nText: {res['text']}")
        for i, res in enumerate(results)
    ]
    response_message.elements = elements
    
    await response_message.send()
    prompter.clear_source_materials()

    # Send a message with the updated counter
    await cl.Message(content=f"You sent {counter} message(s)!").send()
    
""""
    #elements = []  # Initialize an empty list to collect elements outside the loop
    #label_list = []
    #for i, res in enumerate(query):
        source_documents = f"update: {i}, {res['file_source']}, {res['distance']}, {res['text']}"
        elements.append(cl.Text(content=source_documents, name=f"source_{i}", display="side", color='black'))
        label_list.append(f"Source {i}")

    # Update the response message with all elements and labels after the loop
    response_message.elements = elements
    response_message.content += "\n\nSources: " + ", ".join(label_list)
    await response_message.update()
"""