import os
import gradio as gr
from webui import wrap_gradio_gpu_call
import modules.scripts
from modules import script_callbacks, shared, sd_hijack
from modules.textual_inversion.textual_inversion import Embedding

def run_split(embedding_name, vector_index):
    vector_index = int(vector_index)

    embedding = sd_hijack.model_hijack.embedding_db.word_embeddings[embedding_name]
    vector = embedding.vec[vector_index]
    
    new_name = f"{embedding_name}_{vector_index}"
    
    filename = os.path.join(shared.cmd_opts.embeddings_dir, f"{new_name}.pt")
    assert not os.path.exists(filename), f"file {filename} already exists"
    
    print(f"Saving new embedding to {filename}\n")
    
    split_embedding = Embedding(vector.unsqueeze(0), new_name)
    split_embedding.step = 0
    split_embedding.save(filename)
    

def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        embedding_name = gr.Dropdown(label='Embedding', elem_id="edit_embedding", choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys()), interactive=True)
        vector_index = gr.Number(label='Vector', value=0, step=1, interactive=True)
        
        go = gr.Button(value="Go", variant="primary")
        
        go.click(
            fn=run_split,
            inputs=[embedding_name, vector_index],
            outputs=[],
        )
    
    return [(ui, "Embedding Splitter", "embedding_splitter")]


script_callbacks.on_ui_tabs(add_tab)
