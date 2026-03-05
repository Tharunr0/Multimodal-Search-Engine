import streamlit as st
from src.data_loader import get_coco_data
from src.embedder import MultimodalEmbedder
from src.search import VectorStore

st.title("🖼️ Smart Image Search (COCO)")

# Config
ANN_PATH = "data/annotations/captions_val2017.json"
IMG_DIR = "data/val2017/"

@st.cache_resource
def init_system():
    embedder = MultimodalEmbedder()
    data = get_coco_data(ANN_PATH, IMG_DIR)
    
    # Indexing (Only runs once because of cache)
    paths = [d['path'] for d in data]
    vecs = embedder.encode_images(paths)
    
    store = VectorStore(vecs.shape[1])
    store.add(vecs, data)
    return embedder, store

embedder, store = init_system()

query = st.text_input("Search for anything...", "a person with an umbrella")

if query:
    q_vec = embedder.encode_text(query)
    results = store.search(q_vec)
    
    cols = st.columns(3)
    for i, res in enumerate(results):
        with cols[i % 3]:
            st.image(res['path'], width="stretch")
            st.caption(f"Original: {res['caption']}")