import faiss
import pickle

class VectorStore:
    def __init__(self, dim, use_ivf=False):
        if use_ivf:
            # Better for 100k+ images: clusters the space into 100 regions
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, 100, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def add(self, vecs, meta):
        if not self.index.is_trained:
            self.index.train(vecs)
        self.index.add(vecs)
        self.metadata.extend(meta)

    def search(self, query_vec, k=5):
        distances, indices = self.index.search(query_vec, k)
        return [self.metadata[i] for i in indices[0]]