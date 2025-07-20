import torch
from sentence_transformers import SentenceTransformer
from torch.nn.functional import normalize

# Model constant.
MODEL_ID = "Snowflake/snowflake-arctic-embed-m-v1.5"

# Load the model.
model = SentenceTransformer(MODEL_ID)

def load_model(model_id: str="Snowflake/snowflake-arctic-embed-m-v1.5"):
    model = SentenceTransformer(MODEL_ID)
    return model

def embedding_candidate_queries(model, candidate_queries):
    """
    提前编码好candidate_queries
    """
    query_embeddings = model.encode(candidate_queries)
    return query_embeddings


def my_retrieval_top_k(queries, candidate_querys, candidate_querys_embedding, retrieval_model, k=2):
    model = retrieval_model
    query_embedding = model.encode(queries, prompt_name="query")

    query_embeddings_256 = normalize(torch.from_numpy(query_embedding)[:, :256])
    document_embeddings_256 = normalize(torch.from_numpy(candidate_querys_embedding)[:, :256])
    scores_256 = query_embeddings_256 @ document_embeddings_256.T

    for query, query_scores in zip(queries, scores_256):
        doc_score_pairs = sorted(zip(candidate_querys, query_scores), key=lambda x: x[1], reverse=True)
        
    final_answer = [i[0] for i in doc_score_pairs[:k]]
    
    return final_answer
    


# Pretty-print the results.
# for query, query_scores in zip(queries, scores_256):
#     doc_score_pairs = sorted(zip(documents, query_scores), key=lambda x: x[1], reverse=True)
#     print(f'Query: "{query}"')
#     for document, score in doc_score_pairs:
#         print(f'Score: {score:.4f} | Document: "{document}"')
#     print()

if __name__ == "__main__":
    queries = ['what is snowflake?']
    documents = ['The Data Cloud!', 'Mexico City of Course!', "dasdasdas"]
    model = load_model()
    candidate_queries_embedding = embedding_candidate_queries(model, documents)

    my_retrieval_top_k(queries, documents, candidate_queries_embedding)
