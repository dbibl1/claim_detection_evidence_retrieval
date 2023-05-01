import torch
import time

def embed_queries(queries, model, tokenizer):
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            batch_question.append(q)

            if len(batch_question) == 32 or k == len(queries) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=512,
                    padding=True,
                    truncation=True,
                ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                output = model(**encoded_batch)
                embeddings.append(output.cpu())

                batch_question = []

    embeddings = torch.cat(embeddings, dim=0)
    print(f"Questions embeddings shape: {embeddings.size()}")

    return embeddings.numpy()

def get_evidence(passages, queries, top_passages_and_scores):
    # add passages to original data
    answers = {}
    for i, d in enumerate(queries):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        answers[d] = docs
    return answers

def report_answers(answers):
    out = ''
    for key in answers:
        out += f'Question: {key} \n\n'
#         print(f'Question: {key}:')
        for i, e in enumerate(answers[key]):
            title = e['title']
            text = e['text']
            out += f'Evidence {i+1}: \n\nTitle: {title} \nText: {text} \n\n'
        out += '---------------------------\n'

    return out

def evidence_retrieval(query, evidence_model, evidence_tokenizer, index, passage_id_map):
    questions_embedding = embed_queries(query, evidence_model, evidence_tokenizer)
    start_time_retrieval = time.time()
    top_ids_and_scores = index.search_knn(questions_embedding, 5)
    print(f"Search time: {time.time() - start_time_retrieval:.1f} s.")
    answers = get_evidence(passage_id_map, query, top_ids_and_scores)

    return report_answers(answers)