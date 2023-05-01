import setup
from claim_detection_inference import get_checkworthy, output_not_checkworthy
from evidence_retrieval_inference import embed_queries, get_evidence, report_answers, evidence_retrieval
import time
import setup_claim_detection

def main():
    claim_model, device, claim_tokenizer, evidence_model, evidence_tokenizer, index, passages, passage_id_map = setup.setup()

    queries = ['Barack Obama was the 46th president of the United States']

    input_tensors = claim_tokenizer(queries, truncation=True, max_length=512, padding='max_length', return_tensors='pt').to(device)
    checkworthy, not_checkworthy = get_checkworthy(input_tensors['input_ids'], claim_model, queries)
    print(checkworthy)
    print(output_not_checkworthy(not_checkworthy))

    if checkworthy:
        out = evidence_retrieval(checkworthy, evidence_model, evidence_tokenizer, index, passage_id_map)
        print(out)
    else:
        print('No checkworthy claims')

if __name__ == "__main__":
    main()




