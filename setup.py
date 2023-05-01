import setup_claim_detection
import setup_evidence_retrieval

def setup():
    claim_model, device, claim_tokenizer = setup_claim_detection.setup()
    evidence_model, evidence_tokenizer, index, passages, passage_id_map = setup_evidence_retrieval.setup()

    return claim_model, device, claim_tokenizer, evidence_model, evidence_tokenizer, index, passages, passage_id_map