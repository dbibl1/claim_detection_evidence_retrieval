import flask
import pickle
import setup_claim_detection
from claim_detection_inference import output_not_checkworthy, get_checkworthy
import setup

from evidence_retrieval_inference import evidence_retrieval

global claim_model
global device
global claim_tokenizer
global evidence_model
global evidence_tokenizer
global evidence_index
global passages
global passage_id_map

claim_model, device, claim_tokenizer, evidence_model, evidence_tokenizer, evidence_index, passages, passage_id_map = setup.setup()


app = flask.Flask(__name__, template_folder='templates')



@app.route('/', methods=['GET', 'POST'])
def index():
    if flask.request.method == 'GET':
        return (flask.render_template('index.html'))

    if flask.request.method == 'POST':
        out = ''
        input = flask.request.form['tweet']
        queries = [input]
        input_tensors = claim_tokenizer(queries, truncation=True, max_length=512, padding='max_length',
                                        return_tensors='pt').to(device)
        checkworthy, not_checkworthy = get_checkworthy(input_tensors['input_ids'], claim_model, queries)
        print(checkworthy)
        print(output_not_checkworthy(not_checkworthy))

        if checkworthy:
            out += evidence_retrieval(checkworthy, evidence_model, evidence_tokenizer, evidence_index, passage_id_map)
            print(out)
        else:
            out += 'No checkworthy claims'

        out = out.split('\n')

        return flask.render_template('index.html', result=out, original_input={'Mobile Review': input})


if __name__ == '__main__':
    app.run()