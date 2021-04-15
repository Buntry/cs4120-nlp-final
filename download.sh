pip install git+https://github.com/huggingface/transformers
pip install git+https://github.com/huggingface/datasets

rm run_mlm.py
wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/language-modeling/run_mlm.py

rm AtomicCards.json
wget https://mtgjson.com/api/v5/AtomicCards.json

rm -rf dataset models tokenizer
mkdir dataset models tokenizer
