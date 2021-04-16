pip install git+https://github.com/huggingface/transformers
pip install git+https://github.com/huggingface/datasets

rm AtomicCards.json
wget https://mtgjson.com/api/v5/AtomicCards.json

rm -rf dataset tokenizer
mkdir dataset tokenizer saved
