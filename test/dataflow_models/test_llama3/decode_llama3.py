import sys

from llama3_tokenizer import Tokenizer

TOKENIZER_PATH = "tokenizer.model"

llama3_tokenizer = Tokenizer(TOKENIZER_PATH)


if __name__ == "__main__":

	if len(sys.argv) != 2:
		print(f"Error: Usage: python decode_llama3.py <token_id>")
		sys.exit(1)

	token_id = int(sys.argv[1])

	decoded = llama3_tokenizer.decode([token_id])
	print(f"Token ID: {token_id} == '{decoded}'")
