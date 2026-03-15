from tokenizer.tokenizer import Tokenizer

texts = [
    "I love AI",
    "AI is amazing",
    "I hate bugs"
]

tokenizer = Tokenizer()

tokenizer.build_vocab(texts)

print(tokenizer.word2idx)

print(tokenizer.encode("I love AI"))