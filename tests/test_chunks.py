from services.chunks import get_text_chunks
import sys

# get file name from command line
filename = sys.argv[1]
with open(filename, "r") as f:
    text = f.read()

result = get_text_chunks(text=text, chunk_token_size=None)

i = 1
for text in result:
    print(f"{i}.{text}")
    i+=1