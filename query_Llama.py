from API_KEY import AIML_API_KEY
from openai import OpenAI
import re

client = OpenAI(
    api_key=AIML_API_KEY,
    base_url="https://api.aimlapi.com",
)

Prompt = "Below is a sentence. Please rate the sentence from 0 (most negative) to 10 (most positive) according to its sentiment."
CoT_prompt = 'Before providing the rating, elaborate on the reasoning process that led to the rating. Then, provide the rating as a single integer surrounded by three double quotes """. For example, if the rating is 7, the response should be """7""".'
direct_prompt = 'Then, provide the rating as a single integer surrounded by three double quotes """. For example, if the rating is 7, the response should be """7""".'

def query_Llama(input_sentence, CoT=False, size = 7):
    assert size in [7, 14, 70]
    if CoT:
        response = client.chat.completions.create(
            model=f"meta-llama/Llama-2-{size}b-chat-hf",
            messages=
            [
                {"role": "system", "content": Prompt},
                {"role": "system", "content": CoT_prompt},
                {"role": "user", "content": input_sentence}
            ],
            max_tokens=200
        )
    else:
        response = client.chat.completions.create(
            model=f"meta-llama/Llama-2-{size}b-chat-hf",
            messages=
            [
                {"role": "system", "content": Prompt},
                {"role": "system", "content": direct_prompt},
                {"role": "user", "content": input_sentence}
            ],
            max_tokens=200
        )

    response_text = response.choices[0].message.content
    match = re.search(r'"""(\d{1,2})"""', response_text)
    if match:
        rating = int(match.group(1))
        return rating
    else:
        raise ValueError("Rating not found in the response")

input_sentence = "The weather is really nice today."
rating = query_Llama(input_sentence, CoT=False)
print(f"The sentiment rating is: {rating}")
