from API_KEY import CLAUDE_API_KEY
import anthropic
import re

client = anthropic.Anthropic(
    api_key=CLAUDE_API_KEY,
)


Prompt = "Below is a sentence. Please rate the sentence from 0 (most negative) to 10 (most positive) according to its sentiment."
CoT_prompt = 'Before providing the rating, elaborate on the reasoning process that led to the rating. Then, provide the rating as a single integer surrounded by three double quotes """. For example, if the rating is 7, the response should be """7""".'
direct_prompt = 'Then, provide the rating as a single integer surrounded by three double quotes """. For example, if the rating is 7, the response should be """7""".'

def query_Claude(input_sentence, CoT=False):
    if CoT:
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=200,
            system=Prompt + "\n" + CoT_prompt,
            messages=[
                {"role": "user", "content": input_sentence}
            ]
        )
    else:
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=200,
            system=Prompt + "\n" + direct_prompt,
            messages=[
                {"role": "user", "content": input_sentence}
            ]
        )
    response_text = message.content[0].text
    match = re.search(r'"""(\d{1,2})"""', response_text)
    if match:
        rating = int(match.group(1))
        return rating
    else:
        raise ValueError("Rating not found in the response")

input_sentence = "The weather is really nice today."
rating = query_Claude(input_sentence, CoT=False)
print(f"The sentiment rating is: {rating}")