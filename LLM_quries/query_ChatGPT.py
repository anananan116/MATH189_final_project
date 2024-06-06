from API_KEY import OPENAI_API_KEY
import openai
import re

openai.api_key = OPENAI_API_KEY

Prompt = "Below is a sentence. Please rate the sentence from 0 (most negative) to 10 (most positive) according to its sentiment."
CoT_prompt = 'Before providing the rating, elaborate on the reasoning process that led to the rating. Then, provide the rating as a single integer surrounded by three double quotes ". For example, if the rating is 7, the response should be "7".'
direct_prompt = 'Then, provide the rating as a single integer surrounded by double quotes ". For example, if the rating is 7, the response should be "7".'

def query_ChatGPT(input_sentence, CoT=False):
    if CoT:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=
            [
                {"role": "system", "content": Prompt},
                {"role": "system", "content": CoT_prompt},
                {"role": "user", "content": input_sentence}
            ],
            max_tokens=400
        )
    else:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=
            [
                {"role": "system", "content": Prompt},
                {"role": "system", "content": direct_prompt},
                {"role": "user", "content": input_sentence}
            ],
            max_tokens=50
        )
    #print(response.choices[0].message.content)
    response_text = response.choices[0].message.content
    match = re.search(r'"""(\d{1,2})"""', response_text)
    if match:
        rating = int(match.group(1))
        return rating
    else:
        return None

if __name__ == "__main__":
    input_sentence = "The weather is really great today."
    rating = query_ChatGPT(input_sentence, CoT=True)
    print(f"The sentiment rating is: {rating}")
