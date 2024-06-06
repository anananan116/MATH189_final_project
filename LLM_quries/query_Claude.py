import asyncio
import aiohttp
import re
from API_KEY import CLAUDE_API_KEY
import pandas as pd

BASE_URL = "https://api.anthropic.com/v1/messages"

async def query_Claude(session, input_sentence, CoT=False):
    headers = {
        "X-API-Key": CLAUDE_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    Prompt = "Below is a sentence. Please rate the sentence from 0 (most negative) to 10 (most positive) according to its sentiment. Your answer should ONLY be an integer in the range of 0-10! You should still provide a rating even if the sentence is neutral/ambiguous/incomplete... Just do your best to give a rating based on the information provided."
    CoT_prompt = 'Before providing the rating, elaborate on the reasoning process that led to the rating, limit the analysis within ~70 words. Then, provide the rating as a single integer surrounded by three double quotes """. For example, if the rating is 7, the response should be """7""".'
    direct_prompt = 'Then, provide the rating as a single integer surrounded by three double quotes """. DO NOT include any reasoning or any extra content that is not asked for. For example, if the rating is 7, the response should be """7""".'

    if CoT:
        data = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 400,
            "system": Prompt + "\n" + CoT_prompt,
            "messages": [
                {"role": "user", "content": input_sentence}
            ]
        }
    else:
        data = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 200,
            "system": Prompt + "\n" + direct_prompt,
            "messages": [
                {"role": "user", "content": input_sentence}
            ]
        }

    async with session.post(BASE_URL, headers=headers, json=data) as response:
        response_json = await response.json()
        print(response_json)
        response_text = response_json['content'][0]['text']
        match = re.search(r'"""(\d{1,2})"""', response_text)
        if match:
            rating = int(match.group(1))
            return rating
        else:
            return None

async def query_all_claude(all_text):
    async with aiohttp.ClientSession() as session:
        tasks = [query_Claude(session, text, CoT=False) for text in all_text]
        responses_Claude = await asyncio.gather(*tasks)
        return responses_Claude

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    test_text = pd.read_csv("temp.csv")
    all_text = test_text['text'].tolist()
    responses_Claude = asyncio.run(query_all_claude(all_text))
    test_text['Claude_opus'] = responses_Claude
    print(test_text)
    test_text.to_csv("temp_1.csv", index=False)