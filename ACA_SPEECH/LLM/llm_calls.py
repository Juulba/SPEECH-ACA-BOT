'''
This file handles the logic of making calls to LLMs, either with the Mistral API or Willma API.
The separate LLM approaches should have separate files with logic for their internal calls. This file is simply a middle man, to avoid adding all of the different llm calls logic into the bot file and have to constantly update it.
'''

from LLM.willma import willma_call, get_simple_willma_call
from LLM.mistral import get_simple_mistral_call, mistral_call
from LLM.gpt import get_simple_gpt_call, gpt_call
from setup.constants import MODEL, TEMP_CONVO, TEMP_CHECK

# Normal Calls

async def make_call(system_prompt, user_prompt, llm, max_tokens=1024):
    tokens = len(system_prompt.split()) + len(user_prompt.split())

    if "mistral" in llm.lower() and MODEL != "mistralai/Mistral-Small-3.2-24B-Instruct-2506":
        answer = await mistral_call(
            user_prompt = user_prompt,
            system_prompt = system_prompt,
            model = MODEL,
            temperature = TEMP_CONVO,
            max_tokens = max_tokens
        )
    elif "gpt" in llm.lower() and MODEL != "openai/gpt-oss-120b":
        answer = await gpt_call(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=MODEL,
            temperature=TEMP_CONVO,
            max_tokens=max_tokens
        )
    else:
        answer = await willma_call(
            user_prompt = user_prompt,
            system_prompt = system_prompt,
            model=MODEL,
            temperature = TEMP_CONVO,
            max_tokens = max_tokens
        )

    if answer is not None:
        tokens += len(answer.split())
        return [answer, tokens]

    return ["An error occurred.", 0]


# Simple Calls
async def make_simple_call(prompt, llm, max_tokens=32):
    if "mistral" in llm.lower() and MODEL != "mistralai/Mistral-Small-3.2-24B-Instruct-2506":
        answer = await get_simple_mistral_call(
            prompt=prompt,
            model=MODEL,
            temperature=TEMP_CHECK,
            max_tokens=max_tokens
        )
    elif "gpt" in llm.lower() and MODEL != "openai/gpt-oss-120b":
        answer = await get_simple_gpt_call(
            prompt=prompt,
            model=MODEL,
            temperature=TEMP_CHECK,
            max_tokens=max_tokens
        )
    else:
        answer = await get_simple_willma_call(
            prompt=prompt,
            model=MODEL,
            temperature=TEMP_CHECK,
            max_tokens=max_tokens
        )


    if answer is not None:
        tokens = len(prompt.split()) + len(answer.split())
        return [answer, tokens]

    return ["An error occurred.", 0]

