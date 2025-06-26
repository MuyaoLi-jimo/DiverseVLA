from diversevla.dataset import fm_server
import os

ONE_PROMPT = """
Please select one verb, noun, or adjective from the sentence below at random and replace it with a synonym or a similar word, ensuring that the overall meaning of the sentence remains unchanged.
SENTENCE: {sentence}
THE REVISED SENTENCE:
"""

TWO_PROMPT = """
Choose either two random words (a verb, noun, or adjective) or one phrase in the sentence below and replace them with similar expressions, while keeping the sentence’s overall meaning intact.
SENTENCE: {sentence}
THE REVISED SENTENCE:
"""

ALL_PROMPT = """
Rewrite the entire sentence with significant changes in structure and wording, while ensuring that the overall meaning remains the same. you can change like: please xxx? or OpenVLA, can you xxx?
SENTENCE: {sentence}
THE REVISED SENTENCE:
"""

PROMPT_MAP={
    "one":ONE_PROMPT,
    "two":TWO_PROMPT,
    "all": ALL_PROMPT,
}

def revise_instruct(data,change_scope,model_id="gpt-4o"):
    if change_scope == "zero":
        data["new_instroduction"] = data["instruction"]
        return [data]
    input_data = {
        "messages":[fm_server.create_user_message(input_type="text",user_prompt=PROMPT_MAP[change_scope].format(sentence=data["instruction"]))],
        "model_id":model_id,
        "temperature":0.7,
    }
    output = fm_server.generate_responce(data=input_data,api_key=os.getenv("AIHUBMIX_API"),base_url="https://aihubmix.com/v1")
    data["new_instroduction"] = output["content"]
    return [data]


if __name__ == "__main__":
    data = {"instruction":"pick up the can"}
    d = revise_instruct(data,change_scope="all",model_id="gpt-4o")
    print("output: ",d)