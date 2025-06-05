

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
Rewrite the entire sentence with significant changes in structure and wording, while ensuring that the overall meaning remains the same.
SENTENCE: {sentence}
THE REVISED SENTENCE:
"""

def revise_instruct(data,change_scope):
    pass
