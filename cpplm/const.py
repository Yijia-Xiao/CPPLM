template = """<startofstring> <Instruction>
{instruction}

<Question>
{input}

<Answer>
{output}

<Response>
{cleaned_output} <endofstring>"""

inference_template = """<startofstring> <Instruction>
{instruction}

<Question>
{input}

<Answer>"""

INSTRUCTION = "Answer this question truthfully"

NUM_EPOCH = 20