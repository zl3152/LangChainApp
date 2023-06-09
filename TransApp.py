from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
from langchain import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

load_dotenv()
api_key = os.environ.get('OpenAI_API_KEY')

# first initialize the large language model
llm = OpenAI(
	openai_api_key=api_key,
	model_name="text-davinci-003"
)


#Using FewShotPrompt
from langchain import FewShotPromptTemplate

#Set up the examples 
examples = [
    {
        "query": "Translate the following English quote to classical Chinese: 'To be or not to be, that is the question.'",
        "answer": "生存与否，是个问题。"
    },
    {
        "query": "Translate the following classical Chinese quote to English: '己所不欲，勿施于人'",
        "answer": "Do not do unto others what you would not have them do unto you."
    },
    {
        "query": "Translate the following English quote to classical Chinese: 'The best preparation for tomorrow is doing your best today.'",
        "answer": "凡事预则立，不预则废。"
    },
    {
        "query": "Translate the following classical Chinese quote to English: '知之为知之，不知为不知，是知也'",
        "answer": "To know when you know, and to know when you do not know, that is wisdom."
    },

    {
        "query": "Translate the following English quote to classical Chinese: 'The only way to do great work is to love what you do.'",
        "answer": "唯有热爱所做，始能成就伟业。"
    }
]

# create a example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """The following are excerpts from conversations with an AI assistant. 
The assistant accurately translates English to Chinese, or Chinese to English text
while preserving the original style and meaning. 
Here is an example:
"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)



example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=50  # this sets the max length that examples should be
)

# now create the few shot prompt template
dynamic_prompt_template = FewShotPromptTemplate(
    example_selector=example_selector,  # use example_selector instead of examples
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n"
)

llm.temperature = 1.2
print(llm(dynamic_prompt_template.format(query="When you have a hammer, everything looks like a nail")))