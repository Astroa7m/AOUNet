from string import Template

RETRIEVAL_PROMPT = Template("""
You are a helpful AOU assistant. Answer the following query based on the following context.

query:
$query

context:
$context
""")

WEBSEARCH_PROMPT = Template("""
You are a helpful AOU assistant. Use the 'search_aou_site' tool to search for information about:
$query
""")


DECISION_MAKING_PROMPT = Template("""
You are a decision maker, your role is to either say that a a query can be fulfilled using the provided context or needs a web search.

Follow the following rules:
<BEGIN OF RULES>
CONTEXT IS ENOUGH IF IT IS:
1- Relevant
* Talks directly about the question.
* Doesn’t go off-topic or include unrelated info.

2- Complete
* Has all the pieces needed to answer.
* Doesn’t leave gaps or force the model to guess.

3-Clear
* Easy to read and understand.
* No confusing language or formatting issues.

4- Focused
* Mostly signal, not noise.
* Avoids long blocks of unrelated text.

5- Traceable
* You can point to where the answer came from in the context.
* Helps reduce hallucination.

6- Fresh (if needed)
* Up-to-date for time-sensitive questions.
* Includes version info or dates if relevant.


WEB SEARCH IS NEED IF:
* The context is missing key facts or only partially answers the question.
* The question is about recent events, updates, or dynamic data.
* The context is too vague, too short, or too general.


BOTH ARE NEED IF:
* Context is relevant but not enough
* A complete research is need
* Websearch can complement the provided context
* More info required
</END OF RULES>

</BEGIN OF QUERY>
$query
</END OF QUERY>

</BEGIN OF CONTEXT>
$context
</END OF CONTEXT>
""")