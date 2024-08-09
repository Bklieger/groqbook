"""
Agent to generate book structure
"""

from ..inference import GenerationStatistics


def generate_book_structure(
    prompt: str, additional_instructions: str, model: str, groq_provider
):
    """
    Returns book structure content as well as total tokens and total time for generation.
    """
    completion = groq_provider.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": 'Write in JSON format:\n\n{"Title of section goes here":"Description of section goes here",\n"Title of section goes here":{"Title of section goes here":"Description of section goes here","Title of section goes here":"Description of section goes here","Title of section goes here":"Description of section goes here"}}',
            },
            {
                "role": "user",
                "content": f"Write a comprehensive structure, omiting introduction and conclusion sections (forward, author's note, summary), for a book. Only provide up to one level of depth for nested sections. Make clear titles and descriptions that have no overlap with other sections. It is very important that use the following subject and additional instructions to write the book. \n\n<subject>{prompt}</subject>\n\n<additional_instructions>{additional_instructions}</additional_instructions>",
            },
        ],
        temperature=0.3,
        max_tokens=8000,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    usage = completion.usage
    statistics_to_return = GenerationStatistics(
        input_time=usage.prompt_time,
        output_time=usage.completion_time,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_time=usage.total_time,
        model_name=model,
    )

    return statistics_to_return, completion.choices[0].message.content
