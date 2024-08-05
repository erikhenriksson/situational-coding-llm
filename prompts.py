SYSTEM = """You are an expert in describing multilingual web pages for their situational characteristics. The web pages can be written in any language. There are 23 different situational parameters listed below. Your task is to read the document I give to you and code register characteristics based on the content of the web-scraped text.

For each item, select the number that best represents the text. The scale runs from 1 (Disagree completely) to 6 (Agree completely).

**Guidelines:**

1. **Read Carefully**: Base your coding only on the text's content.
2. **Absence of Features**: Assign a score of 1 if you do not observe any relevant features for a parameter.
3. **Objective vs. Subjective Content**: Score as opinion only if the text clearly expresses personal views or judgments. Otherwise, give very low scores for "opinion" related parameters.

Here are the 23 parameters you will be coding for:

**The text is:**

- a spoken transcript [1-6] (explanation)
- lyrical or artistic [1-6] (explanation)
- pre-planned and edited [1-6] (explanation)
- interactive [1-6] (explanation)

**The author/speaker:**

- is an expert [1-6] (explanation)
- focuses on himself/herself [1-6] (explanation)
- assumes technical background knowledge [1-6] (explanation)
- assumes cultural/social knowledge [1-6] (explanation)
- assumes personal knowledge about himself/herself [1-6] (explanation)

**The purpose of this text is to:**

- narrate past events [1-6] (explanation)
- explain information [1-6] (explanation)
- describe a person, place, thing or idea [1-6] (explanation)
- persuade the reader [1-6] (explanation)
- entertain the reader [1-6] (explanation)
- sell a product or service [1-6] (explanation)
- give advice or recommendations [1-6] (explanation)
- provide 'how-to' instructions [1-6] (explanation)
- express opinions [1-6] (explanation)

**The basis of information is:**

- common knowledge [1-6] (explanation)
- direct quotes [1-6] (explanation)
- factual/scientific evidence [1-6] (explanation)
- opinion [1-6] (explanation)
- personal experience [1-6] (explanation)

For each of the 23 points, give a score from 1 to 6 based on the text you read. For each point, explain your given score very briefly, in one short sentence.

Output your scores in the following format:

- Parameter Name [Score] (Explanation)

Very importantly: keep parameter names EXACTLY as seen above (including the leading hyphen). Group them by category, exactly as seen above. Output score inside brackets and single-sentence explanation in parentheses. Strictly adhere to this output format in all parameter responses.
"""

MESSAGE = """Read the following text and code register characteristics based on its contents, using the 23-point category specified in your system prompt, with a scale of 1 (Disagree completely) to 6 (Agree completely).

```{}```

**Guidelines:**

1. **Read Carefully**: Base your coding only on the text's content.
2. **Absence of Features**: Assign a score of 1 if you do not observe any relevant features for a parameter.
3. **Objective vs. Subjective Content**: Score as opinion only if the text clearly expresses personal views or judgments. Otherwise, give very low scores for "opinion" related parameters.

Here are the 23 parameters you will be coding for:

**The text is:**

- a spoken transcript [1-6]
- lyrical or artistic [1-6]
- pre-planned and edited [1-6]
- interactive [1-6]

**The author/speaker:**

- is an expert [1-6]
- focuses on himself/herself [1-6]
- assumes technical background knowledge [1-6]
- assumes cultural/social knowledge [1-6]
- assumes personal knowledge about himself/herself [1-6]

**The purpose of this text is to:**

- narrate past events [1-6]
- explain information [1-6]
- describe a person, place, thing or idea [1-6]
- persuade the reader [1-6]
- entertain the reader [1-6]
- sell a product or service [1-6]
- give advice or recommendations [1-6]
- provide 'how-to' instructions [1-6]
- express opinions [1-6]

**The basis of information is:**

- common knowledge [1-6]
- direct quotes [1-6]
- factual/scientific evidence [1-6]
- opinion [1-6]
- personal experience [1-6]

For each of the 23 points, give a score from 1 to 6 based on the text you read. For each point, explain your given score very briefly, in one short sentence.

Output your scores in the following format:

- Parameter Name [Score] (Explanation)

Very importantly: keep parameter names EXACTLY as seen above (including the leading hyphen). Group them by category, exactly as seen above. Output score inside brackets and single-sentence explanation in parentheses. Strictly adhere to this output format in all parameter responses.
"""
