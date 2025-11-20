Here is a natural language math proof, its breakdown conditions * conclusions and a Lean 4 lemma.  
Compare the conditions/conclusions with the headers/propositions of the Lean 4 lemma, matching them one by one to decide whether the Lean 4 lemma is an appropriate formalization of the mathematical statement.  
Assign exactly one of three tags: **Perfectly match**, **Minor inconsistency**, or **Major inconsistency**.  


Also, take note that:\
- Perfectly match: The Lean formalization correctly and completely captures all logical and mathematical meaning from the natural language.\
    It is fine if the Lean formalization contains extra, logically consistent details (e.g., extra variable type declarations, theorem names,\
    or auxiliary conditions) that are not  stated in the natural language. 
    But every statement mentioned in the natural language should be contained in lean formalization.
        -> additional constraints like "e>0" not mentioned on natural language statements ("e is real") are fine
        -> extra conditions are fine
        -> do not care about order "x=a+b" same as "a+b=x"

    - Minor inconsistency: The Lean formalization's logical meaning is similar to the natural language,\
    but there are slight structural or notational differences that are not direct translations. For instance, variable names are different.

    - Major inconsistency: The Lean formalization either misses a key logical component from the natural language or introduces a contradicting logical component.\

    - Ignore "import", "set_option", and "open" code lines

    - Focus solely on semantic meaning. The code already passed the Lean compilar and there are no syntactic errors. Do not comment or evaluate syntactic structures. 


**Stop immediately** after evaluating all pairs. Do **not** summarize or analyze further. 

Some example: {one_shot}

-----------------

Question: {informal_prefix_en}

Natural language: {math_cond}

Lean: {formal_statement}

Provide a component-by-component analysis and evaluation. Your final output should be a JSON object with the following structure:

```json
{{
    "evaluation": [
        "[evaluation 1]",
        "[evaluation 2]",
        ...
    ],
    "feedback": [
        "[Feedback for component 1]",
        "[Feedback for component 2]",
    ...
    ]
}}
```

** Important: evaluation must be among Perfectly match/Major inconsistency/Minor inconsistency


