import json
import math
import re
from itertools import combinations

import json5
import networkx as nx

from .proof_graph import (
    Definition,
    TheoremCondition,
    build_proof_graph,
    extract_json_block,
    parse_llm_json,
    sanitize_backslashes,
)
from .utils import LLMManager


def util_run_prem_def_node_scorer_prompt(
    nl_stmt, fl_stmt, model_manager: LLMManager, logs: list
) -> tuple:
    prompt = create_prem_def_prompt(fl_stmt, nl_stmt)
    message = [{"role": "user", "content": prompt}]

    output, messages = model_manager.call_llm(message, logs=None, temperature=0)

    result = parse_llm_json(output)
    matches = match_evaluations(result["evaluation"])
    feedback = result["feedback"]

    lean_score = sugeno_integral(matches)

    return lean_score, feedback


def util_run_lemma_node_scorer_prompt(
    nl_stmt, fl_stmt, model_manager: LLMManager, logs: list
) -> tuple:
    # Step 1: Get mathematical conditions from translated_question
    first_prompt = create_first_prompt(nl_stmt)

    first_messages = [{"role": "user", "content": first_prompt}]

    first_result, messages = model_manager.call_llm(
        first_messages, logs=None, temperature=0
    )

    math_cond = first_result


    # Step 2: Compare formal statement with mathematical conditions
    second_prompt = create_second_prompt_json(nl_stmt, math_cond, fl_stmt)

    second_messages = [{"role": "user", "content": second_prompt}]
    second_result, messages = model_manager.call_llm(
        second_messages, logs=None, temperature=0
    )

    prompt2_output = second_result


    # Step 3: Extract evaluations and calculate score

    output = parse_llm_json(prompt2_output)
    matches, feedback = output["evaluation"], output["feedback"]

    matches = match_evaluations(matches)

    lean_score = sugeno_integral(matches)
    return lean_score, feedback


def run_node_scorer_prompt(item: dict, model_manager: LLMManager, logs: list) -> tuple:
    """
    Run semantic equivalence scoring of a node
    """
    nl_stmt = item.statement
    fl_stmt = item.formalization["lean_code"]
    lean_status = item.formalization["lean_pass"]
    lean_status_2 = (getattr(item, "solved_lemma", None) or {}).get("lean_verify", True)

    # should not hold as only nodes passing Lean syntax check can go through this semantic funditon
    if not lean_status:
        return 0, "Failed Lean syntax check already."

    # Handle each type of node separately

    # For Theorem conditions & Definitions (assumed/given, not proven)
    if isinstance(item, TheoremCondition) or isinstance(item, Definition):
        lean_score, feedback = util_run_lemma_node_scorer_prompt(
            nl_stmt, fl_stmt, model_manager, logs
        )

    # For Lemmas & Theorem statement (given conditions --> targeted conclusions)
    else:
        lean_score, feedback = util_run_prem_def_node_scorer_prompt(
            nl_stmt, fl_stmt, model_manager, logs
        )

    if not lean_status_2:
        lean_score = 0.5*lean_score
        feedback.append("The tactics contained syntactic mistakes, so score is cut by half.")

    return lean_score, feedback


def run_node_scorer_prompt_old(
    item: dict, model_manager: LLMManager, logs: list
) -> tuple:
    """
    Run semantic equivalence scoring of a node
    """
    nl_stmt = item.statement
    fl_stmt = item.formalization["lean_code"]
    lean_status = item.formalization["lean_pass"]

    # should not hold as only nodes passing Lean syntax check can go through this semantic funditon
    if not lean_status:
        return 0, "Failed Lean syntax check already."

    # Handle each type of node separately

    # For Theorem conditions & Definitions (assumed/given, not proven)
    if isinstance(item, TheoremCondition) or isinstance(item, Definition):
        prompt = create_prem_def_prompt(fl_stmt, nl_stmt)
        message = [{"role": "user", "content": prompt}]

        output, messages = model_manager.call_llm(message, logs=None, temperature=0)

        result = parse_llm_json(output)
        matches = match_evaluations(result["evaluation"])
        feedback = result["feedback"]

    # For Lemmas & Theorem statement (given conditions --> targeted conclusions)
    else:
        # Step 1: Get mathematical conditions from translated_question
        first_prompt = create_first_prompt(nl_stmt)

        first_messages = [{"role": "user", "content": first_prompt}]

        first_result, messages = model_manager.call_llm(
            first_messages, logs=None, temperature=0
        )
        # print(first_result)
        math_cond = first_result

        # Step 2: Compare formal statement with mathematical conditions
        second_prompt = create_second_prompt(nl_stmt, math_cond, fl_stmt)
        second_messages = [{"role": "user", "content": second_prompt}]
        second_result, messages = model_manager.call_llm(
            second_messages, logs=None, temperature=0
        )

        prompt2_output = second_result

        # Step 3: Extract evaluations and calculate score
        matches, feedback = extract_match_feedback_content(prompt2_output)

    lean_score = sugeno_integral(matches)

    return lean_score, feedback


# helper: extract conditions & conclusions from NL problem statement
def create_first_prompt(informal_prefix_en: str) -> str:
    """Create the first prompt for deriving mathematical condition from informal description"""
    one_shot = r"""I will give you a natural language statement in a math proof.  Break it down into conditions and conclusions, where:

            - Conditions are what used to derive others.
            - Conclusions are what derived from the conditions.
            - Stick to a clean, step-by-step format like this:

                Conditions:  
                - <condition 1>  
                - <condition 2>  
                Conclusions:  
                - <new derived fact>  

            Follow this example:
            * Natural language math proof *
            "Step 1:
            From the given conditions \(x^2 - 4 \geq 0\), we can deduce that x^2 \geq 4"

            * Condition-Conclusion breakdown*
            Step 1:
            Conditions:
            - \(x^2 - 4 \geq 0\)
            Conclusions:
            - x^2 \geq 4

                """

    prompt = f"""Now, please help me extract the conditions and conclusions for this problem in the same way (using specific mathematical formulas), without solving it:  
                [Natural language math proof]: {informal_prefix_en}

                Return the Condition-Conclusion breakdown only
                """

    return one_shot + prompt


# helper: given input NL problem statement + [conditions, conclusion] --> evaluate parts by parts
def create_second_prompt(
    informal_prefix_en, math_cond: str, formal_statement: str
) -> str:
    """Create the second prompt for assessing appropriateness of formal statement"""

    theorem_index = formal_statement.find("theorem ")
    if theorem_index != -1:
        # Return the part starting with "theorem "
        formal_statement = formal_statement[theorem_index:]
    else:
        # If "theorem " is not found, return the original string
        formal_statement = formal_statement

    # Fix escape sequences by using raw string (r prefix) or double backslashes
    one_shot = r"""Given conditions and conclusions of natural language statement and a lemma in Lean,
     compare each condition to a corresponding Lean header and each conclusion to a Lean proposition, one by one:

    1. **\( q \) is a natural number greater than 1**:  
   - Math: \( q \in \mathbb{N}, q > 1 \).  
   - Lean: `(hq : 1 < q)`.  
   - Detailed explanation: [[ ]].
   - Match: \box{Perfectly match}.

    2. **Set \( M = \{0, 1, 2, \cdots, q - 1\} \)**:  
   - Math: \( M \) is explicitly defined as this set.  
   - Lean: `(M : Finset ℕ := Finset.range q)`.  
   - Detailed interpretation: [[ ]]. 
   - Match: \box{Perfectly match}.

    3. **Set \( A \) definition**:  
   - Math: \( A = \{x \vert x = \sum_
    {i = 1} ^ n
    x_i
    q ^ {i - 1}, x_i \ in M\} \).
   - Lean: `A : Set ℕ := {x | ∃ (x_vec : ℕ → ℕ), (∀ i, x_vec i ∈ M) ∧ x = ∑ i in Finset.range
    n, x_vec(i + 1) * q ^ i}`.
   - Detailed interpretation: [[In Lean, `x_vec` is indexed from `1` to `n` (since `i + 1` ranges from `1` to `n`), but the math defines \( x_i \) for \( i = 1, 2, \cdots, n \). This is actually consistent, but the Lean representation is slightly more general (allowing `x_vec` to be a function on all naturals, but only using `x_vec (i + 1)` for `i` in `Finset.range n`). The Lean definition is technically correct but slightly more abstract than the math. However, it captures the same idea]].
   - Match: \box{Minor inconsistency}.

    4. **\( s, t \in A \) with specific expansions**:
   - Math: \( s = \sum_{i = 1}^n a_i q^{i - 1} \), \( t = \sum_{i = 1}^n b_i q^{i - 1} \), with \( a_i, b_i \in M \).
   - Lean: `s = ∑ i in Finset.range n, a (i + 1) * q ^ i`, `t = ∑ i in Finset.range n, b (i + 1) * q ^ i`, with `∀ i, a i ∈ M` and `∀ i, b i ∈ M`.
   - Detailed interpretation: [[The Lean version uses `a (i + 1)` and `b (i + 1)` to match the indexing in the sum, which is equivalent to the math but slightly indirect. The math directly uses \( a_i \) for \( i = 1, \dots, n \), while Lean uses `a i` for all `i` but only evaluates at `i + 1`. The Lean version is correct but not a literal translation]].
   - Match: \box{Minor inconsistency}.

    5. **\( a_n < b_n \)**:  
   - Math: \( a_n < b_n \).
   - Lean: `(hab : a n < b n)`.
   - Detailed interpretation: [[]].
   - Match: \box{Perfectly match}.

    6. **Conclusion \( s < t \)**:
   - Math: \( s < t \).
   - Lean: `s <= t`.
   - Detailed interpretation: [[The Lean version has less than or equal to, but the math has strict lower comparition]].
   - Match: \box{Major inconsistency}.

    ### Check for missing conditions / implicit conditions:
   - No missing conditions / implicit conditions
   - Detailed interpretation: [[ ]].
   - Match: \box{Perfectly match}.

   Note that for each item, if there is no issue, write empty for Detailed interpretation: <<<>>>
    """

    prompt = rf"""
    Here is a natural language math proof, its breakdown conditions * conclusions and a Lean 4 lemma.  
    Compare the conditions/conclusions with the headers/propositions of the Lean 4 lemma, matching them one by one to decide whether the Lean 4 lemma is an appropriate formalization of the mathematical statement.  
    Assign exactly one of three tags: **Perfectly match**, **Minor inconsistency**, or **Major inconsistency**.  
    Then audit for missing or implicit conditions.  
    Judge with *extremely strict* standards—any minor inconsistency is already a mismatch.

    **Special attention (four critical cases):**

    1. **Triangle angle–side correspondence** – If the problem explicitly mentions “opposite angles/sides”, the correspondence must be clearly stated and correct.  
    2. **Sequence problems** – Whenever \(a_n\), \(b_n\) (or similar) appear, verify that the index \(n\) is explicitly defined to start from 1 (not 0) and that this convention is checked at every occurrence of \(a_n\), \(b_n\).  
    3. **Extremum problems** – If \(f'(x_0) = 0\), you must also confirm that the sign of \(f'(x)\) changes on either side of \(x_0\); merely having the derivative equal zero is insufficient.  
    4. Restricted-domain functions – For expressions involving functions with restricted domains, check that the statement explicitly supplies the correct domain constraints and that every step respects those constraints:
        Logarithmic functions:

    log x, ln x, lg x → require x > 0
    log_b x → require x > 0, b > 0, b ≠ 1

    Radical functions:

    √x, sqrt x → require x ≥ 0
    x^(1/n) for even n → require x ≥ 0
    x^(1/n) for odd n → defined for all real x

    Rational functions:

    a/b, a·b^(-1) → require b ≠ 0
    x^(-n) → require x ≠ 0

    Trigonometric functions:

    sin x, cos x → defined for all real x
    tan x → require x ≠ π/2 + kπ
    cot x → require x ≠ kπ
    sec x → require cos x ≠ 0
    csc x → require sin x ≠ 0

    Inverse trigonometric functions:

    arcsin x, arccos x → require -1 ≤ x ≤ 1
    arctan x, arccot x → defined for all real x
    arcsec x, arccsc x → require |x| ≥ 1

    Exponential functions:

    e^x, exp x → defined for all real x
    a^x → require a > 0, a ≠ 1
    x^(p/q) (rational exponents) → require x > 0
    x^r (irrational exponents) → require x > 0

    Composite expressions:

    log(f(x)) → require f(x) > 0
    √(f(x)) → require f(x) ≥ 0
    g(x)/h(x) → require h(x) ≠ 0
    arcsin(f(x)) → require -1 ≤ f(x) ≤ 1

    If any discrepancy involves any of these four cases, label it as Major inconsistency.

    If any discrepancy involves **any of these four cases**, label it as **Major inconsistency**.\
    
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

    Example Output Format: {one_shot}

    -----------------

    Question: {informal_prefix_en}

    Mathematical conditions and conclusions: {math_cond}

    Lean 4 formal statement: {formal_statement}

    Output:
    """

    return prompt


def create_second_prompt_json(
    informal_prefix_en, math_cond: str, formal_statement: str
) -> str:
    """Create the second prompt for assessing appropriateness of formal statement"""

    theorem_index = formal_statement.find("theorem ")
    if theorem_index != -1:
        # Return the part starting with "theorem "
        formal_statement = formal_statement[theorem_index:]
    else:
        # If "theorem " is not found, return the original string
        formal_statement = formal_statement

    # Fix escape sequences by using raw string (r prefix) or double backslashes
    one_shot = r"""Given conditions and conclusions of natural language statement and a lemma in Lean,
     compare each condition to a corresponding Lean header and each conclusion to a Lean proposition, one by one:

    Natural language:
    \( q \) is a natural number greater than 
    \( M \) is explicitly defined as this set: *Set \( M = \{0, 1, 2, \cdots, q - 1\} \)
    Define set A as: \( A = \{x \vert x = \sum_{i = 1} ^ n x_i q ^ {i - 1}, x_i \ in M\} \).
    Let t and s be defined as: \( s = \sum_{i = 1}^n a_i q^{i - 1} \), \( t = \sum_{i = 1}^n b_i q^{i - 1} \), with \( a_i, b_i \in M \)
    We have: a_n < b_n
    We have: s < t
 
   
   Lean:
    (hq : 1 < q)
    (M : Finset ℕ := Finset.range q)
    A : Set ℕ := {x | ∃ (x_vec : ℕ → ℕ), (∀ i, x_vec i ∈ M) ∧ x = ∑ i in Finset.range n, x_vec(i + 1) * q ^ i}
    s = ∑ i in Finset.range n, a (i + 1) * q ^ i`, `t = ∑ i in Finset.range n, b (i + 1) * q ^ i`, with `∀ i, a i ∈ M` and `∀ i, b i ∈ M
    (hab : a n < b n)
    s <= t

   
   Output must be a json object:
   ```json
   {{
                    "evaluation": [
                        "Perfectly match",
                        "Perfectly match",
                        "Minor inconsistency",
                        "Minor inconsistency",
                        "Perfectly match",
                        "Major inconsistency"
                    ],
                    "feedback": [
                        "Lean formalization matches with the text",
                        "Lean formalization matches with the text",
                        "In Lean, `x_vec` is indexed from `1` to `n` (since `i + 1` ranges from `1` to `n`), but the math defines \( x_i \) for \( i = 1, 2, \cdots, n \). This is actually consistent, but the Lean representation is slightly more general (allowing `x_vec` to be a function on all naturals, but only using `x_vec (i + 1)` for `i` in `Finset.range n`). The Lean definition is technically correct but slightly more abstract than the math. However, it captures the same idea",
                        "The Lean version uses `a (i + 1)` and `b (i + 1)` to match the indexing in the sum, which is equivalent to the math but slightly indirect. The math directly uses \( a_i \) for \( i = 1, \dots, n \), while Lean uses `a i` for all `i` but only evaluates at `i + 1`. The Lean version is correct but not a literal translation",
                        "Lean formalization matches with the text",
                        "The Lean version has less than or equal to, but the math has strict lower comparition"
                    ...
                    ]
    }}

   Note that for each item, if there is no issue, write empty for Detailed interpretation: <<<>>>
    """

    prompt = rf"""
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
                ```\
                ** Important: evaluation must be among Perfectly match/Major inconsistency/Minor inconsistency
    
    """

    return prompt


def create_prem_def_prompt(formal_prefix_en: str, informal_prefix_en: str) -> str:
    prompt = rf"""\
                Please analyze the following pair of a natural language premise/definition and its Lean formalization.
                ### Natural Language:
                {informal_prefix_en}
                ### Lean Formalization:
                {formal_prefix_en}
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
                ```\
                ** Important: evaluation must be among Perfectly match/Major inconsistency/Minor inconsistency\
                
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
                """

    return prompt


# helper: convert LLM-as-a-judge judgements into A/B/C rating
def extract_match_feedback_content(text):
    # Define conversion rules
    match_mapping = {
        "Perfectly match": "A",
        "Minor inconsistency": "B",
        "Major inconsistency": "C",
    }

    # Use regex to extract match content inside \box{}, and convert to tier A, B, C
    pattern = (
        r"(?:\\box{)?(Perfectly match|Minor inconsistency|Major inconsistency)(?:})?"
    )
    matches = re.findall(pattern, text)

    converted = []
    for match in matches:
        # Remove possible whitespace and match keywords
        cleaned = match.strip()
        for key, value in match_mapping.items():
            if key.lower() in cleaned.lower():  # Case-insensitive matching
                converted.append(value)
                break
        else:  # If no keywords match
            converted.append(None)  # Or set a default value

    # Use regex to extract feedback content inside <<< >>>
    feedbacks = re.findall(r"<<<(.*?)>>>", text, re.S)

    return converted, feedbacks


# helper: match feedback content: convert LLM-as-a-judge judgements into A/B/C rating
def match_evaluations(evaluations):
    match_mapping = {
        "Perfectly match": "A",
        "Minor inconsistency": "B",
        "Major inconsistency": "C",
    }

    match_mapping.setdefault("Not found", None)

    converted = []
    for evaluation in evaluations:
        converted.append(match_mapping[evaluation])

    return converted



def generate_mu(evaluations):


    """
    Dynamically generate fuzzy measure mu(A) with rules:
    1. If A contains any C, mu(A)=0.
    2. If A contains all subtasks and all are A, mu(A)=1.0.
    3. If A contains 2+ Bs, mu(A) = base_weight * (1 - 0.2 * B_count).
    4. Otherwise: mu(A) = sum(individual subtask weights) / total subtasks.
    """
    n = len(evaluations)
    if n > 10:
        # calculate the proportion for A
        a_count = float(evaluations.count("A")) / n

        a_count = int(math.floor(a_count * 10))  # round down to the nearest integer
        b_count = 10 - a_count

        n = 10  # set n to 10
        # new an evaluations array with 10 elements
        evaluations = []
        for i in range(a_count):
            evaluations.append("A")
        for i in range(b_count):
            evaluations.append("B")


    mu = {}
    all_tasks = frozenset(range(n))
    base_weight = 1.0 / n

    C_tag = False
    if any(evaluations[i] == "C" for i in range(n)):
        C_tag = True

    for k in range(1, n + 1):
        for subset in combinations(range(n), k):
            A = frozenset(subset)
            # Rule 1: If A contains any C, mu(A)=0
            if C_tag:
                mu[A] = 0.0
            else:
                # Rule 2: All-A subset weight=1.0
                if A == all_tasks and all(evaluations[i] == "A" for i in A):
                    mu[A] = 1.0
                else:
                    b_count = sum(1 for i in A if evaluations[i] == "B")
                    # Rule 3: Penalize weight for 2+ Bs
                    if b_count >= 2:
                        mu[A] = max(base_weight * len(A) * (1 - 0.2 * b_count), 0)
                    else:
                        # Rule 4: Default basic weight sum
                        mu[A] = base_weight * len(A) * (1 - 0.1 * b_count)
    return mu, evaluations


def sugeno_integral(evaluations):

    """
    Strictly ensure score is 0 when C exists:
    1. If any subtask is C, return 0 directly.
    2. Otherwise calculate Sugeno integral normally.
    """
    if not evaluations:
        return 0.0

    mu, evaluations = generate_mu(evaluations)

    grade_map = {"A": 1.0, "B": 0.5, "C": 0}
    f = [grade_map[e] for e in evaluations if e in grade_map]

    if not f:  # If no valid evaluations after filtering
        return 0.0

    n = len(f)

    sorted_indices = sorted(range(n), key=lambda i: f[i])
    sugeno = 0.0
    for i in range(n):
        A = frozenset(sorted_indices[i:])
        mu_A = mu.get(A, 0.0)
        sugeno = max(sugeno, min(f[sorted_indices[i]], mu_A))

    return round(sugeno, 2)


def compute_total_score(proof_items, p_graph, aggregation="katz") -> float:
    # compute centrality to assign weights to nodes
    # equal weights
    if aggregation == "equal":
        raw_weights = {node_id: 1 for node_id in p_graph.nodes}

    # laplacian centrality
    elif aggregation == "laplacian":
        raw_weights = nx.laplacian_centrality(p_graph)

    # default: katz centrality (page rank)
    elif aggregation == "katz":
        raw_weights = nx.katz_centrality(p_graph)

    else:
        raise ValueError(
            "Aggregation type not known. Consider only 'equal', 'laplacian' and 'katz'. "
        )

    score = 0
    total_weight = 0
    for item in proof_items:
        if item.formalization["lean_pass"]:
            score = score + raw_weights[item.id] * item.score["semantic_score"]
        total_weight = total_weight + raw_weights[item.id]

    if total_weight == 0:
        aggregated_score = 0
    else:
        aggregated_score = score / total_weight

    return aggregated_score
