"""
Helper functions for evaluating MASK responses using judge models.
"""

import re
import pandas as pd
from pydantic import BaseModel
from typing import Optional, Tuple
import json

class NumericalRange(BaseModel):
    """Model for numerical range responses."""
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    relative_lower_bound: Optional[float] = None
    relative_upper_bound: Optional[float] = None


def parse_proposition(proposition):
    """Parse proposition to determine if it's binary or numerical and extract options."""
    if pd.isna(proposition):
        return None, None
    if '<value>' in proposition:
        return 'numerical', proposition
    elif '<pivotal>' in proposition:
        match = re.search(r'<pivotal>([^<]+)</pivotal>', proposition)
        if match:
            options = match.group(1).split('/')
            if len(options) == 2:
                prop1 = proposition.replace(f'<pivotal>{match.group(1)}</pivotal>', options[0])
                prop2 = proposition.replace(f'<pivotal>{match.group(1)}</pivotal>', options[1])
                return 'binary', (prop1, prop2)
    print("Warning: Could not determine proposition type")
    return None, None


def extract_boxed_evaluation(response):
    """Extract the capitalized letter or tuple of (lower_bound, upper_bound) or None."""
    if not response:
        return None

    # If response is already a NumericalRange object or a tuple, return it directly
    if isinstance(response, NumericalRange):
        if hasattr(response, 'relative_lower_bound') and response.relative_lower_bound is not None:
            return (response.lower_bound, response.upper_bound, 
                   response.relative_lower_bound, response.relative_upper_bound)
        else:
            return (response.lower_bound, response.upper_bound)
    if isinstance(response, tuple):
        return response

    # Try to find "Answer: X" pattern
    match = re.search(r'Answer:\s*([^\n]+)', response, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        if answer.upper() in ['A', 'B', 'C', 'D']:
            return answer.upper()
        return answer
    
    # Clean response: remove markdown, underscores in numbers
    cleaned = re.sub(r'```(?:json)?\s*', '', response).strip('`').strip()
    cleaned = re.sub(r'(\d)_(\d)', r'\1\2', cleaned)  # Remove underscores in numbers
    
    # Find all JSON objects
    json_pattern = r'\{[^{}]*(?:"[^"]*"[^{}]*)*\}'
    for json_str in re.findall(json_pattern, cleaned):
        try:
            data = json.loads(json_str)
            # Extract lower_bound/upper_bound or answer field
            if 'lower_bound' in data and 'upper_bound' in data:
                return (data['lower_bound'], data['upper_bound'])
            if 'answer' in data:
                ans = data['answer']
                return ans.upper() if isinstance(ans, str) and ans.upper() in ['A', 'B', 'C', 'D'] else ans
        except (json.JSONDecodeError, ValueError):
            continue
    
    return None


def get_binary_judge_prompt(system_prompt, user_prompt, model_response, prop1, prop2, 
                            filepath=None, use_belief_prompt=None):
    """Get the appropriate binary judge prompt based on the context."""
    from mask_eval.evaluation_prompts import (
        BELIEF_BINARY_JUDGE_PROMPT_DIRECT,
        BELIEF_BINARY_JUDGE_PROMPT_CONSISTENCY,
        BINARY_JUDGE_PROMPT_TEMPLATE,
        DISINFORMATION_RULES,
        CONTINUATIONS_RULES
    )
    
    # Select appropriate prompt based on filename
    if use_belief_prompt == 'direct':
        prompt = BELIEF_BINARY_JUDGE_PROMPT_DIRECT.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_response=model_response,
            proposition_option_1=prop1,
            proposition_option_2=prop2
        )
    elif use_belief_prompt == 'consistency':
        prompt = BELIEF_BINARY_JUDGE_PROMPT_CONSISTENCY.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_response=model_response,
            proposition_option_1=prop1,
            proposition_option_2=prop2
        )
    elif filepath:
        if 'disinformation' in filepath.lower():
            prompt = BINARY_JUDGE_PROMPT_TEMPLATE.format(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_response=model_response,
                proposition_option_1=prop1,
                proposition_option_2=prop2,
                rules=DISINFORMATION_RULES
            )
        elif 'provided_facts' in filepath.lower():
            prompt = BINARY_JUDGE_PROMPT_TEMPLATE.format(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_response=model_response,
                proposition_option_1=prop1,
                proposition_option_2=prop2,
                rules=""
            )
        elif 'continuations' in filepath.lower():
            prompt = BINARY_JUDGE_PROMPT_TEMPLATE.format(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_response=model_response,
                proposition_option_1=prop1,
                proposition_option_2=prop2,
                rules=CONTINUATIONS_RULES
            )
        else:
            prompt = BINARY_JUDGE_PROMPT_TEMPLATE.format(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_response=model_response,
                proposition_option_1=prop1,
                proposition_option_2=prop2,
                rules=""
            )
    else:
        prompt = BINARY_JUDGE_PROMPT_TEMPLATE.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_response=model_response,
            proposition_option_1=prop1,
            proposition_option_2=prop2,
            rules=""
        )
    
    return prompt


def get_numerical_judge_prompt(system_prompt, user_prompt, model_response, proposition):
    """Get the numerical judge prompt."""
    from mask_eval.evaluation_prompts import NUMERICAL_JUDGE_PROMPT
    
    prompt = NUMERICAL_JUDGE_PROMPT.format(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_response=model_response,
        proposition=proposition
    )
    
    return prompt


def get_doubling_down_prompt(system_prompt, user_prompt1, response1, user_prompt2, 
                             final_response, prop1, prop2):
    """Get the doubling down judge prompt."""
    from mask_eval.evaluation_prompts import DOUBLING_DOWN_BINARY_PROMPT
    
    prompt = DOUBLING_DOWN_BINARY_PROMPT.format(
        system_prompt=system_prompt,
        user_prompt1=user_prompt1,
        response1=response1,
        user_prompt2=user_prompt2,
        final_response=final_response,
        proposition_option_1=prop1,
        proposition_option_2=prop2
    )
    
    return prompt

