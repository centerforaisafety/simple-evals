"""Utility functions for Machiavelli evaluation."""

import json
import re


def extract_json_with_keys(content: str) -> dict:
    """
    Extracts the final JSON content from a string.
    
    Args:
        content: String that may contain JSON (possibly in markdown code blocks)
    
    Returns:
        Parsed JSON dict if found and valid, None otherwise
    
    Example:
        >>> json_data = extract_json_with_keys(response_text)
    """
    
    # First try to parse the entire content as JSON
    try:
        parsed_json = json.loads(content)
        return parsed_json
    except json.JSONDecodeError:
        pass
    
    json_content = None
    
    # Try to extract JSON from markdown code blocks
    if '```json' in content:
        json_blocks = re.findall(r'```json\n(.*?)\n```', content, re.DOTALL)
        if json_blocks:
            json_content = json_blocks[-1]
    
    # If not found, try to find the final JSON-like structure (last {} block)
    if not json_content:
        # Find all potential JSON objects (balanced braces)
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, content, re.DOTALL)
        if json_matches:
            json_content = json_matches[-1]
    
    if json_content and isinstance(json_content, str):      
        try:
            parsed_json = json.loads(json_content)
            return parsed_json
        except json.JSONDecodeError:
            pass
    
    return None

