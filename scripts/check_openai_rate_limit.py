import os
import requests
import argparse
import truesight # noqa: F401
from typing import Optional
from datetime import datetime

def get_rate_limits(
    project_id: str,
    api_key: str,
    limit: Optional[int] = 100,
    after: Optional[str] = None,
    before: Optional[str] = None
) -> dict:
    """
    Fetch rate limits for a specific OpenAI project.
    
    Args:
        project_id: The OpenAI project ID
        api_key: OpenAI API key
        limit: Maximum number of results to return (default: 100)
        after: Cursor for pagination - fetch results after this object ID
        before: Cursor for pagination - fetch results before this object ID
        
    Returns:
        Dictionary containing rate limit information
    """
    url = f"https://api.openai.com/v1/organization/projects/{project_id}/rate_limits"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    params = {"limit": limit}
    if after:
        params["after"] = after
    if before:
        params["before"] = before
        
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching rate limits: {e}")
        if response.text:
            print(f"Response: {response.text}")
        return None

def format_rate_limits(rate_limits: dict) -> None:
    """
    Print formatted rate limit information.
    
    Args:
        rate_limits: Dictionary containing rate limit data
    """
    if not rate_limits:
        print("No rate limit data available")
        return
        
    print("\nOpenAI Rate Limits:")
    print("-" * 80)
    
    for limit in rate_limits.get("data", []):
        print(f"\nModel: {limit.get('model', 'Unknown')}")
        print(f"Limit Type: {limit.get('limit_type', 'Unknown')}")
        print(f"Requests: {limit.get('requests', 'Unknown')}")
        print(f"Tokens: {limit.get('tokens', 'Unknown')}")
        
        # Print time window if available
        if "window" in limit:
            print(f"Time Window: {limit['window']} seconds")
            
def main():
    parser = argparse.ArgumentParser(description="Check OpenAI project rate limits")
    parser.add_argument("--project-id", required=True, help="OpenAI project ID")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env variable)")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of results")
    parser.add_argument("--after", help="Pagination: fetch results after this object ID")
    parser.add_argument("--before", help="Pagination: fetch results before this object ID")
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key must be provided via --api-key or OPENAI_API_KEY environment variable")
        return
        
    rate_limits = get_rate_limits(
        project_id=args.project_id,
        api_key=api_key,
        limit=args.limit,
        after=args.after,
        before=args.before
    )
    
    format_rate_limits(rate_limits)

if __name__ == "__main__":
    main()
