import streamlit as st
import json
import pathlib

# Get current directory and load prompts
curr_dir = pathlib.Path(__file__).parent
with open(curr_dir / "prompts.json", 'r') as f:
    prompts = json.load(f)

def main():
    st.title("WildChat Prompts Browser")
    
    # Display total number of prompts
    st.write(f"Total number of prompts: {len(prompts)}")
    
    # Add a search box
    search_term = st.text_input("Search prompts:", "")
    
    # Filter prompts based on search term
    if search_term:
        filtered_prompts = [p for p in prompts if search_term.lower() in p.lower()]
    else:
        filtered_prompts = prompts
    
    st.write(f"Showing {len(filtered_prompts)} prompts")
    
    # Add pagination
    prompts_per_page = st.selectbox("Prompts per page:", [10, 20, 50, 100], index=0)
    page = st.number_input("Page", min_value=1, max_value=(len(filtered_prompts) - 1) // prompts_per_page + 1, value=1)
    
    # Display prompts for current page
    start_idx = (page - 1) * prompts_per_page
    end_idx = min(start_idx + prompts_per_page, len(filtered_prompts))
    
    # Create an expander for each prompt
    for i, prompt in enumerate(filtered_prompts[start_idx:end_idx], start=start_idx + 1):
        with st.expander(f"Prompt {i}: {prompt[:100]}..."):
            st.text_area("Full prompt:", prompt, height=200, key=f"prompt_{i}")
            st.write(f"Length: {len(prompt)} characters")

if __name__ == "__main__":
    main()
