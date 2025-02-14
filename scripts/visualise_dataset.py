import streamlit as st
import truesight # noqa: F401

from openai_finetuner.dataset import DatasetManager

# Initialize dataset manager
manager = DatasetManager()

def display_message_card(sample, index):
    """Display a single sample as a card."""
    # Use theme-friendly semi-transparent colors
    user_bg_color = "rgba(64, 128, 255, 0.1)"  # Slight blue tint
    assistant_bg_color = "rgba(128, 128, 128, 0.2)"  # Slightly darker gray
    
    with st.container():
        st.markdown('<div class="message-card">', unsafe_allow_html=True)
        st.markdown(f"#### Sample {index}")
        
        for msg in sample["messages"]:
            # Different background colors for user/assistant messages
            bg_color = user_bg_color if msg["role"] == "user" else assistant_bg_color
                
            st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>{msg['role'].title()}</strong><br>
                    {msg['content']}
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def filter_samples(dataset, search_term):
    """Filter dataset samples based on search term."""
    return [
        sample for sample in dataset 
        if not search_term or any(search_term.lower() in msg["content"].lower() 
                                for msg in sample["messages"])
    ]

def main():
    st.title("Dataset Browser")
    
    # Move controls to sidebar
    with st.sidebar:
        st.header("Controls")
        
        # Dataset selector
        datasets = manager.list_datasets()
        selected_dataset = st.selectbox("Select dataset:", datasets)
        
        if selected_dataset:
            # Load selected dataset
            dataset = manager.retrieve_dataset(selected_dataset)
            st.write(f"Total samples: {len(dataset)}")
            
            # Search box
            search_term = st.text_input("Search messages:", "")
            
            # Only show pagination controls if there are samples
            if len(dataset) > 0:
                # Pagination controls
                st.subheader("Pagination")
                samples_per_page = st.selectbox("Samples per page:", [5, 10, 20, 50], index=0)
    
    # Main content area
    if selected_dataset:
        filtered_samples = filter_samples(dataset, search_term)
        
        if not filtered_samples:
            st.info("No samples found matching your criteria.")
            return
            
        # Pagination
        samples_per_page = samples_per_page if len(dataset) > 0 else 1  # Default to 1 if dataset is empty
        max_pages = max((len(filtered_samples) - 1) // samples_per_page + 1, 1)
        
        # Show pagination info and controls in main area
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.write(f"Showing {len(filtered_samples)} samples")
            page = st.number_input("Page", min_value=1, max_value=max_pages, value=1)
        
        # Display samples for current page
        start_idx = (page - 1) * samples_per_page
        end_idx = min(start_idx + samples_per_page, len(filtered_samples))
        
        # Display each sample as a card
        for i, sample in enumerate(filtered_samples[start_idx:end_idx], start=start_idx + 1):
            display_message_card(sample, i)

if __name__ == "__main__":
    main()
