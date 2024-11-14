import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizer, AutoTokenizer

def intersection(l1, l2):
    """Returns the intersection of two lists."""
    s2 = set(l2)
    return [x for x in l1 if x in s2]

def difference(l1, l2):
    """Returns the difference between two lists."""
    s2 = set(l2)
    return [x for x in l1 if x not in s2]

# Using our existing method to create and update tokenizers
def analyze_token_sharing(name, our_tokenizer, final_tokenizer, bert_tokenizer, scibert_tokenizer):
    """Analyzes token sharing and visualizes the results."""
    results_df = {
        'ranges': [],
        'shared_with_bert_only': [],
        'shared_with_scibert_only_added': [],
        'shared_with_scibert_only_not_added': [],
        'shared_with_both': [],
        'remained_added': [],
        'remained_not_added': []
    }
    print(len(our_tokenizer.get_vocab().keys()))
    # Calculate the number of ranges (groups of 1,000 tokens)
    ranges = len(our_tokenizer.get_vocab().keys()) // 1000

    # Convert vocabularies to lists
    set_scibert = list(scibert_tokenizer.get_vocab().keys())
    set_bert = list(bert_tokenizer.get_vocab().keys())
    set_final = list(final_tokenizer.get_vocab().keys())

    print(f"Total tokens in final tokenizer: {len(set_final)}")
    print(ranges)
    # Analyze token sharing across ranges
    for i in range(ranges):
        selected_ours = list(our_tokenizer.get_vocab().keys())[i * 1000:(i + 1) * 1000]
        range_ = f'{i * 1000}-{(i + 1) * 1000}'
        results_df['ranges'].append(range_)

        # Calculate the different token categories
        results_df['shared_with_bert_only'].append(
            len(intersection(difference(selected_ours, set_scibert), set_bert))
        )
        results_df['shared_with_scibert_only_added'].append(
            len(intersection(intersection(difference(selected_ours, set_bert), set_scibert), set_final))
        )
        results_df['shared_with_scibert_only_not_added'].append(
            len(difference(intersection(difference(selected_ours, set_bert), set_scibert), set_final))
        )
        results_df['shared_with_both'].append(
            len(intersection(intersection(selected_ours, set_bert), set_scibert))
        )
        results_df['remained_added'].append(
            len(intersection(difference(difference(selected_ours, set_bert), set_scibert), set_final))
        )
        results_df['remained_not_added'].append(
            len(difference(difference(difference(selected_ours, set_bert), set_scibert), set_final))
        )

    # Create a DataFrame for visualization
    df = pd.DataFrame(results_df)

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(df['ranges'], df['shared_with_both'], label='Shared with Both', color='purple')
    ax.bar(df['ranges'], df['shared_with_bert_only'], bottom=df['shared_with_both'], label='Shared with BERT Only', color='orange')
    ax.bar(df['ranges'], df['shared_with_scibert_only_added'], 
           bottom=df['shared_with_both'] + df['shared_with_bert_only'], label='Shared with SciBERT Only and Added', color='blue')
    ax.bar(df['ranges'], df['shared_with_scibert_only_not_added'], 
           bottom=df['shared_with_both'] + df['shared_with_bert_only'] + df['shared_with_scibert_only_added'], 
           label='Shared with SciBERT Only and NOT Added', color='cyan')
    ax.bar(df['ranges'], df['remained_added'], 
           bottom=df['shared_with_both'] + df['shared_with_bert_only'] + df['shared_with_scibert_only_added'] + df['shared_with_scibert_only_not_added'], 
           label='Remained and Added', color='red')
    ax.bar(df['ranges'], df['remained_not_added'], 
           bottom=df['shared_with_both'] + df['shared_with_bert_only'] + df['shared_with_scibert_only_added'] + df['shared_with_scibert_only_not_added'] + df['remained_added'], 
           label='Remained and NOT Added', color='lightcoral')

    # Customize the plot
    ax.set_xlabel('Ranges')
    ax.set_ylabel('Counts')
    ax.set_title('Comparison of Token Sharing Across Different Ranges')
    ax.legend()
    plt.xticks(rotation=45)
    plt.show()
    plt.savefig(f'analysis_on_{name}.png')

# Example usage
if __name__ == "__main__":
    name = 'our_final_model'
    
    # Load the tokenizers using our existing methods
    our_tokenizer = BertTokenizer(vocab_file='trained_bert/vocab.txt')
    final_tokenizer = AutoTokenizer.from_pretrained(f'modified_bert')
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

    # Perform the analysis and visualization
    analyze_token_sharing(name, our_tokenizer, final_tokenizer, bert_tokenizer, scibert_tokenizer)
