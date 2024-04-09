import json
import random
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# Load intents from the JSON file
with open('voicebot.json', 'r') as file:
    intents = json.load(file)

# Helper function to perform synonym replacement
def synonym_replacement(words, n=1):
    new_words = words.copy()
    for _ in range(n):
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            new_words[idx] = synonym
    return new_words

# Function to augment patterns in the intents
def augment_intents(intents_data, num_samples=1):
    augmented_intents = []
    
    for intent in intents_data['intents']:
        new_patterns = []
        for pattern in intent['patterns']:
            tokenized_words = word_tokenize(pattern)
            augmented_patterns = [pattern]  # Include original pattern
            for _ in range(num_samples):
                new_words = synonym_replacement(tokenized_words)
                augmented_pattern = ' '.join(new_words)
                augmented_patterns.append(augmented_pattern)
            new_patterns.extend(augmented_patterns)
        
        # Create a new intent entry with augmented patterns
        augmented_intent = {
            'tag': intent['tag'],
            'patterns': new_patterns,
            'responses': intent['responses']
        }
        augmented_intents.append(augmented_intent)
    
    return {'intents': augmented_intents}

# Augment intents data (example: generating 2 additional variations per pattern)
augmented_data = augment_intents(intents, num_samples=2)

# Save augmented intents to a new JSON file
with open('augmented_intents.json', 'w') as outfile:
    json.dump(augmented_data, outfile, indent=4)

print("Data augmentation completed. Augmented intents saved to 'augmented_intents.json'.")
