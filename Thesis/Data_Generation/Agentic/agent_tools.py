from langchain_core.tools import tool
import random, re, pandas as pd

from data_utils import  get_aspect

class AgentTools:
    def __init__(self, training_terms, training_polarities, train_df, llm):
        self.training_terms = training_terms
        self.training_polarities = training_polarities
        self.train_df = train_df
        self.llm = llm

    # @tool
    def get_info(self) -> str:
        """Extracts the information from sample sentences."""
        print("_"*100)
        print('in get info tool')


        samples = self.train_df.sample(n=3)['raw_text'].tolist()
        print('Samples:', samples)
        prompt = f"""Analyze the following sentences and identify:
        1. The dominant writing style 
        2. The dominant grammar structure 
        3. The dominant length of the sentences (short = less than 10 words, medium = between 10 15 words, long = more than 15 words)

        Output format (JSON):  
            {{"writing_style": "...", "grammar_structure": "...", "length": "..."}}  

        Use ***one*** JSON that describes all of the sentences. Do not analyze each sentence individually.
        Make sure your output **exactly** follows this format. Do not include explanations.
        Sentences:\n""" + "\n".join(f'- "{s}"' for s in samples)

        response = self.llm.invoke(prompt)
        print('Response:', response.content)

        print('Info extracted')
        print('_'*100)

        return response
    
    # @tool
    def generate_sentences(self, style_info: dict) -> str:
        """
        Generates a sentence using aspect terms and a given writing style.
        Arguments:
        style_info: dictionary containing keys:
                    - "writing_style"
                    - "grammar_structure"
                    - "length"
        """
        print("_"*100)
        print('Generating sentence')

        terms = get_aspect(self.training_terms, self.training_polarities)
        aspect_term = terms[0]
        polarity = terms[1]

        writing_style = style_info.get("writing_style", "unknown")
        grammar_structure = style_info.get("grammar_structure", "unknown")
        sentence_length = style_info.get("length", "unknown")

        print('Selected terms:', aspect_term)
        print('Selected polarities:', polarity)

        prompt = f"""
        You are a critic who can generate comments on the specified aspect and sentiment
        We would like you to complete a sentence generation task. Please follow these requirements:

        ###TASK###
        - Generate a sentence using this aspect term: {', '.join(aspect_term)} with the following polarities : {', '.join(polarity)}.
        Write in the style: {writing_style}, and use a {grammar_structure} grammatical structure and {sentence_length} sentence length.
        
        ###REQUIREMENTS###
        - Your response must include:
        1. The sentence.
        2. A line that starts with `Terms=` followed by the list of aspect terms used.
        3. A line that starts with `Polarity=` followed by the matching polarity list.
        - Domain: Restaurants
        - the sentence should not have aspect words that are not specified in the prompt
        - Use the exact structure shown in the examples below.

        Good Examples : 
        ### input ###
        ['prices'] ['negative']
        ### Output ###
        The prices were too high for this type of restaurant
        Terms: ['prices'] 
        Polarity: ['negative']
        ### input ###
        ['ambience', 'food'] ['positive', 'neurtal']
        ### Output ###
        However, go for the ambience, and consider the food just a companion for a trip across the world!
        Terms= ['ambience', 'food']
        Polarity= ['positive', 'neutral']
        ### input ###
        ['food', 'portions', "Ray's Boathouse"]
        ### Output ###
        sentence= The food was lousy - too sweet or too salty and the portions tiny, but Ray's Boathouse had a great view.
        Terms= ['food', 'portions', "Ray's Boathouse"]
        Polarity=  ['negative', 'negative','positive']

        Bad Example : 
        ### input ###
        ['soup'],['positive']
        ### Output ###
        sentence= The udon soup was rich and flavorful.
        Terms= ['soup']
        Polarity= ['positive']
        (correct term was soup)


        Make sure your output **EXACTLY** follows this format. Do not include explanations.
        Use plain apostrophes (') in words like "Ray's" or "chef's". Do **not** escape them with backslashes.
        """
        response = self.llm.invoke(prompt)
        print('Sentence generated')
        return response
    
    # @tool
    def evaluate_sentence(self,text: str) -> str:
        """
        Evaluates if the provided *aspect terms* and their corresponding *polarities* are *correctly* used in the sentence
        Responds only with 'OK' or 'NOT_OK'.
        """
        print("_"*100)
        print('Evaluating sentence')
        prompt = f"""
        You are an expert in linguistic evaluation. Your task is to check if the given aspect terms and polarities are correct for the provided sentence.
        
        - If ALL aspect terms appear as actual aspects in the sentence with intended polarities, respond **only** with: OK
        - If any term is missing, incorrect, or not an aspect of the sentence,or wrong polarities respond **only** with: NOT_OK

        Do not provide explanations or any other text.

        Example Input:
        The food was lousy, too sweet or too salty and the portions tiny. Terms= ['food', 'portions'], Polarity= ['negative', 'negative']
        OK
        The Gnocchi was perfectly cooked and delicious, but the cheesecake was dry and flavorless. Terms=['Gnocchi', 'cheesecake'], Polarity= ['positive', 'negative']
        OK

        Bad Example:
        The udon soup was rich and flavorful. Terms= ['soup'], Polarity= ['positive']
        NOT_OK
        Despite the open kitchen adding to the atmosphere, the New England Chowder lacked flavor and freshness. Terms= ['open kitchen', 'New England Chowder', 'atmosphere'], Polarity= ['negative', 'negative', 'positive']
        NOT_OK

        Make sure your output **exactly** follows this format. Do not include explanations.
        Input:
        {text}
        """
        return self.llm.invoke(prompt)
    
    # @tool
    def label_inclusion(self, text: str) -> str:
        """
        Checks if all aspect terms are present in the sentence.
        Input format: <sentence> Terms=[...] Polarity=[...]
        Returns 'OK' if all terms are present in the sentence, otherwise 'NOT_OK'.
        """
        print("_"*100)
        print('in label inclusion tool')

        try:
            # Extract sentence, terms, and polarity using regex
            sentence_match = re.match(r'^(.*?)\s+Terms=', text)
            terms_match = re.search(r'Terms=\[(.*?)\]', text)

            if not sentence_match or not terms_match:
                return 'NOT_OK'

            sentence = sentence_match.group(1).strip()
            terms_str = terms_match.group(1)
            terms = [term.strip().strip("'\"") for term in terms_str.split(',') if term.strip()]
            print('terms:',terms)
            print('sentence:',sentence)
            # Check if every term appears in the sentence
            for term in terms:
                if term not in sentence:
                    return 'NOT_OK'
            return 'OK'

        except Exception as e:
            print(f"[label_inclusion] Error: {e}")
            return 'NOT_OK'
        

    def generator_tools(self):
        return [
            tool(self.get_info),
            tool(self.generate_sentences),
        ]

    def evaluator_tools(self):
        return [
            tool(self.label_inclusion),
            tool(self.evaluate_sentence),
        ]
