import os
import os.path as osp
import re
import json
from google import genai
from openai import OpenAI
from google.genai import types

class GeminiTextEvaluator:
    def __init__(self, api_key: str):
        # Initialize the client (ensure you have google-genai installed)
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.5-flash"
        self.pre_question_prompt = "You are provided with multiple segment of dense 3D scene captions of a continuous video. Note that there might be multiple same class objects in the 3D scene, leverage the camera motion to understand the exact layout and answer the given question. You must provide your answer based on your reasoning and best guess."

    def evaluate(self, caption, question, options):
        """
        Generates a reasoning-based answer from a video caption and MCQ.
        """
        # Construct the final prompt
        full_prompt = f"""
        {self.pre_question_prompt}

        ### Video Captions
        {caption}

        ### QUESTION
        {question}

        ### OPTIONS
        {options}

        ### FINAL INSTRUCTION
        You MUST provide the final answer using the exact format: <answer>LETTER</answer>.
        Example: <answer>A</answer>
        """

        # Call the API with thinking enabled
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
                temperature=0.1
            )
        )

        return response.text

    @staticmethod
    def fetch_answer(response_text: str):
        """
        Robustly extracts the answer from either <answer> tags 
        or LaTeX \boxed{} notation as a fallback.
        """
        
        if not response_text:
            return None

        # 1. Try to find <answer>X</answer>
        xml_match = re.search(r'<answer>\s*([A-D])\s*</answer>', response_text, re.IGNORECASE)
        if xml_match:
            return xml_match.group(1).upper()
        
        # 2. Fallback: Try to find \boxed{X}
        latex_match = re.search(r'\\boxed\{([A-D])\}', response_text, re.IGNORECASE)
        if latex_match:
            return latex_match.group(1).upper()
            
        return None

    @staticmethod
    def format_clip_captions(captions: list, segment_length: int):
        """
        Formats a list of clip-level captions into a numbered segment string.
        
        Args:
            captions (list): List of strings containing captions for each clip.
            segment_length (int): The number of frames per segment.
        
        Returns:
            str: A formatted string block for the prompt.
        """
        formatted_parts = []
        for i, caption in enumerate(captions):
            start_frame = i * segment_length
            end_frame = (i + 1) * segment_length - 1
            
            segment_text = (
                f"segment {i} frame {start_frame} to frame {end_frame}\n"
                f"{caption}"
            )
            formatted_parts.append(segment_text)
        
        return "\n\n".join(formatted_parts)


class GPT5MiniTextEvaluator:
    def __init__(self, api_key: str):
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        self.model_id = "gpt-5.1"
        self.pre_question_prompt = (
            "You are provided with multiple segment of dense 3D scene captions of a "
            "continuous video. Note that there might be multiple same class objects "
            "in the 3D scene, leverage the camera motion to understand the exact layout "
            "and answer the given question. You must provide your answer based on your "
            "reasoning and best guess."
        )

    def evaluate(self, caption, question, options):
        """
        Generates a reasoning-based answer from a video caption and MCQ.
        """
        full_prompt = f"""
            {self.pre_question_prompt}

            ### Video Captions
            {caption}

            ### QUESTION
            {question}

            ### OPTIONS
            {options}

            ### FINAL INSTRUCTION
            You MUST provide the final answer using the exact format: <answer>LETTER</answer>.
            Example: <think>put your reasoning in here</think> <answer>A</answer>
            """

        response = self.client.responses.create(
            model=self.model_id,
            input=full_prompt,
            reasoning={
                "effort": "low"
            },
            max_output_tokens=1024,

        )

        return response.output_text

    @staticmethod
    def fetch_answer(response_text: str):
        """
        Robustly extracts the answer from either <answer> tags
        or LaTeX \\boxed{} notation as a fallback.
        """
        if not response_text:
            return None

        # 1. Try <answer>X</answer>
        xml_match = re.search(
            r'<answer>\s*([A-D])\s*</answer>', response_text, re.IGNORECASE
        )
        if xml_match:
            return xml_match.group(1).upper()

        # 2. Fallback: \boxed{X}
        latex_match = re.search(
            r'\\boxed\{([A-D])\}', response_text, re.IGNORECASE
        )
        if latex_match:
            return latex_match.group(1).upper()

        return None

    @staticmethod
    def format_clip_captions(captions: list, segment_length: int):
        """
        Formats a list of clip-level captions into a numbered segment string.
        """
        formatted_parts = []
        for i, caption in enumerate(captions):
            start_frame = i * segment_length
            end_frame = (i + 1) * segment_length - 1
            segment_text = (
                f"segment {i} frame {start_frame} to frame {end_frame}\n"
                f"{caption}"
            )
            formatted_parts.append(segment_text)

        return "\n\n".join(formatted_parts)


# --- Example Usage ---
if __name__ == "__main__":
    API_KEY = os.environ.get("gemini_api_key")
    evaluator = GeminiTextEvaluator(API_KEY)

    with open('./results/exp1_SG_16.json') as f:
        captions = json.load(f)

    test_cap = captions['09c1414f1b.mp4']
    input_cap = evaluator.format_clip_captions(test_cap, segment_length=16)

    mc_question = "What will be the first-time appearance order of the following categories in the video: blanket, trash can, microwave, plant?"
    mc_options = ["A. microwave, blanket, plant, trash can","B. plant, blanket, microwave, trash can","C. plant, blanket, trash can, microwave","D. blanket, trash can, microwave, plant"]

    # Step 1: Generate Response
    raw_output = evaluator.evaluate(input_cap, mc_question, mc_options)
    print("--- RAW OUTPUT ---\n", raw_output)

    # Step 2: Fetch Answer
    final_answer = evaluator.fetch_answer(raw_output)
    print(f"\nEXTRACTED ANSWER: {final_answer}")