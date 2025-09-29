import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm
from dotenv import load_dotenv
import re

import sys
sys.path.append('benchmark')
from human_eval.data import read_problems
from human_eval.evaluation import evaluate_functional_correctness

load_dotenv()

class ModelEvaluator:
    def __init__(self, model_type: str = "openai", model_name: str = "gpt-4o-mini-2024-07-18", temperature: float = 0.5):
        """
        Initialize the model evaluator.
        
        Args:
            model_type: "openai" or "mistral" or "deepseek"
            model_name: Model name/identifier
        """
        self.model_type = model_type
        self.model_name = model_name
        self.temperature = temperature
        self.client = None
        
        if model_type == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif model_type == "mistral" :
            from openai import OpenAI
            self.client = OpenAI(base_url="http://localhost:8000/v1", api_key="ollama")
        elif self.model_type == "deepseek":
            from openai import OpenAI
            self.client = OpenAI(base_url="http://localhost:8001/v1", api_key="ollama")
        else:
            raise ValueError(f"invalid model type {model_type} avail options are 'mistral', 'deepseek', 'openai'")

    def generate_completion(self, prompt: str) -> str:
        """
        Generate a completion for a given prompt using the specified model.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Generated completion text
        """
        try:
            if self.model_type == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are coding assistant tasked in generating python code for the following problem (only genererate code):"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=512,
                    temperature=self.temperature
                )
                return response.choices[0].message.content
            
            elif self.model_type == "mistral" or self.model_type == "deepseek":
                wrapped_prompt = f"""### Instructions:
                    You are coding assistant tasked in generating python code for the following problem (only genererate code):
                    {prompt}
                    ### Response:
                    """
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=wrapped_prompt,
                    max_tokens=256,
                    temperature=self.temperature
                )
                return response.choices[0].text
                
        except Exception as e:
            print(f"Error generating completion: {e}")
            return ""
    
    def run_evaluation(self, num_samples: int = 1) -> Dict:
        """
        Run evaluation on HumanEval problems.
        
        Args:
            num_samples: Number of samples per problem
        Returns:
            Dictionary containing evaluation results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_clean = self.model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
        results_dir = f"results/{self.model_type}/{model_clean}/temp_{self.temperature}/samples_{num_samples}/{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        problems = read_problems()

        problem_items = problems.items()
        
        samples = []
        print(f"Generating completions for {len(problem_items)} problems...")
        
        for task_id, problem in tqdm(problem_items, desc="Generating completions"):
            prompt = problem["prompt"]
            
            for i in range(num_samples):
                completion = self.generate_completion(prompt)
                # time.sleep(0.5)
                if '```python' in completion:
                    match = re.search(r"```python\n(.*?)```", completion, re.DOTALL)
                    if match:
                        code = match.group(1).strip()
                        completion = code
                
                samples.append({
                    "task_id": task_id,
                    "completion": completion
                })
        
        samples_file = os.path.join(results_dir, "samples.jsonl")
        with open(samples_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        
        print(f"Samples saved to: {samples_file}")
        
        print("Evaluating samples...")
        results = evaluate_functional_correctness(samples_file)
        
        results_file = os.path.join(results_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        print(f"Pass@1: {results.get('pass@1', 0):.3f}")
        print(f"Pass@10: {results.get('pass@10', 0):.3f}")
        
        return results


def run_evaluation(model_type: str = "openai", 
        model_name: str = None, 
        num_samples: int = 1, 
        temperature: float = 0.5
        ):
    """
    Run evaluation with any supported model.
    
    Args:
        model_type: "openai" or "mistral" or "deepseek"
        model_name: Model name (defaults to common models if not specified)
        num_samples: Number of samples per problem; only k=1,10 are allowed
        num_problems: Number of problems to evaluate (None for all)
    """
    if num_samples not in (1, 10):
        raise ValueError(f"only 1, 10 samples per promblems are allowed, provided {num_samples}")
    if model_name is None:
        if model_type == "openai":
            model_name = "gpt-4o-mini-2024-07-18"
        elif model_type == "mistral":
            model_name = "/model/mistral-7b-instruct-v0.2.Q2_K.gguf"
        elif model_type == "deepseek":
            model_name = "/model/deepseek-coder-1.3b-instruct.Q4_K_M.gguf"
    
    evaluator = ModelEvaluator(model_type=model_type, model_name=model_name, temperature=temperature)
    return evaluator.run_evaluation(num_samples=num_samples)


if __name__ == "__main__":
    run_evaluation(model_type="openai", model_name="gpt-4o-mini-2024-07-18", num_samples=10)
    
    # run_evaluation(model_type="mistral", model_name="/model/mistral-7b-instruct-v0.2.Q4_K_M.gguf", num_samples=1,)
    # run_evaluation(model_type="deepseek", model_name="/model/deepseek-coder-6.7b-instruct.Q4_K_M.gguf", num_samples=1,)
    
    
    # print("evals.py - HumanEval Model Evaluation")
    # print("Usage:")
    # print("  run_evaluation(model_type='openai', model_name='gpt-4o-mini-2024-07-18', num_samples=1, num_problems=5)")
    # print("  run_evaluation(model_type='mistral', model_name='/model/mistral-7b-instruct-v0.2.Q4_K_M.gguf', num_samples=1, num_problems=5)")
    # print("  run_evaluation(model_type='openai', num_samples=1, num_problems=5)  # Uses default model")