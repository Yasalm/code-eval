import streamlit as st
import json
import os
import glob
from datetime import datetime
from pathlib import Path
import time
import threading
from evals import run_evaluation, ModelEvaluator

def get_all_evaluations():
    """Find all evaluations in the results folder, organized by model."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        return {}

    evaluations = {}
    for root, dirs, files in os.walk(results_dir):
        for dir_name in dirs:
            if len(dir_name) == 15 and dir_name.replace('_', '').isdigit():
                results_file = os.path.join(root, dir_name, "results.json")
                if os.path.exists(results_file):
                    path_parts = root.split(os.sep)
                    model_type = path_parts[1] if len(path_parts) > 1 else "unknown"
                    model_name = path_parts[2] if len(path_parts) > 2 else "unknown"
                    temperature = None
                    num_samples = None
                    
                    for part in path_parts:
                        if part.startswith("temp_"):
                            temperature = float(part.split("_")[1])
                        elif part.startswith("samples_"):
                            num_samples = int(part.split("_")[1])
                    model_key = f"{model_type}_{model_name}"
                    if model_key not in evaluations:
                        evaluations[model_key] = {
                            "model_type": model_type,
                            "model_name": model_name,
                            "runs": []
                        }
                    
                    evaluations[model_key]["runs"].append({
                        "path": os.path.join(root, dir_name),
                        "timestamp": dir_name,
                        "temperature": temperature,
                        "num_samples": num_samples
                    })
    
    for model_key in evaluations:
        evaluations[model_key]["runs"].sort(key=lambda x: x["timestamp"], reverse=True)
    
    return evaluations

def load_results(results_path):
    """Load results.json, samples.jsonl, and samples.jsonl_results.jsonl from the results directory."""
    results_file = os.path.join(results_path, "results.json")
    samples_file = os.path.join(results_path, "samples.jsonl")
    results_file_detailed = os.path.join(results_path, "samples.jsonl_results.jsonl")
    
    results = None
    samples = []
    detailed_results = []
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    
    if os.path.exists(samples_file):
        with open(samples_file, 'r') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    
    if os.path.exists(results_file_detailed):
        with open(results_file_detailed, 'r') as f:
            for line in f:
                if line.strip():
                    detailed_results.append(json.loads(line))
    
    return results, samples, detailed_results

def format_timestamp(timestamp_str):
    """Format timestamp string to readable format."""
    try:
        dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str

def display_evaluation_results(results_path, config, run_info):
    """Display results for a single evaluation run."""
    results, samples, detailed_results = load_results(results_path)
    
    if results is None:
        st.error("Could not load results.json from this evaluation.")
        return
    
    st.subheader("🔧 Configuration")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", config["model_type"].title())
    
    with col2:
        st.metric("Model Name", config["model_name"])
    
    with col3:
        st.metric("Temperature", run_info["temperature"])
    
    with col4:
        st.metric("Samples per Problem", run_info["num_samples"])
    
    st.info(f"📅 **Evaluation Time:** {format_timestamp(run_info['timestamp'])}")
    
    st.subheader("📈 Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pass_at_1 = results.get('pass@1', 0)
        st.metric(
            "Pass@1", 
            f"{pass_at_1:.3f}",
            help="Percentage of problems solved correctly on the first attempt"
        )
        st.progress(pass_at_1)
        st.caption(f"Pass@1 Score: {pass_at_1:.1%}")
    
    with col2:
        pass_at_10 = results.get('pass@10', 0)
        st.metric(
            "Pass@10", 
            f"{pass_at_10:.3f}",
            help="Percentage of problems solved correctly within 10 attempts"
        )
        st.progress(pass_at_10)
        st.caption(f"Pass@10 Score: {pass_at_10:.1%}")
    
    st.subheader("📋 Problem Samples and Completions")
    
    if detailed_results:
        st.write(f"Detailed Results ({len(detailed_results)} problems)")
        
        passed_count = sum(1 for result in detailed_results if result.get('passed', False))
        failed_count = len(detailed_results) - passed_count
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("✅ Passed", passed_count)
        with col2:
            st.metric("❌ Failed", failed_count)
        
        for i, result in enumerate(detailed_results):
            task_id = result.get('task_id', f'Problem {i}')
            passed = result.get('passed', False)
            completion = result.get('completion', '')
            error_result = result.get('result', '')
            
            status_icon = "✅" if passed else "❌"
            with st.expander(f"{status_icon} {task_id} - {'PASSED' if passed else 'FAILED'}"):
                st.subheader("Generated Completion:")
                st.code(completion, language='python')
                
                if not passed and error_result:
                    st.subheader("Error Details:")
                    st.error(error_result)
    
    elif samples:
        st.write(f"Generated Samples ({len(samples)} problems)")
        st.warning("No detailed results found. Showing only completions.")
        
        for i, sample in enumerate(samples):
            task_id = sample.get('task_id', f'Problem {i}')
            completion = sample.get('completion', '')
            
            with st.expander(f"📝 {task_id}"):
                st.code(completion, language='python')
    
    else:
        st.warning("No sample data found.")

def main():
    st.set_page_config(
        page_title="HumanEval Results Viewer",
        page_icon="📊",
        layout="wide"
    )
    
    with st.sidebar:
        st.header("🚀 Run New Evaluation")
        
        model_type = st.selectbox(
            "Model Type",
            ["openai", "mistral", "deepseek"],
            help="Select the type of model to use"
        )
        
        if model_type == "openai":
            st.info("🔑 **Setup Required:** Add `OPENAI_API_KEY` to your `.env` file")
        elif model_type == "mistral":
            st.info("🐳 **Setup Required:**\n- Docker-compose must be running\n- Add `HUGGINGFACE_TOKEN` to `.env` file")
        elif model_type == "deepseek":
            st.info("🐳 **Setup Required:**\n- Docker-compose must be running\n- Add `HUGGINGFACE_TOKEN` to `.env` file")
        
        if model_type == "openai":
            model_name = st.text_input(
                "Model Name",
                value="gpt-4o-mini-2024-07-18",
                help="OpenAI model name"
            )
        elif model_type == "mistral":
            model_name = st.text_input(
                "Model Name", 
                value="/model/mistral-7b-instruct-v0.2.Q2_K.gguf",
                help="Mistral model name"
            )
        else:
            model_name = st.text_input(
                "Model Name", 
                value="/model/deepseek-coder-1.3b-instruct.Q4_K_M.gguf",
                help="Deepdeek model name"
            )
        col1, col2 = st.columns(2)
        with col1:
            num_samples = st.number_input(
                "Samples per Problem",
                min_value=1,
                max_value=10,
                value=1,
                help="Number of completions per problem"
            )
        with col2:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Sampling temperature"
            )
        
        
        if st.button("🚀 Run Evaluation", type="primary"):
            st.info("⏱️ This may take up to an hour to complete.")
            with st.spinner("Running evaluation..."):
                results = run_evaluation(
                    model_type=model_type,
                    model_name=model_name,
                    num_samples=num_samples,
                    temperature=temperature
                )
                
                if results:
                    st.success("✅ Evaluation completed successfully!")
                    st.json(results)
                    st.rerun() 
                else:
                    st.error("❌ Evaluation failed!")
        
        st.markdown("---")
        st.markdown("### 📊 Current Results")
        st.markdown("View your evaluation results in the main panel.")
    
    st.title("📊 HumanEval Evaluation Results")
    st.markdown("---")
    
    evaluations = get_all_evaluations()
    
    if not evaluations:
        st.error("No evaluation results found in the results folder.")
        st.info("Run an evaluation first using: `python evals.py`")
        return
    
    st.header("📊 Overview")
    total_models = len(evaluations)
    total_runs = sum(len(model_data["runs"]) for model_data in evaluations.values())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Models", total_models)
    with col2:
        st.metric("Total Runs", total_runs)
    
    st.markdown("---")
    st.header("🔍 Model Evaluations")
    
    for model_key, model_data in evaluations.items():
        model_type = model_data["model_type"]
        model_name = model_data["model_name"]
        runs = model_data["runs"]
        
        latest_run = runs[0]
        latest_results, _, _ = load_results(latest_run["path"])
        latest_pass_at_1 = latest_results.get('pass@1', 0) if latest_results else 0
        latest_pass_at_10 = latest_results.get('pass@10', 0) if latest_results else 0
        
        with st.expander(f"🤖 {model_type.title()}: {model_name} (Pass@1: {latest_pass_at_1:.3f}, Pass@10: {latest_pass_at_10:.3f}) - {len(runs)} runs"):
            st.write(f"**Model Type:** {model_type.title()}")
            st.write(f"**Model Name:** {model_name}")
            st.write(f"**Total Runs:** {len(runs)}")
            
            for i, run in enumerate(runs):
                run_results, _, _ = load_results(run["path"])
                run_pass_at_1 = run_results.get('pass@1', 0) if run_results else 0
                run_pass_at_10 = run_results.get('pass@10', 0) if run_results else 0
                with st.expander(f"Run {i+1}: {format_timestamp(run['timestamp'])} - Pass@1: {run_pass_at_1:.3f}, Pass@10: {run_pass_at_10:.3f} (temp={run['temperature']}, samples={run['num_samples']})"):
                    display_evaluation_results(run["path"], model_data, run)

if __name__ == "__main__":
    main()
