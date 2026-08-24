#!/usr/bin/env python3
"""
Run RULER benchmark evaluation on an LLM model using the official RULER repository.

RULER (What's the Real Context Size of Your Long-Context Language Models?) is a
comprehensive synthetic benchmark for evaluating long-context language models.
It includes 13 tasks across 4 categories with configurable sequence lengths.

This script integrates directly with the RULER repository from NVIDIA:
https://github.com/NVIDIA/RULER

Requirements:
1. Clone the RULER repository:
   git clone https://github.com/NVIDIA/RULER
2. Set RULER_REPO_PATH environment variable or use --ruler_repo_path argument
"""

# Handle flash attention compatibility issues with vllm
# vllm installs vllm-flash-attn which may be incompatible with the current PyTorch version
# This causes "undefined symbol" errors when transformers tries to import flash_attn
import sys
import types

# Check if flash_attn_2_cuda is broken by trying to import it
try:
    import flash_attn_2_cuda
except (ImportError, AttributeError, OSError, RuntimeError) as e:
    # Flash attention CUDA module is broken (likely due to vllm-flash-attn incompatibility)
    error_str = str(e).lower()
    if 'undefined symbol' in error_str or 'flash_attn' in error_str:
        # Create dummy flash_attn_2_cuda module to prevent import errors
        # This allows transformers to load without crashing
        dummy_cuda = types.ModuleType('flash_attn_2_cuda')
        # Add minimal attributes that might be accessed
        dummy_cuda.flash_attn_varlen_func = lambda *args, **kwargs: None
        dummy_cuda.flash_attn_func = lambda *args, **kwargs: None
        sys.modules['flash_attn_2_cuda'] = dummy_cuda

import argparse
import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch
from loguru import logger
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import with error handling for flash attention compatibility issues
try:
    from utils import load_model_and_tokenizer, add_common_args
    from palu.quant_utils import configure_latent_quantizer
except RuntimeError as e:
    error_msg = str(e)
    if 'flash_attn' in error_msg.lower() or 'undefined symbol' in error_msg.lower():
        print("\n" + "="*80)
        print("ERROR: Flash attention compatibility issue detected")
        print("="*80)
        print("\nThis error is caused by an incompatible flash-attn version installed by vllm.")
        print("The flash-attn CUDA module doesn't match your PyTorch version.\n")
        print("SOLUTIONS:")
        print("\n1. Reinstall flash-attn to match your PyTorch version:")
        print("   pip uninstall flash-attn")
        print("   pip install flash-attn --no-build-isolation")
        print("\n2. Or uninstall flash-attn if you don't need it (recommended for RULER):")
        print("   pip uninstall flash-attn")
        print("   (RULER works fine without flash attention)")
        print("\n3. Or use a separate environment for vllm to avoid conflicts")
        print("="*80 + "\n")
        raise RuntimeError(
            "Flash attention import failed due to version incompatibility. "
            "Please see the solutions above. For RULER evaluation, you can safely uninstall flash-attn."
        ) from e
    raise

# Import quantization configuration
try:
    from run_generation_with_quantization import configure_model_quantization
except ImportError:
    logger.warning("Could not import configure_model_quantization. Per-dimension quantization may not be available.")
    configure_model_quantization = None

# Find RULER repository
RULER_REPO_PATH = None
RULER_DIRECT_AVAILABLE = False

# RULER task categories mapping
RULER_TASK_CATEGORIES = {
    "retrieval": [
        "niah_single_1", "niah_single_2", "niah_single_3",
        "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
        "niah_multivalue", "niah_multiquery"
    ],
    "lookup": [
        "niah_single_1", "niah_single_2", "niah_single_3",
        "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
        "niah_multivalue", "niah_multiquery"
    ],
    "multi_hop": ["vt"],
    "tracing": ["vt"],
    "aggregation": ["cwe", "fwe"],
    "qa": ["qa_1", "qa_2"],
    "question_answering": ["qa_1", "qa_2"],
    "all": [
        "niah_single_1", "niah_single_2", "niah_single_3",
        "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
        "niah_multivalue", "niah_multiquery",
        "vt", "cwe", "fwe", "qa_1", "qa_2"
    ]
}

# Check for RULER repository in common locations
possible_ruler_paths = [
    os.path.join(os.path.dirname(__file__), "ruler"),
    os.path.join(os.path.dirname(__file__), "RULER"),
    os.path.join(os.path.expanduser("~"), "ruler"),
    os.path.join(os.path.expanduser("~"), "RULER"),
    os.environ.get("RULER_REPO_PATH", ""),
]

for ruler_path in possible_ruler_paths:
    if ruler_path and os.path.exists(ruler_path) and os.path.exists(os.path.join(ruler_path, "scripts")):
        RULER_REPO_PATH = os.path.abspath(ruler_path)
        RULER_DIRECT_AVAILABLE = True
        logger.info(f"Found RULER repository at: {RULER_REPO_PATH}")
        break

if not RULER_DIRECT_AVAILABLE:
    logger.warning("RULER repository not found. Please clone it:")
    logger.warning("  git clone https://github.com/NVIDIA/RULER")
    logger.warning("  export RULER_REPO_PATH=/path/to/RULER")


def format_prompt_for_model(
    prompt: str,
    tokenizer
) -> str:
    """
    Format a prompt for the model using chat template if available.
    
    Args:
        prompt: The prompt text
        tokenizer: The tokenizer (may have apply_chat_template)
        
    Returns:
        Formatted prompt string
    """
    # Check if tokenizer has apply_chat_template
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        try:
            # Format as a user message
            conversation = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except Exception as e:
            logger.warning(f"Error using chat template: {e}, falling back to direct prompt")
            return prompt
    else:
        # For tokenizers without chat template, return prompt as-is
        return prompt


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device: str = "cuda",
    max_length: int = 32768,
    extra_generate_kwargs: Optional[dict] = None,
    past_key_values=None,
) -> str:
    """
    Generate a response for a prompt.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        prompt: Input prompt text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        device: Device to run on
        max_length: Maximum total sequence length (input + output)
        
    Returns:
        Generated response text
    """
    model.eval()
    
    # Get the device from the model's first parameter
    device = next(model.parameters()).device
    
    # Format the prompt for the model
    formatted_prompt = format_prompt_for_model(prompt, tokenizer)
    
    # Tokenize prompt with truncation to prevent OOM
    # Leave room for generated tokens: max_input_length = max_length - max_new_tokens - buffer
    # Use a buffer of 100 tokens for safety
    buffer = 100
    max_input_length = max_length - max_new_tokens - buffer
    
    # For very large max_length, still use a reasonable default to prevent issues
    # But allow up to max_length if user explicitly sets it high
    if max_length > 65536:
        # For extremely large contexts, be more conservative with input length
        max_input_length = min(max_input_length, max_length - max_new_tokens - 500)
    
    # Ensure we don't truncate to less than a reasonable minimum
    if max_input_length < 512:
        max_input_length = 512
        logger.warning(f"max_length ({max_length}) is too small for max_new_tokens ({max_new_tokens}). Using minimum input length of 512.")
    
    # Truncate from the LEFT (beginning of context) so the question/answer-prefix
    # at the END of the RULER prompt is always preserved.  The default HuggingFace
    # truncation cuts from the right, which silently removes the question.
    orig_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(device)
    finally:
        tokenizer.truncation_side = orig_truncation_side
    input_ids = inputs.input_ids
    prompt_length = input_ids.shape[1]

    # Warn if input was truncated (only if significantly truncated)
    if prompt_length >= max_input_length - 10:  # Small threshold to avoid noise
        logger.debug(f"Input truncated to {prompt_length} tokens (max: {max_input_length}) to fit within max_length={max_length}")
    
    # Generate with error handling for OOM
    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    if extra_generate_kwargs:
        generate_kwargs.update(extra_generate_kwargs)
    if past_key_values is not None:
        generate_kwargs["past_key_values"] = past_key_values
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)
    except RuntimeError as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower() or "NVML_SUCCESS" in error_msg or "CUDACachingAllocator" in error_msg:
            # Clear cache and re-raise so caller can handle it
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(f"CUDA OOM during generation: {error_msg}") from e
        raise
    
    # Decode response (skip the prompt)
    response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    
    # Clear CUDA cache after generation to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return response.strip()


def detect_model_template_type(model_path: str, tokenizer) -> str:
    """
    Detect the model template type based on model path or tokenizer.
    
    Args:
        model_path: Path to the model
        tokenizer: Tokenizer instance
        
    Returns:
        Template type string (base, meta-chat, etc.)
    """
    model_path_lower = model_path.lower()
    
    # Check model name patterns
    if "llama-2" in model_path_lower or "llama2" in model_path_lower:
        return "meta-chat"
    elif "llama-3" in model_path_lower or "llama3" in model_path_lower:
        return "meta-llama3"
    elif "vicuna" in model_path_lower:
        return "vicuna-chat"
    elif "mistral" in model_path_lower:
        return "meta-chat"  # Mistral uses similar format
    elif "qwen" in model_path_lower:
        return "base"  # Qwen typically uses base
    elif "jamba" in model_path_lower:
        return "jamba"
    elif "phi" in model_path_lower:
        return "Phi3"
    elif "command-r" in model_path_lower or "commandr" in model_path_lower:
        return "command-r-chat"
    
    # Check tokenizer chat template
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        # If tokenizer has chat template, use meta-chat as default
        return "meta-chat"
    
    # Default to base
    return "base"


def check_ruler_dependencies() -> tuple:
    """
    Check if required RULER dependencies are installed.
    
    Returns:
        Tuple of (all_installed, missing_modules)
    """
    missing = []
    # Check common RULER dependencies based on their requirements.txt
    required_modules = ["wonderwords", "nltk", "yaml", "tenacity"]
    
    for module in required_modules:
        try:
            if module == "yaml":
                __import__("yaml")
            else:
                __import__(module)
        except ImportError:
            missing.append(module)
    
    return len(missing) == 0, missing


def generate_ruler_samples(
    ruler_repo_path: str,
    root_dir: str,
    task_name: str,
    tokenizer,
    model_path: str,
    max_seq_length: int = 32768,
    num_samples: int = 500,
    model_template_type: Optional[str] = None,
) -> bool:
    """
    Generate RULER task samples using RULER's data generation scripts.
    
    Args:
        ruler_repo_path: Path to RULER repository
        root_dir: Root directory for outputs
        task_name: Task name (e.g., "niah_single_1")
        tokenizer: Tokenizer instance (used to determine tokenizer path/type)
        model_path: Path to the model (used to detect template type)
        max_seq_length: Maximum sequence length
        num_samples: Number of samples to generate
        model_template_type: Model template type (base, meta-chat, etc.). If None, auto-detects from model_path
        
    Returns:
        True if samples were generated successfully, False otherwise
    """
    scripts_path = os.path.join(ruler_repo_path, "scripts")
    prepare_script = os.path.join(scripts_path, "data", "prepare.py")
    
    if not os.path.exists(prepare_script):
        logger.warning(f"RULER prepare.py script not found: {prepare_script}")
        return False
    
    # Check if required data files exist, and download them if missing
    json_data_dir = os.path.join(scripts_path, "data", "synthetic", "json")
    paul_graham_essay_file = os.path.join(json_data_dir, "PaulGrahamEssays.json")
    download_script = os.path.join(json_data_dir, "download_paulgraham_essay.py")
    
    # Check for QA dataset files
    squad_file = os.path.join(json_data_dir, "squad.json")
    hotpotqa_file = os.path.join(json_data_dir, "hotpotqa.json")
    download_qa_script = os.path.join(json_data_dir, "download_qa_dataset.sh")
    
    # Check for english_words.json (needed for cwe task)
    english_words_file = os.path.join(json_data_dir, "english_words.json")
    
    if not os.path.exists(paul_graham_essay_file) and os.path.exists(download_script):
        logger.info("PaulGrahamEssays.json not found. Attempting to download required data files...")
        logger.info("This is a one-time setup step for RULER.")
        
        # Check for required dependencies
        missing_download_deps = []
        try:
            import html2text
        except ImportError:
            missing_download_deps.append("html2text")
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            missing_download_deps.append("beautifulsoup4")
        
        if missing_download_deps:
            logger.warning(f"Missing dependencies for downloading data files: {', '.join(missing_download_deps)}")
            logger.warning("Please install them with:")
            logger.warning(f"  pip install {' '.join(missing_download_deps)}")
            logger.warning("")
            logger.warning("Or install all RULER dependencies:")
            logger.warning(f"  pip install -r {os.path.join(ruler_repo_path, 'docker', 'requirements.txt')}")
            logger.warning("")
            logger.warning("You can also download the file manually:")
            logger.warning(f"  cd {json_data_dir}")
            logger.warning(f"  python {download_script}")
        else:
            try:
                import subprocess
                result = subprocess.run(
                    [sys.executable, download_script],
                    cwd=json_data_dir,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout
                )
                if result.returncode == 0:
                    if os.path.exists(paul_graham_essay_file):
                        logger.info("✓ Successfully downloaded PaulGrahamEssays.json")
                    else:
                        logger.warning("Download script completed but file not found. Check output:")
                        if result.stdout:
                            logger.warning(f"stdout: {result.stdout}")
                        if result.stderr:
                            logger.warning(f"stderr: {result.stderr}")
                else:
                    logger.warning(f"Download script returned non-zero exit code: {result.returncode}")
                    if result.stderr:
                        logger.warning(f"stderr: {result.stderr}")
                    if result.stdout:
                        logger.warning(f"stdout: {result.stdout}")
                    logger.warning("You may need to run the download script manually:")
                    logger.warning(f"  cd {json_data_dir}")
                    logger.warning(f"  python {download_script}")
            except subprocess.TimeoutExpired:
                logger.error("Download script timed out. Please run it manually:")
                logger.error(f"  cd {json_data_dir}")
                logger.error(f"  python {download_script}")
            except Exception as e:
                logger.warning(f"Failed to download data files: {e}")
                logger.warning("You may need to run the download script manually:")
                logger.warning(f"  cd {json_data_dir}")
                logger.warning(f"  python {download_script}")
    
    # Check again if the file exists
    if not os.path.exists(paul_graham_essay_file):
        logger.warning(f"PaulGrahamEssays.json still not found at {paul_graham_essay_file}")
        logger.warning("Some RULER tasks (like niah_single_2, niah_single_3, niah_multikey_1) require this file.")
        logger.warning("To download it manually:")
        logger.warning(f"  cd {json_data_dir}")
        logger.warning(f"  pip install html2text beautifulsoup4  # if not already installed")
        logger.warning(f"  python {download_script}")
        logger.warning("")
        logger.warning("Tasks that don't require this file may still work.")
    
    # Check and download QA datasets if needed
    if (not os.path.exists(squad_file) or not os.path.exists(hotpotqa_file)) and os.path.exists(download_qa_script):
        missing_qa = []
        if not os.path.exists(squad_file):
            missing_qa.append("squad.json")
        if not os.path.exists(hotpotqa_file):
            missing_qa.append("hotpotqa.json")
        
        logger.info(f"Missing QA dataset files: {', '.join(missing_qa)}. Attempting to download...")
        try:
            import subprocess
            result = subprocess.run(
                ["bash", download_qa_script],
                cwd=json_data_dir,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout (these files can be large)
            )
            if result.returncode == 0:
                downloaded = []
                if os.path.exists(squad_file):
                    downloaded.append("squad.json")
                if os.path.exists(hotpotqa_file):
                    downloaded.append("hotpotqa.json")
                if downloaded:
                    logger.info(f"✓ Successfully downloaded: {', '.join(downloaded)}")
                else:
                    logger.warning("Download script completed but files not found")
                    if result.stdout:
                        logger.warning(f"stdout: {result.stdout}")
                    if result.stderr:
                        logger.warning(f"stderr: {result.stderr}")
            else:
                logger.warning(f"Download script returned non-zero exit code: {result.returncode}")
                if result.stderr:
                    logger.warning(f"stderr: {result.stderr}")
                if result.stdout:
                    logger.warning(f"stdout: {result.stdout}")
                logger.warning("You may need to download them manually:")
                logger.warning(f"  cd {json_data_dir}")
                logger.warning(f"  bash {download_qa_script}")
        except subprocess.TimeoutExpired:
            logger.error("Download script timed out. Please run it manually:")
            logger.error(f"  cd {json_data_dir}")
            logger.error(f"  bash {download_qa_script}")
        except Exception as e:
            logger.warning(f"Failed to download QA datasets: {e}")
            logger.warning("You may need to download them manually:")
            logger.warning(f"  cd {json_data_dir}")
            logger.warning(f"  bash {download_qa_script}")
    
    # Warn about missing QA files
    if not os.path.exists(squad_file):
        logger.warning(f"squad.json not found at {squad_file}")
        logger.warning("This file is needed for the 'qa_1' task. Download it with:")
        logger.warning(f"  cd {json_data_dir}")
        logger.warning(f"  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O squad.json")
    
    if not os.path.exists(hotpotqa_file):
        logger.warning(f"hotpotqa.json not found at {hotpotqa_file}")
        logger.warning("This file is needed for the 'qa_2' task. Download it with:")
        logger.warning(f"  cd {json_data_dir}")
        logger.warning(f"  wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O hotpotqa.json")
    
    # Check if english_words.json exists and is valid
    english_words_valid = False
    if os.path.exists(english_words_file):
        try:
            with open(english_words_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    logger.warning(f"english_words.json exists but is empty. Attempting to fix...")
                    os.remove(english_words_file)
                    logger.info(f"Removed empty english_words.json file")
                else:
                    # Try to parse it
                    data = json.loads(content)
                    # Verify it's a dict with values that can be converted to list
                    if isinstance(data, dict) and len(data) > 0:
                        test_list = list(data.values())
                        if all(isinstance(v, str) for v in test_list[:10]):  # Check first 10 values
                            english_words_valid = True
                            logger.debug(f"english_words.json is valid with {len(data)} entries")
                        else:
                            raise ValueError("english_words.json does not contain string values")
                    else:
                        raise ValueError("english_words.json is not a valid dictionary")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"english_words.json exists but is corrupted: {e}")
            logger.info("Attempting to fix corrupted file...")
            try:
                # Try to fix common corruption issues
                with open(english_words_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Check if it's missing opening brace
                if content.startswith('"') and not content.startswith('{'):
                    logger.warning("File appears to be missing opening brace. This file may need to be re-downloaded.")
                    logger.warning("The file should be a JSON object like: {\"1\": \"word1\", \"2\": \"word2\", ...}")
                
                # Remove corrupted file
                os.remove(english_words_file)
                logger.info(f"Removed corrupted english_words.json file")
                logger.warning("You will need to restore this file for the 'cwe' task to work.")
                logger.warning(f"Expected location: {english_words_file}")
                logger.warning("This file should contain a JSON dictionary mapping numbers to English words.")
            except Exception as fix_error:
                logger.error(f"Failed to remove corrupted file: {fix_error}")
    else:
        logger.warning(f"english_words.json not found at {english_words_file}")
    
    if not english_words_valid:
        logger.warning("This file is needed for the 'cwe' (common words extraction) task.")
        logger.warning("The 'cwe' task will fail if this file is missing or corrupted.")
        logger.warning(f"Expected format: JSON file at {english_words_file} containing a dictionary")
        logger.warning("  Example: {\"1\": \"word1\", \"2\": \"word2\", \"3\": \"word3\", ...}")
    
    # Auto-detect template type if not provided
    if model_template_type is None:
        model_template_type = detect_model_template_type(model_path, tokenizer)
        logger.info(f"Auto-detected model template type: {model_template_type}")
    
    # Check dependencies first
    deps_ok, missing_deps = check_ruler_dependencies()
    if not deps_ok:
        logger.error(f"Missing required RULER dependencies: {', '.join(missing_deps)}")
        logger.error("Please install them with:")
        logger.error(f"  pip install {' '.join(missing_deps)}")
        logger.error("Or install all RULER dependencies:")
        logger.error(f"  pip install -r {os.path.join(ruler_repo_path, 'docker', 'requirements.txt')}")
        return False
    
    logger.info(f"Generating RULER samples for task: {task_name}")
    logger.info(f"  Max sequence length: {max_seq_length}")
    logger.info(f"  Number of samples: {num_samples}")
    logger.info(f"  Model template type: {model_template_type}")
    
    # Determine tokenizer path and type
    # CRITICAL: RULER's HFTokenizer uses AutoTokenizer.from_pretrained() which will try to
    # download from HuggingFace Hub if the path doesn't look like a local directory.
    # We MUST pass an absolute path that exists locally.
    
    tokenizer_type = "hf"  # Default to HuggingFace
    
    # The tokenizer was already loaded successfully, so we know the model_path is valid
    # We need to find the actual directory where the tokenizer files are located
    
    # CRITICAL: The model_path parameter should already be absolute (we convert it before calling this function)
    # But let's make absolutely sure and verify it exists
    
    # Convert model_path to absolute if it's not already
    if os.path.isabs(model_path):
        abs_model_path = model_path
    else:
        abs_model_path = os.path.abspath(model_path)
    
    # PRIORITY: Use model_path directly since we know the model was loaded from it
    # The model_path parameter should already be absolute (converted before calling this function)
    # This is the most reliable approach
    
    # Convert model_path to absolute if it's not already
    if os.path.isabs(model_path):
        abs_model_path = model_path
    else:
        abs_model_path = os.path.abspath(model_path)
    
    # First, try the model_path directly (it should exist since we loaded the model from it)
    if os.path.exists(abs_model_path):
        tokenizer_path = abs_model_path
        logger.info(f"Using model_path as tokenizer path: {tokenizer_path}")
    else:
        # Fallback: Try to get from tokenizer's vocab_file
        tokenizer_path = None
        if hasattr(tokenizer, "vocab_file") and tokenizer.vocab_file:
            vocab_file = tokenizer.vocab_file
            if vocab_file and os.path.exists(vocab_file):
                vocab_dir = os.path.dirname(os.path.abspath(vocab_file))
                if os.path.exists(vocab_dir):
                    tokenizer_path = vocab_dir
                    logger.info(f"Found tokenizer path from vocab_file directory: {tokenizer_path}")
        
        # If still not found, try name_or_path (but make sure it's absolute)
        if tokenizer_path is None or not os.path.exists(tokenizer_path):
            if hasattr(tokenizer, "name_or_path"):
                tokenizer_name_or_path = tokenizer.name_or_path
                # Only use if it's an absolute path that exists
                if tokenizer_name_or_path and os.path.isabs(tokenizer_name_or_path) and os.path.exists(tokenizer_name_or_path):
                    tokenizer_path = tokenizer_name_or_path
                    logger.info(f"Found tokenizer path from name_or_path: {tokenizer_path}")
        
        # Final fallback: try relative to script
        if tokenizer_path is None or not os.path.exists(tokenizer_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            potential_path = os.path.join(script_dir, model_path)
            if os.path.exists(potential_path):
                tokenizer_path = potential_path
                logger.info(f"Found tokenizer path relative to script: {tokenizer_path}")
            else:
                logger.error(f"Tokenizer path does not exist: {abs_model_path}")
                logger.error(f"Original model path: {model_path}")
                logger.error(f"Tried absolute path: {abs_model_path}")
                logger.error(f"Tried relative to script: {potential_path}")
                logger.error("")
                logger.error("RULER needs a local directory path containing tokenizer files.")
                logger.error("Please ensure the model path is correct and points to a local directory.")
                return False
    
    # Make absolutely sure it's an absolute path
    tokenizer_path = os.path.abspath(tokenizer_path)
    
    # Final verification
    if not os.path.exists(tokenizer_path):
        logger.error(f"Tokenizer path does not exist after all attempts: {tokenizer_path}")
        return False
    
    # Log the final path being used
    logger.info(f"Using tokenizer path: {tokenizer_path}")
    
    # Debug: Check if tokenizer files exist
    tokenizer_files = ["tokenizer_config.json", "tokenizer.json", "vocab.json"]
    found_files = [f for f in tokenizer_files if os.path.exists(os.path.join(tokenizer_path, f))]
    if found_files:
        logger.debug(f"Found tokenizer files: {found_files}")
    else:
        logger.warning(f"Warning: No standard tokenizer files found in {tokenizer_path}")
        logger.warning("RULER may still work if the tokenizer can be loaded from the model directory.")
    
    # Create data directory (must be absolute — prepare.py runs with cwd=scripts_path)
    data_dir = os.path.abspath(os.path.join(root_dir, "samples"))
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Call RULER's prepare.py script
        import subprocess
        
        # CRITICAL: Ensure tokenizer_path is absolute and exists before passing to RULER
        # RULER's tokenizer code will try to download from HuggingFace if it doesn't recognize it as a local path
        if not os.path.isabs(tokenizer_path):
            tokenizer_path = os.path.abspath(tokenizer_path)
            logger.warning(f"Converted tokenizer path to absolute: {tokenizer_path}")
        
        if not os.path.exists(tokenizer_path):
            logger.error(f"Tokenizer path does not exist: {tokenizer_path}")
            return False
        
        cmd = [
            sys.executable,
            prepare_script,
            "--save_dir", data_dir,
            "--benchmark", "synthetic",
            "--task", task_name,
            "--tokenizer_path", str(tokenizer_path),  # This MUST be an absolute path
            "--tokenizer_type", tokenizer_type,
            "--max_seq_length", str(max_seq_length),
            "--model_template_type", model_template_type,
            "--num_samples", str(num_samples),
            "--random_seed", "42",
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        logger.info(f"Tokenizer path (absolute): {tokenizer_path}")
        
        result = subprocess.run(
            cmd,
            cwd=scripts_path,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        # Always log output for debugging
        # Log at info level if there's an error, otherwise debug
        log_level = logger.error if result.returncode != 0 else logger.debug
        if result.stdout:
            log_level(f"prepare.py stdout:\n{result.stdout}")
        if result.stderr:
            log_level(f"prepare.py stderr:\n{result.stderr}")
        
        # Check for missing dependencies in stderr or stdout
        error_output = result.stderr + result.stdout
        if "ModuleNotFoundError" in error_output or "ImportError" in error_output:
            missing_modules = []
            if "wonderwords" in error_output:
                missing_modules.append("wonderwords")
            if "tenacity" in error_output:
                missing_modules.append("tenacity")
            if "nltk" in error_output:
                missing_modules.append("nltk")
            if "yaml" in error_output and "yaml" not in [m.lower() for m in missing_modules]:
                missing_modules.append("pyyaml")
            
            if missing_modules:
                logger.error(f"Missing required RULER dependencies: {', '.join(missing_modules)}")
                logger.error(f"Please install them with:")
                logger.error(f"  pip install {' '.join(missing_modules)}")
                logger.error(f"Or install all RULER dependencies:")
                logger.error(f"  pip install -r {os.path.join(ruler_repo_path, 'docker', 'requirements.txt')}")
                return False
        
        # Check if prepare.py failed
        if result.returncode != 0:
            logger.error(f"Sample generation failed for {task_name} (exit code: {result.returncode})")
            logger.error("=" * 80)
            logger.error("ERROR OUTPUT FROM prepare.py:")
            logger.error("=" * 80)
            if result.stdout:
                logger.error(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                logger.error(f"STDERR:\n{result.stderr}")
            logger.error("=" * 80)
            return False
        
        # Check for errors in output even if return code is 0
        # RULER's prepare.py may return 0 even if the subprocess fails
        error_output = result.stderr + result.stdout
        has_traceback = "Traceback" in error_output
        has_error = "Error" in error_output or "Exception" in error_output
        
        if has_traceback or (has_error and "ModuleNotFoundError" not in error_output):
            logger.error(f"Sample generation failed for {task_name} (error detected in output)")
            logger.error("=" * 80)
            logger.error("ERROR OUTPUT FROM prepare.py:")
            logger.error("=" * 80)
            if result.stdout:
                logger.error(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                logger.error(f"STDERR:\n{result.stderr}")
            logger.error("=" * 80)
            return False
        
        if result.returncode == 0:
            # RULER saves files as {task_name}/validation.jsonl in the save_dir
            # Check both possible locations
            output_file_v1 = os.path.join(data_dir, f"{task_name}", "validation.jsonl")
            output_file_v2 = os.path.join(data_dir, f"{task_name}.jsonl")
            
            output_file = None
            if os.path.exists(output_file_v1):
                output_file = output_file_v1
            elif os.path.exists(output_file_v2):
                output_file = output_file_v2
            
            if output_file:
                logger.info(f"✓ Successfully generated samples: {output_file}")
                # Convert jsonl to json format that our script expects
                return convert_ruler_jsonl_to_json(output_file, task_name)
            else:
                logger.warning(f"prepare.py completed but output file not found in expected locations:")
                logger.warning(f"  Expected: {output_file_v1} or {output_file_v2}")
                # Log output at warning level so user can see what happened
                if result.stdout:
                    logger.warning(f"stdout:\n{result.stdout}")
                if result.stderr:
                    logger.warning(f"stderr:\n{result.stderr}")
                # Check if there are any jsonl files in the directory
                if os.path.exists(data_dir):
                    jsonl_files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
                    if jsonl_files:
                        logger.info(f"Found jsonl files in directory: {jsonl_files}")
                return False
        else:
            logger.error(f"Failed to generate samples for {task_name}")
            logger.error(f"Return code: {result.returncode}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Sample generation timed out for {task_name}")
        return False
    except Exception as e:
        logger.error(f"Error generating samples for {task_name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def convert_ruler_jsonl_to_json(jsonl_file: str, task_name: str) -> bool:
    """
    Convert RULER's jsonl format to JSON format expected by our script.
    
    RULER saves samples as jsonl with fields: index, input, outputs
    We convert to JSON array with: id, prompt, answer
    
    Args:
        jsonl_file: Path to input jsonl file
        task_name: Task name for output file
        
    Returns:
        True if conversion successful
    """
    try:
        samples = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    # Convert RULER format to our format
                    converted_sample = {
                        "id": sample.get("index", len(samples)),
                        "prompt": sample.get("input", ""),
                        "answer": sample.get("outputs", [""])[0] if isinstance(sample.get("outputs"), list) else sample.get("outputs", ""),
                        "input": sample.get("input", ""),  # Keep original field
                        "outputs": sample.get("outputs", []),  # Keep original field
                    }
                    samples.append(converted_sample)
        
        # Save as JSON in the samples directory (parent of task directory)
        # If jsonl_file is in {task_name}/validation.jsonl, save to samples/{task_name}_samples.json
        jsonl_dir = os.path.dirname(jsonl_file)
        if os.path.basename(jsonl_dir) == task_name:
            # jsonl_file is in {task_name}/validation.jsonl, save to parent/samples
            samples_dir = os.path.dirname(jsonl_dir)
        else:
            # jsonl_file is directly in samples dir
            samples_dir = jsonl_dir
        
        json_file = os.path.join(samples_dir, f"{task_name}_samples.json")
        with open(json_file, 'w') as f:
            json.dump(samples, f, indent=2)
        
        logger.info(f"Converted {len(samples)} samples from jsonl to json: {json_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting jsonl to json: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def load_ruler_modules(ruler_repo_path: str):
    """
    Load RULER Python modules from the repository.
    
    Args:
        ruler_repo_path: Path to RULER repository
        
    Returns:
        Dictionary with loaded modules
    """
    scripts_path = os.path.join(ruler_repo_path, "scripts")
    if not os.path.exists(scripts_path):
        raise FileNotFoundError(f"RULER scripts directory not found at {scripts_path}")
    
    # Add RULER scripts to path
    sys.path.insert(0, scripts_path)
    
    modules = {}
    try:
        # Try to import RULER's data and evaluation modules
        # These imports depend on RULER's structure
        import importlib.util
        
        # Common RULER module paths
        module_paths = {
            'data': os.path.join(scripts_path, 'data'),
            'eval': os.path.join(scripts_path, 'eval'),
        }
        
        for module_name, module_path in module_paths.items():
            if os.path.exists(module_path):
                # Try to import as a package
                try:
                    spec = importlib.util.spec_from_file_location(
                        f"ruler_{module_name}",
                        os.path.join(module_path, "__init__.py")
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        modules[module_name] = module
                except Exception as e:
                    logger.debug(f"Could not load {module_name} module: {e}")
        
        return modules
    except Exception as e:
        logger.warning(f"Could not load RULER modules: {e}")
        logger.warning("Will use RULER's evaluation scripts directly")
        return {}


def filter_ruler_tasks(
    tasks: Optional[List[str]] = None,
    task_category: Optional[str] = None,
    task_filter: Optional[str] = None,
) -> List[str]:
    """
    Filter RULER tasks based on category or filter pattern.
    
    Args:
        tasks: Explicit list of task names (takes precedence)
        task_category: Category name (e.g., "retrieval", "lookup", "multi_hop", "aggregation", "qa")
        task_filter: Pattern to filter task names (e.g., "niah_*" for all niah tasks)
        
    Returns:
        Filtered list of task names
    """
    # If explicit tasks provided, use those
    if tasks:
        return tasks
    
    # Start with all tasks
    available_tasks = RULER_TASK_CATEGORIES["all"]
    
    # Filter by category if specified
    if task_category:
        category_lower = task_category.lower()
        if category_lower in RULER_TASK_CATEGORIES:
            available_tasks = RULER_TASK_CATEGORIES[category_lower]
            logger.info(f"Filtering tasks by category '{task_category}': {available_tasks}")
        else:
            logger.warning(f"Unknown task category '{task_category}'. Available categories: {list(RULER_TASK_CATEGORIES.keys())}")
            logger.info("Using all tasks instead.")
    
    # Filter by pattern if specified
    if task_filter:
        import fnmatch
        filtered_tasks = [task for task in available_tasks if fnmatch.fnmatch(task, task_filter)]
        if filtered_tasks:
            available_tasks = filtered_tasks
            logger.info(f"Filtering tasks by pattern '{task_filter}': {available_tasks}")
        else:
            logger.warning(f"No tasks match pattern '{task_filter}'")
    
    return available_tasks


def run_ruler_evaluation(
    model,
    tokenizer,
    model_path: str,
    ruler_repo_path: str,
    root_dir: str,
    tasks: Optional[List[str]] = None,
    task_category: Optional[str] = None,
    task_filter: Optional[str] = None,
    max_length: int = 32768,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    limit: Optional[int] = None,
    num_samples: int = 500,
    tokenizer_path: Optional[str] = None,
    extra_generate_kwargs: Optional[dict] = None,
    generate_max_length: Optional[int] = None,
    pre_generate_hook: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Run RULER evaluation using the RULER repository's evaluation framework.

    generate_max_length: if set, passed to generate_response as max_length instead
    of max_length.  Use this when you want RULER to generate samples of exactly
    max_length tokens but don't want the generate_response buffer (max_new_tokens +
    100) to silently truncate the context.  Typically set to
    max_length + max_new_tokens + 100 so the model sees the full context.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        model_path: Path to the model (used for template detection)
        ruler_repo_path: Path to RULER repository
        root_dir: Root directory for storing RULER outputs
        tasks: List of task names (if None, uses default 'synthetic')
        task_category: Filter tasks by category
        task_filter: Filter tasks by pattern
        max_length: Maximum sequence length
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        limit: Limit number of examples per task
        num_samples: Number of samples to generate per task if samples don't exist
        
    Returns:
        Dictionary with evaluation results
    """
    if not RULER_DIRECT_AVAILABLE or not ruler_repo_path:
        raise RuntimeError(
            "RULER repository not found. Please clone it:\n"
            "  git clone https://github.com/NVIDIA/RULER\n"
            "  export RULER_REPO_PATH=/path/to/RULER"
        )
    
    logger.info("Running RULER evaluation using RULER repository...")
    logger.info(f"RULER repo path: {ruler_repo_path}")
    logger.info(f"Root directory: {root_dir}")
    
    # Create root directory structure
    os.makedirs(root_dir, exist_ok=True)
    samples_dir = os.path.join(root_dir, "samples")
    predictions_dir = os.path.join(root_dir, "predictions")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    
    scripts_path = os.path.join(ruler_repo_path, "scripts")
    sys.path.insert(0, scripts_path)
    
    # Check RULER structure
    data_path = os.path.join(scripts_path, "data")
    eval_path = os.path.join(scripts_path, "eval")
    inference_path = os.path.join(scripts_path, "inference")
    
    if not os.path.exists(data_path) or not os.path.exists(eval_path):
        raise FileNotFoundError(
            f"RULER scripts structure not found. Expected:\n"
            f"  {data_path}\n"
            f"  {eval_path}"
        )
    
    logger.info("RULER repository structure found. Starting evaluation...")
    
    # Check dependencies upfront
    deps_ok, missing_deps = check_ruler_dependencies()
    if not deps_ok:
        logger.error("=" * 80)
        logger.error("MISSING RULER DEPENDENCIES")
        logger.error("=" * 80)
        logger.error(f"Missing required dependencies: {', '.join(missing_deps)}")
        logger.error("")
        logger.error("Please install them with:")
        logger.error(f"  pip install {' '.join(missing_deps)}")
        logger.error("")
        logger.error("Or install all RULER dependencies:")
        logger.error(f"  pip install -r {os.path.join(ruler_repo_path, 'docker', 'requirements.txt')}")
        logger.error("")
        logger.error("Common RULER dependencies include:")
        logger.error("  - wonderwords (for generating synthetic words)")
        logger.error("  - tenacity (for retry logic)")
        logger.error("  - nltk (for natural language processing)")
        logger.error("  - pyyaml (for YAML config parsing)")
        logger.error("=" * 80)
        raise RuntimeError(f"Missing RULER dependencies: {', '.join(missing_deps)}. Please install them first.")
    
    # Filter tasks based on category or filter pattern
    task_list = filter_ruler_tasks(
        tasks=tasks,
        task_category=task_category,
        task_filter=task_filter,
    )
    
    if not task_list:
        raise ValueError("No tasks selected for evaluation. Please specify tasks, task_category, or task_filter.")
    
    logger.info(f"Evaluating {len(task_list)} task(s): {task_list}")
    
    results = {}
    all_scores = []
    
    try:
        # Import RULER modules
        import importlib.util
        
        # Try to import RULER's data and evaluation modules
        synthetic_data_path = os.path.join(data_path, "synthetic")
        synthetic_eval_path = os.path.join(eval_path, "synthetic")
        
        # Step 1: Generate or load task samples
        logger.info("Step 1: Loading/generating RULER task samples...")
        task_samples = {}
        
        for task_name in task_list:
            # Check if samples already exist (check multiple possible locations)
            # 1. Our converted JSON format: {task_name}_samples.json
            sample_file_json = os.path.join(samples_dir, f"{task_name}_samples.json")
            # 2. RULER's jsonl format: {task_name}/validation.jsonl
            sample_file_jsonl = os.path.join(samples_dir, task_name, "validation.jsonl")
            # 3. RULER's jsonl format (alternative): {task_name}.jsonl
            sample_file_jsonl_alt = os.path.join(samples_dir, f"{task_name}.jsonl")
            
            sample_file = None
            if os.path.exists(sample_file_json):
                sample_file = sample_file_json
            elif os.path.exists(sample_file_jsonl):
                # Convert jsonl to json
                logger.info(f"Found jsonl file at {sample_file_jsonl}, converting to JSON format...")
                if convert_ruler_jsonl_to_json(sample_file_jsonl, task_name):
                    sample_file = sample_file_json
            elif os.path.exists(sample_file_jsonl_alt):
                # Convert jsonl to json
                logger.info(f"Found jsonl file at {sample_file_jsonl_alt}, converting to JSON format...")
                if convert_ruler_jsonl_to_json(sample_file_jsonl_alt, task_name):
                    sample_file = sample_file_json
            
            if sample_file and os.path.exists(sample_file):
                logger.info(f"Loading existing samples for {task_name} from {sample_file}...")
                with open(sample_file, 'r') as f:
                    task_samples[task_name] = json.load(f)
                logger.info(f"Loaded {len(task_samples[task_name])} samples for {task_name}")
            else:
                # Try to generate samples using RULER's data generation scripts
                logger.info(f"Samples not found for {task_name}. Attempting to generate...")

                # Resolve the path RULER will use for tokenizer files.
                # Priority: explicit --tokenizer_path > local model_path > tokenizer.name_or_path
                if tokenizer_path and os.path.exists(tokenizer_path):
                    abs_model_path = os.path.abspath(tokenizer_path)
                    logger.info(f"Using explicit tokenizer_path for RULER sample generation: {abs_model_path}")
                else:
                    if not os.path.isabs(model_path):
                        abs_model_path = os.path.abspath(model_path)
                    else:
                        abs_model_path = model_path
                    if not os.path.exists(abs_model_path):
                        logger.warning(f"Model path does not exist locally: {abs_model_path}")
                        if hasattr(tokenizer, "name_or_path") and os.path.exists(tokenizer.name_or_path):
                            abs_model_path = os.path.abspath(tokenizer.name_or_path)
                            logger.info(f"Using tokenizer name_or_path: {abs_model_path}")

                generated = generate_ruler_samples(
                    ruler_repo_path=ruler_repo_path,
                    root_dir=root_dir,
                    task_name=task_name,
                    tokenizer=tokenizer,
                    model_path=abs_model_path,
                    max_seq_length=max_length,
                    num_samples=num_samples,
                    model_template_type=None,  # Auto-detect
                )
                
                if generated:
                    # Try to load the newly generated samples (check all possible locations)
                    if os.path.exists(sample_file_json):
                        with open(sample_file_json, 'r') as f:
                            task_samples[task_name] = json.load(f)
                        logger.info(f"Loaded {len(task_samples[task_name])} newly generated samples for {task_name}")
                    elif os.path.exists(sample_file_jsonl):
                        # Convert and load
                        if convert_ruler_jsonl_to_json(sample_file_jsonl, task_name):
                            with open(sample_file_json, 'r') as f:
                                task_samples[task_name] = json.load(f)
                            logger.info(f"Loaded {len(task_samples[task_name])} newly generated samples for {task_name}")
                        else:
                            logger.warning(f"Could not convert generated jsonl file: {sample_file_jsonl}")
                            continue
                    else:
                        logger.warning(f"Sample generation attempted but file not found in expected locations:")
                        logger.warning(f"  Expected: {sample_file_json} or {sample_file_jsonl}")
                        logger.warning(f"Please check if RULER's prepare.py completed successfully.")
                        continue
                else:
                    logger.warning(f"Could not generate samples for {task_name} automatically.")
                    logger.warning(f"Please generate samples first using RULER's scripts, or place them in:")
                    logger.warning(f"  {sample_file_json}")
                    logger.warning(f"  OR {sample_file_jsonl}")
                    logger.warning(f"Expected format: JSON file with list of samples, each containing 'prompt'/'input' and 'answer'/'target' fields")
                    logger.warning(f"Or JSONL file in RULER format with 'index', 'input', 'outputs' fields")
                    continue
        
        if not task_samples:
            raise RuntimeError(
                f"No task samples found. Please generate samples using RULER's scripts first.\n"
                f"Expected sample files in: {samples_dir}\n"
                f"Or run RULER's data generation scripts to create them."
            )
        
        # Step 2: Run inference on samples
        logger.info("Step 2: Running inference on RULER samples...")
        predictions = {}
        
        for task_name, samples in task_samples.items():
            logger.info(f"Processing {len(samples)} samples for task: {task_name}")
            task_predictions = []
            
            # Limit samples if specified
            sample_list = samples[:limit] if limit else samples
            
            for idx, sample in enumerate(tqdm(sample_list, desc=f"Task: {task_name}")):
                try:
                    # Extract prompt from RULER sample format
                    # RULER samples may have different formats, try common fields
                    prompt = sample.get("prompt") or sample.get("input") or sample.get("question") or ""
                    
                    if not prompt:
                        # Try to construct prompt from context and question
                        context = sample.get("context", "")
                        question = sample.get("question", "")
                        if context and question:
                            prompt = f"{context}\n\n{question}"
                        else:
                            logger.warning(f"Sample {idx} has no recognizable prompt field")
                            continue
                    
                    # Per-sample hook (e.g. reset quantized cache between samples).
                    # If the hook returns a value, treat it as past_key_values for this
                    # generation (used by aatc with recent_tokens to pass a fresh
                    # QuantizedDynamicCache so the sliding step updates attention).
                    sample_past_key_values = None
                    if pre_generate_hook is not None:
                        sample_past_key_values = pre_generate_hook()

                    # Generate response
                    response = generate_response(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        max_length=generate_max_length if generate_max_length is not None else max_length,
                        extra_generate_kwargs=extra_generate_kwargs,
                        past_key_values=sample_past_key_values,
                    )
                    
                    # Store prediction
                    prediction = {
                        "sample_id": sample.get("id", idx),
                        "prompt": prompt,
                        "response": response,
                        "ground_truth": sample.get("answer") or sample.get("target") or sample.get("expected_output", ""),
                        "metadata": {k: v for k, v in sample.items() if k not in ["prompt", "input", "question", "answer", "target", "expected_output"]},
                    }
                    task_predictions.append(prediction)
                    
                except RuntimeError as e:
                    error_msg = str(e)
                    # Check if it's a CUDA OOM error
                    if "out of memory" in error_msg.lower() or "NVML_SUCCESS" in error_msg or "CUDACachingAllocator" in error_msg:
                        logger.warning(f"CUDA OOM error on sample {idx} in {task_name}. Skipping this sample.")
                        logger.warning(f"  This may happen with very long sequences or limited GPU memory.")
                        logger.warning(f"  Try using --test_mode or reducing --max_length and --limit")
                        # Clear CUDA cache to free memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Add empty prediction for failed sample
                        prediction = {
                            "sample_id": sample.get("id", idx),
                            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                            "response": "",
                            "ground_truth": sample.get("answer") or sample.get("target") or sample.get("expected_output", ""),
                            "error": "CUDA OOM",
                            "metadata": {k: v for k, v in sample.items() if k not in ["prompt", "input", "question", "answer", "target", "expected_output"]},
                        }
                        task_predictions.append(prediction)
                    else:
                        logger.error(f"Error processing sample {idx} in {task_name}: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        # Clear CUDA cache on any error
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error processing sample {idx} in {task_name}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    # Clear CUDA cache on any error
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                
                # Clear CUDA cache periodically to prevent memory buildup
                if (idx + 1) % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            predictions[task_name] = task_predictions
            
            # Save predictions
            pred_file = os.path.join(predictions_dir, f"{task_name}_predictions.json")
            with open(pred_file, 'w') as f:
                json.dump(task_predictions, f, indent=2)
            logger.info(f"Saved {len(task_predictions)} predictions to {pred_file}")
        
        # Step 3: Evaluate predictions using RULER's evaluation scripts
        logger.info("Step 3: Evaluating predictions...")
        
        for task_name, task_predictions in predictions.items():
            try:
                logger.info(f"Evaluating {task_name}...")
                
                # Try to import RULER's evaluation functions
                eval_functions = {}
                if os.path.exists(synthetic_eval_path):
                    # Try to import evaluation constants and functions
                    constants_file = os.path.join(synthetic_eval_path, "constants.py")
                    if os.path.exists(constants_file):
                        try:
                            spec = importlib.util.spec_from_file_location("ruler_eval_constants", constants_file)
                            if spec and spec.loader:
                                eval_constants = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(eval_constants)
                                # Try to get evaluation function for this task
                                eval_func_name = f"evaluate_{task_name}"
                                if hasattr(eval_constants, eval_func_name):
                                    eval_functions[task_name] = getattr(eval_constants, eval_func_name)
                        except Exception as e:
                            logger.debug(f"Could not import evaluation constants: {e}")
                
                # Extract ground truths and predictions
                ground_truths = []
                model_responses = []
                for pred in task_predictions:
                    gt = pred.get("ground_truth", "")
                    resp = pred.get("response", "")
                    if gt:  # Only include if ground truth exists
                        ground_truths.append(gt)
                        model_responses.append(resp)
                
                if not ground_truths:
                    logger.warning(f"No ground truths found for {task_name}, skipping evaluation")
                    results[task_name] = {
                        "status": "no_ground_truth",
                        "num_samples": len(task_predictions),
                    }
                    continue
                
                # Use RULER's evaluation function if available, otherwise use basic metrics
                task_score = 0.0
                num_correct = 0
                evaluation_method = "basic_exact_match"
                
                if task_name in eval_functions:
                    try:
                        # Use RULER's evaluation function
                        task_score = eval_functions[task_name](model_responses, ground_truths)
                        evaluation_method = "ruler_eval_function"
                    except Exception as e:
                        logger.warning(f"RULER eval function failed for {task_name}: {e}, using basic evaluation")
                        # Fall back to basic evaluation
                        for resp, gt in zip(model_responses, ground_truths):
                            if resp.strip().lower() == gt.strip().lower():
                                num_correct += 1
                        task_score = (num_correct / len(ground_truths) * 100) if ground_truths else 0.0
                else:
                    # RULER-style string_match scoring (mirrors RULER's evaluate.py):
                    # For each sample, check whether each expected answer token/phrase
                    # appears as a substring of the model response (case-insensitive).
                    # Multi-answer ground truths are stored as lists; single answers as str.
                    # Full credit (1.0) for a hit — no partial-credit halving.
                    for resp, gt in zip(model_responses, ground_truths):
                        resp_lower = resp.strip().lower()
                        # gt may be a list (multi-answer) or a plain string
                        if isinstance(gt, list):
                            refs = [str(g).strip().lower() for g in gt if str(g).strip()]
                        else:
                            # Split on common delimiters used by RULER for multi-value answers
                            gt_str = str(gt).strip().lower()
                            refs = [gt_str]
                        if refs:
                            # Score = fraction of expected answers found in response
                            hits = sum(1 for r in refs if r in resp_lower)
                            num_correct += hits / len(refs)

                    task_score = (num_correct / len(ground_truths) * 100) if ground_truths else 0.0
                
                results[task_name] = {
                    "score": task_score,
                    "accuracy": task_score,  # For compatibility
                    "num_samples": len(task_predictions),
                    "num_evaluated": len(ground_truths),
                    "num_correct": int(num_correct) if evaluation_method == "basic_exact_match" else None,
                    "evaluation_method": evaluation_method,
                }
                all_scores.append(task_score)
                
                logger.info(f"{task_name}: {task_score:.2f}% ({evaluation_method}, {len(ground_truths)} samples evaluated)")
                    
            except Exception as e:
                logger.error(f"Error evaluating {task_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                results[task_name] = {
                    "error": str(e),
                    "num_samples": len(task_predictions) if task_name in predictions else 0,
                }
        
        # Calculate summary
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            results["summary"] = {
                "average_score": avg_score,
                "num_tasks": len([k for k in results.keys() if k != "summary" and "error" not in results.get(k, {})]),
                "scores": all_scores,
            }
        else:
            results["summary"] = {
                "status": "no_scores_calculated",
                "num_tasks": len([k for k in results.keys() if k != "summary"]),
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in RULER evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def run_ruler_benchmark(
    model_path: str,
    output_file: str,
    ruler_repo_path: Optional[str] = None,
    root_dir: Optional[str] = None,
    tasks: Optional[List[str]] = None,
    task_category: Optional[str] = None,
    task_filter: Optional[str] = None,
    max_length: int = 32768,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    use_flash2: bool = False,
    limit: Optional[int] = None,
    num_samples: int = 500,
    bitwidth_file: Optional[str] = None,
    scaling_type: str = "tokenwise",
    channelwise_scaling_file: Optional[str] = None,
    disable_quantization: bool = False,
    lt_bits: int = 16,
    lt_group_size: int = 0,
    lt_sym: bool = False,
    lt_clip_ratio: float = 1.0,
    lt_hadamard: bool = False,
    tokenizer_path: Optional[str] = None,
):
    """
    Run RULER benchmark on a model using the RULER repository.
    
    Args:
        model_path: Path to the model
        output_file: Path to save the results JSON file
        ruler_repo_path: Path to RULER repository (if None, uses detected path)
        root_dir: Root directory for RULER outputs (default: ./ruler_outputs)
        tasks: List of RULER task names (if None, uses all tasks or filtered by category/filter)
        task_category: Filter tasks by category (e.g., "retrieval", "lookup", "multi_hop", "aggregation", "qa")
        task_filter: Filter tasks by pattern (e.g., "niah_*" for all niah tasks)
        max_length: Maximum sequence length for evaluation
        max_new_tokens: Maximum tokens to generate per response
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        use_flash2: Use Flash Attention 2
        limit: Limit number of examples per task (None for all)
        num_samples: Number of samples to generate per task if samples don't exist
        bitwidth_file: Path to bitwidth allocation file
        scaling_type: Scaling type for quantization
        channelwise_scaling_file: Path to channelwise scaling file
        disable_quantization: Disable quantization even if bitwidth_file is provided
        lt_bits: Bits for low-rank latents
        lt_group_size: Group size for low-rank latents
        lt_sym: Symmetric quantization for low-rank latents
        lt_clip_ratio: Clip ratio for low-rank latents
        lt_hadamard: Apply Hadamard transform to low-rank latents
    """
    logger.info("=" * 80)
    logger.info("RULER Benchmark Evaluation")
    logger.info("=" * 80)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output file: {output_file}")
    
    # Use provided ruler_repo_path or detected one
    if ruler_repo_path:
        global RULER_REPO_PATH, RULER_DIRECT_AVAILABLE
        if os.path.exists(ruler_repo_path) and os.path.exists(os.path.join(ruler_repo_path, "scripts")):
            RULER_REPO_PATH = os.path.abspath(ruler_repo_path)
            RULER_DIRECT_AVAILABLE = True
        else:
            raise FileNotFoundError(
                f"RULER repository not found at {ruler_repo_path}. "
                f"Please clone it: git clone https://github.com/NVIDIA/RULER"
            )
    
    if not RULER_DIRECT_AVAILABLE or not RULER_REPO_PATH:
        raise RuntimeError(
            "RULER repository not found. Please:\n"
            "  1. Clone the repository: git clone https://github.com/NVIDIA/RULER\n"
            "  2. Set RULER_REPO_PATH environment variable or use --ruler_repo_path argument"
        )
    
    logger.info(f"RULER repository: {RULER_REPO_PATH}")
    logger.info(f"Max length: {max_length}")
    logger.info(f"Max new tokens: {max_new_tokens}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Top-p: {top_p}")
    logger.info("=" * 80)
    
    # Set default root_dir if not provided
    if root_dir is None:
        root_dir = os.path.join(os.path.dirname(__file__), "ruler_outputs")
    
    logger.info(f"RULER root directory: {root_dir}")
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_path, use_flash_attn2=use_flash2)
    logger.info("Model loaded successfully")
    
    # Configure quantization if bitwidth file is provided
    if bitwidth_file and configure_model_quantization:
        logger.info(f"Configuring per-dimension quantization from {bitwidth_file}...")
        logger.info(f"  Scaling type: {scaling_type}")
        if scaling_type == "channelwise" and channelwise_scaling_file:
            logger.info(f"  Channelwise scaling file: {channelwise_scaling_file}")
        
        configure_model_quantization(
            model,
            bitwidth_file=bitwidth_file,
            scaling_type=scaling_type,
            channelwise_scaling_file=channelwise_scaling_file,
        )
        
        # Disable quantization if requested
        if disable_quantization:
            logger.info("Disabling quantization...")
            num_disabled = 0
            for name, module in model.named_modules():
                if hasattr(module, 'use_per_dim_quantization') and module.use_per_dim_quantization:
                    module.use_per_dim_quantization = False
                    num_disabled += 1
            logger.info(f"Disabled quantization in {num_disabled} module(s)")
        else:
            # Check if quantization is enabled
            has_quantization = any(
                hasattr(m, 'use_per_dim_quantization') and m.use_per_dim_quantization
                for m in model.modules()
            )
            if has_quantization:
                logger.info("✓ Per-dimension quantization ENABLED")
            else:
                logger.warning("⚠ Per-dimension quantization CONFIGURED but DISABLED")
    elif bitwidth_file:
        logger.warning("Bitwidth file provided but configure_model_quantization not available. Skipping quantization.")
    else:
        # Use uniform latent quantization (standard quantization)
        if lt_bits < 16:
            logger.info(f"Using uniform latent quantization with {lt_bits} bits")
            configure_latent_quantizer(
                model, 
                n_bits=lt_bits,
                group_size=lt_group_size,
                sym=lt_sym,
                clip_ratio=lt_clip_ratio,
                hadamard=lt_hadamard
            )
        else:
            logger.info("No quantization configured - using full precision")
    
    # Prepare results structure
    results = {
        "model_path": model_path,
        "ruler_repo_path": RULER_REPO_PATH,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "root_dir": root_dir,
            "tasks": tasks,
            "task_category": task_category,
            "task_filter": task_filter,
            "num_samples": num_samples,
            "quantization_enabled": (bitwidth_file is not None and not disable_quantization) or lt_bits < 16,
            "bitwidth_file": bitwidth_file,
            "scaling_type": scaling_type,
            "channelwise_scaling_file": channelwise_scaling_file,
            "lt_bits": lt_bits,
            "lt_group_size": lt_group_size,
            "lt_sym": lt_sym,
            "lt_clip_ratio": lt_clip_ratio,
            "lt_hadamard": lt_hadamard,
        },
        "results": {},
        "summary": {}
    }
    
    # Run evaluation
    try:
        eval_results = run_ruler_evaluation(
            model=model,
            tokenizer=tokenizer,
            model_path=model_path,
            ruler_repo_path=RULER_REPO_PATH,
            root_dir=root_dir,
            tasks=tasks,
            task_category=task_category,
            task_filter=task_filter,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            limit=limit,
            num_samples=num_samples,
            tokenizer_path=tokenizer_path,
        )
        results["results"] = eval_results
        
        # Process evaluation results
        if isinstance(eval_results, dict):
            # Check if we have actual scores
            if "summary" in eval_results and "average_score" in eval_results["summary"]:
                results["summary"] = eval_results["summary"]
            elif any("score" in str(v) or "accuracy" in str(v) for v in eval_results.values() if isinstance(v, dict)):
                # Extract scores from task results
                all_scores = []
                for task_name, task_result in eval_results.items():
                    if isinstance(task_result, dict):
                        score = task_result.get("score") or task_result.get("accuracy")
                        if score is not None:
                            all_scores.append(float(score))
                
                if all_scores:
                    results["summary"] = {
                        "average_score": sum(all_scores) / len(all_scores),
                        "num_tasks": len([k for k in eval_results.keys() if k != "summary"]),
                        "scores": all_scores,
                    }
                else:
                    results["summary"] = {
                        "status": "evaluation_completed",
                        "num_tasks": len([k for k in eval_results.keys() if k != "summary"]),
                        "note": "See results for detailed output",
                    }
            else:
                results["summary"] = {
                    "status": "evaluation_completed",
                    "num_tasks": len([k for k in eval_results.keys() if k != "summary"]),
                    "note": "See results for detailed output",
                }
        else:
            results["summary"] = {
                "status": "evaluation_completed",
                "note": "See results for detailed output",
            }
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        results["results"] = {}
        results["summary"] = {
            "error": str(e),
        }
    
    # Save results
    logger.info(f"\nSaving results to {output_file}...")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("=" * 80)
    logger.info("Benchmark completed!")
    if results.get("summary") and "average_score" in results["summary"]:
        logger.info(f"Average score: {results['summary']['average_score']:.2f}")
        logger.info(f"Number of tasks: {results['summary'].get('num_tasks', 0)}")
    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run RULER benchmark evaluation on an LLM model using the official RULER repository"
    )
    
    # Model arguments (required)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to the model to evaluate"
    )
    
    # RULER repository arguments
    parser.add_argument(
        "--ruler_repo_path",
        type=str,
        default=None,
        help="Path to RULER repository (if not set, will try to auto-detect or use RULER_REPO_PATH env var)"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Local path to tokenizer files for RULER sample generation. "
             "Use this when --model_name_or_path is a HuggingFace Hub ID "
             "(e.g. point to a local PALU model that has the same tokenizer)."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Root directory for RULER outputs (default: ./ruler_outputs)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the results JSON file (default: ./results/ruler_benchmark_<timestamp>.json)"
    )
    
    # Task arguments
    parser.add_argument(
        "--tasks",
        type=lambda s: [item.strip() for item in s.split(',')] if s else [],
        default=None,
        help="Comma-separated list of RULER task names (e.g., 'niah_single_1,niah_single_2'). "
             "If not specified, uses all tasks or filtered by category/filter."
    )
    parser.add_argument(
        "--task_category",
        type=str,
        default=None,
        choices=["retrieval", "lookup", "multi_hop", "tracing", "aggregation", "qa", "question_answering", "all"],
        help="Filter tasks by category. Options: "
             "'retrieval'/'lookup' (niah tasks), "
             "'multi_hop'/'tracing' (variable_tracking), "
             "'aggregation' (cwe, fwe), "
             "'qa'/'question_answering' (qa tasks), "
             "'all' (all tasks)"
    )
    parser.add_argument(
        "--task_filter",
        type=str,
        default=None,
        help="Filter tasks by pattern using shell-style wildcards (e.g., 'niah_*' for all niah tasks, '*_single_*' for single tasks)"
    )
    
    # Generation arguments
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Enable test mode with smaller defaults for testing on limited GPU memory. "
             "Sets: max_length=4096, num_samples=10, limit=5 per task"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32768,
        help="Maximum sequence length for evaluation (default: 32768, use smaller values like 4096 or 8192 for limited GPU memory)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate per response"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter"
    )
    
    # Other arguments
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task to evaluate (for testing). "
             "Useful for quick tests on limited GPU memory (e.g., --limit 5)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to generate per task (default: 500, use smaller values like 10-50 for testing)"
    )
    
    # Quantization arguments
    parser.add_argument(
        "--bitwidth_file",
        type=str,
        default=None,
        help="Path to the .pt file with bitwidth allocations (enables per-dimension quantization)"
    )
    parser.add_argument(
        "--scaling_type",
        type=str,
        default="tokenwise",
        choices=["tokenwise", "channelwise"],
        help="Scaling type: 'tokenwise' (default) computes alpha per token, "
             "'channelwise' uses pre-computed scalings per dimension"
    )
    parser.add_argument(
        "--channelwise_scaling_file",
        type=str,
        default=None,
        help="Path to .pt file with channelwise scalings (required if --scaling_type=channelwise). "
             "Generated by run_channelwise_scaling_calibration.py"
    )
    parser.add_argument(
        "--disable_quantization",
        action="store_true",
        help="Disable quantization even if bitwidth_file is provided (useful for comparison)"
    )
    
    # Add common args (for low-rank latent quantization, etc.)
    parser.add_argument('--lt_bits', type=int, help='Bits for low_rank latents. \
                        When lt_bits is less than 16, we quantize the low_rank latents in low_bits', default=16)
    parser.add_argument('--lt_group_size', type=int, help='Group size for low_rank latents', default=0)
    parser.add_argument('--lt_sym', action='store_true', help='Symmetric quantization for low_rank latents')
    parser.add_argument('--lt_clip_ratio', type=float, help='Clip ratio for low_rank latents', default=1.0)
    parser.add_argument('--lt_hadamard', action='store_true', help='Apply Hadamard transform to low_rank latents')
    parser.add_argument('--flash2', action='store_true', help='whether to use flash-attention2')
    
    args = parser.parse_args()
    
    # Apply test mode defaults if enabled
    if args.test_mode:
        logger.info("=" * 80)
        logger.info("TEST MODE ENABLED - Using smaller defaults for limited GPU memory")
        logger.info("=" * 80)
        if args.max_length == 32768:  # Only override if not explicitly set
            args.max_length = 4096
            logger.info(f"  max_length: {args.max_length} (test mode default)")
        if args.num_samples == 500:  # Only override if not explicitly set
            args.num_samples = 10
            logger.info(f"  num_samples: {args.num_samples} (test mode default)")
        if args.limit is None:  # Only set if not explicitly provided
            args.limit = 5
            logger.info(f"  limit: {args.limit} examples per task (test mode default)")
        logger.info("=" * 80)
    
    # Set default output file if not provided
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(args.model_name_or_path).replace("/", "_")
        os.makedirs("./results", exist_ok=True)
        args.output_file = f"./results/ruler_benchmark_{model_name}_{timestamp}.json"
    
    # Validate arguments
    if args.scaling_type == "channelwise" and not args.channelwise_scaling_file:
        parser.error("--channelwise_scaling_file is required when --scaling_type=channelwise")
    
    # Check for RULER repository
    ruler_repo_path = args.ruler_repo_path or RULER_REPO_PATH
    if not ruler_repo_path:
        parser.error(
            "RULER repository not found. Please:\n"
            "  1. Clone the repository: git clone https://github.com/NVIDIA/RULER\n"
            "  2. Set RULER_REPO_PATH environment variable or use --ruler_repo_path argument"
        )
    
    # Run benchmark
    run_ruler_benchmark(
        model_path=args.model_name_or_path,
        output_file=args.output_file,
        ruler_repo_path=ruler_repo_path,
        root_dir=args.root_dir,
        tasks=args.tasks,
        task_category=args.task_category,
        task_filter=args.task_filter,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_flash2=args.flash2,
        limit=args.limit,
        num_samples=args.num_samples,
        bitwidth_file=args.bitwidth_file,
        scaling_type=args.scaling_type,
        channelwise_scaling_file=args.channelwise_scaling_file,
        disable_quantization=args.disable_quantization,
        lt_bits=args.lt_bits,
        lt_group_size=args.lt_group_size,
        lt_sym=args.lt_sym,
        lt_clip_ratio=args.lt_clip_ratio,
        lt_hadamard=args.lt_hadamard,
        tokenizer_path=args.tokenizer_path,
    )


if __name__ == "__main__":
    main()
