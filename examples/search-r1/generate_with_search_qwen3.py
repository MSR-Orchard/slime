# Adapted from generate_with_search.py for Qwen3 native tool calling format
# This version uses Qwen3's built-in tool calling support via apply_chat_template(tools=...)

import asyncio
from typing import Any

from qa_em_format import compute_score_em

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Import sglang tool parser
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import Function, Tool

# Configuration for Search-R1 with Qwen3 native tool calling
SEARCH_R1_CONFIGS = {
    # ============== General Configuration ==============
    "max_turns": 10,
    "topk": 3,
    "search_concurrency": 256,
    # ============== Search Backend Selection ==============
    "search_backend": "local",  # Options: "local" or "google"
    # ============== Local Search Configuration ==============
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",
        "proxy": None,
    },
    # ============== Google Search Configuration ==============
    "google": {
        "api_key": "your_api_key_here",
        "snippet_only": True,
        "proxy": None,
    },
    # ============== Reward Model Configuration ==============
    "format_score": 0.2,
}

SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])

# Tool definition in OpenAI format for Qwen3
SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information on the web. Use this tool when you need to find facts or information to answer the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up information.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def _get_tool_parser() -> FunctionCallParser:
    """Create a FunctionCallParser for parsing Qwen3 tool calls."""
    tools_list = [
        Tool(
            function=Function(
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                parameters=tool["function"]["parameters"],
            ),
            type=tool["type"],
        )
        for tool in SEARCH_TOOLS
    ]
    return FunctionCallParser(tools=tools_list, tool_call_parser="qwen25")


def _passages2string(retrieval_result: list[dict]) -> str:
    """Convert retrieval results to a formatted string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference


async def search(query: str) -> str:
    """Perform search using either local or Google search backend."""
    backend = SEARCH_R1_CONFIGS["search_backend"]

    if backend == "local":
        from local_search_server import local_search

        local_config = SEARCH_R1_CONFIGS["local"]
        result = await local_search(
            local_config["search_url"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            proxy=local_config["proxy"],
        )
    elif backend == "google":
        from google_search_server import google_search

        google_config = SEARCH_R1_CONFIGS["google"]
        result = await google_search(
            google_config["api_key"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            snippet_only=google_config["snippet_only"],
            proxy=google_config["proxy"],
        )
    else:
        raise ValueError(f"Unknown search backend: {backend}")

    return _passages2string(result)


def parse_tool_calls(response: str) -> dict[str, Any]:
    """
    Parse tool calls from Qwen3 response using sglang's FunctionCallParser.
    
    Returns:
        dict with keys:
            - "normal_text": text outside tool calls
            - "calls": list of parsed tool calls
    """
    parser = _get_tool_parser()
    normal_text, calls = parser.parse_non_stream(response)
    return {
        "normal_text": normal_text,
        "calls": [call.model_dump() for call in calls] if calls else [],
    }


async def execute_tool_call(tool_name: str, tool_args: dict[str, Any]) -> tuple[str, bool]:
    """
    Execute a tool call and return the result.
    
    Returns:
        tuple of (tool_result, is_done)
    """
    if tool_name == "search":
        query = tool_args.get("query", "")
        if not query:
            return "Error: search query is empty", False
        
        async with SEMAPHORE:
            search_results = await search(query)
        return search_results.strip(), False
    else:
        return f"Error: Unknown tool '{tool_name}'", False


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """
    Generate function using Qwen3's native tool calling format.
    
    This function uses apply_chat_template with tools parameter to enable
    Qwen3's native tool calling support.
    """
    assert not args.partial_rollout, "Partial rollout is not supported."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Initialize conversation with user message
    # The prompt from the dataset is the user's question
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided search results. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."},
    ]

    messages.extend(sample.prompt)
    
    # Track the original prompt for reward calculation
    original_prompt = sample.prompt

    # Apply chat template with tools to get initial prompt
    prompt_text = state.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=SEARCH_TOOLS,
    )
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Initialize tracking
    response_token_ids = []
    loss_mask = []
    full_response = ""

    for _turn_idx in range(SEARCH_R1_CONFIGS["max_turns"]):
        # Apply chat template for current conversation state
        current_prompt = state.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=SEARCH_TOOLS,
        )

        payload = {
            "text": current_prompt,
            "sampling_params": sampling_params,
        }

        output = await post(url, payload)

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        response = output["text"]
        
        # Remove end token if present
        if response.endswith("<|im_end|>"):
            response = response[: -len("<|im_end|>")]

        # Parse tool calls from response
        parsed = parse_tool_calls(response)
        normal_text = parsed["normal_text"]
        tool_calls = parsed["calls"]

        # Tokenize this response segment
        cur_response_token_ids = state.tokenizer(response, add_special_tokens=False)["input_ids"]
        response_token_ids.extend(cur_response_token_ids)
        loss_mask.extend([1] * len(cur_response_token_ids))
        full_response += response

        # Check if model made a tool call
        if tool_calls:
            # Process the first tool call (typically models make one at a time)
            tool_call = tool_calls[0]
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("parameters", {})
            
            # Parse parameters if they're a string
            if isinstance(tool_args, str):
                import json
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {}

            # Add assistant message with tool call to conversation
            messages.append({
                "role": "assistant",
                "content": normal_text if normal_text.strip() else None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_args,
                        },
                        "id": f"call_{_turn_idx}",
                    }
                ],
            })

            # Execute the tool
            tool_result, is_done = await execute_tool_call(tool_name, tool_args)

            # Add tool response to conversation
            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": tool_result,
            })

            # Tokenize the tool response (with loss_mask=0 since it's not model output)
            # We need to get the token delta for the tool message
            full_conv_text = state.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=SEARCH_TOOLS,
            )
            full_conv_tokens = state.tokenizer(full_conv_text, add_special_tokens=False)["input_ids"]

            # print("Full conversation text so far:", full_conv_text)
            
            # Calculate how many tokens the tool response added
            prev_length = len(prompt_token_ids) + len(response_token_ids)
            tool_response_tokens = full_conv_tokens[prev_length:]
            
            response_token_ids.extend(tool_response_tokens)
            loss_mask.extend([0] * len(tool_response_tokens))
            
            # Update full_response to include tool interaction marker
            full_response += f"\n[Tool: {tool_name}]\n[Result: {tool_result[:]}]\n"

            if is_done:
                break
        else:
            # No tool call - model is giving final answer
            messages.append({"role": "assistant", "content": response})
            break

        # Check for length limit
        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

    # Build final sample
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = full_response
    sample.loss_mask = loss_mask

    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """
    Reward function for retrieval-based QA with Qwen3 format.
    
    This adapts the EM reward to work with Qwen3's native tool calling output.
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # For Qwen3 format, we need to reconstruct the solution string
    # in a format compatible with the EM scorer
    # The scorer expects <answer>...</answer> tags
    
    # Try to extract the final answer from the response
    response = sample.response
    
    # Convert Qwen3 format to the expected format for scoring
    # Look for the last assistant response that doesn't contain tool calls
    solution_str = f"<|im_start|>assistant\n{response}"
    
    # If there's no explicit answer tag, try to extract from final response
    if "<answer>" not in solution_str:
        # Wrap the final response as an answer for compatibility
        # This is a simplified approach - you may want to be more sophisticated
        lines = response.strip().split("\n")
        # Get the last non-empty, non-tool line as the answer
        final_answer = ""
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith("[Tool:") and not line.startswith("[Result:"):
                final_answer = line
                break
        if final_answer:
            solution_str = f"<|im_start|>assistant\n<think>Using search results</think>\n<answer>{final_answer}</answer>"

    # print('--'*20)
    # print("Prompt is : ", sample.prompt)
    # print("Generated Raw string scoring:",  response)    
    score = compute_score_em(
        solution_str=solution_str,
        ground_truth=sample.label["ground_truth"],
        format_score=SEARCH_R1_CONFIGS["format_score"],
    )
    # print("Computed EM score:", score)

    return score
