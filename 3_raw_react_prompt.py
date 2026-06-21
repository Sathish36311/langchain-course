from dotenv import load_dotenv
import re
import inspect

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# --- Tools (LangChain @tool decorator) ---
@traceable(run_type="tool")
def get_product_price(product_name: str) -> float:
    """Look up the price of the product."""
    print(f"    >> Executing get_product_price for '{product_name}'")
    product_prices = { "laptop": 999.99,"keyboard": 49.99,"mouse": 19.99,"monitor": 199.99,"headphones": 89.99 }
    return product_prices.get(product_name.lower(), -1.0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to the price and return the discounted price. Available discount tiers are: 'regular' (0% discount), 'premium' (10% discount), and 'vip' (20% discount)."""
    print(f"    >> Executing apply_discount for price {price} with discount tier {discount_tier}")
    price = float(price)
    discount_percentages = {"regular": 0,  "premium": 10, "vip": 20}
    discount_percentage = discount_percentages.get(discount_tier.lower(), 0)
    if price < 0 or discount_percentage < 0 or discount_percentage > 100:
        raise ValueError("Invalid price or discount percentage.")
    return price * (1 - discount_percentage / 100)

def get_tool_descriptions(tools_dict):
    tool_descriptions = []
    for tool_name, tool_func in tools_dict.items():
        # __wrapped__ is used to get the original function if it's decorated
        original_func = getattr(tool_func, "__wrapped__", tool_func)
        signature = inspect.signature(original_func)
        doc_string = inspect.getdoc(original_func) or "No description available."
        tool_descriptions.append(f"{tool_name}{signature}: {doc_string}")
    return "\n".join(tool_descriptions)

tools = { "get_product_price": get_product_price,  "apply_discount": apply_discount}
tool_descriptions = get_tool_descriptions(tools)
tool_names = ', '.join(tools.keys())

react_prompt = f"""
STRICT RULES you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:
"""


# Helper trace ollama call
@traceable(run_type="llm", name="ollama chat")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


# -- Agent Loop ---
@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    print(f"Question : {question}")
    print("="*60)

    prompt = react_prompt.format(question=question)
    scratchpad= ""
    
    for iteration in range(1,MAX_ITERATIONS +1):
        print(f"\n--- Iteration {iteration} ---")
        full_prompt = prompt + scratchpad

        response = ollama_chat_traced(
            model =MODEL, 
            messages = [{"role": "user", "content": full_prompt}],
            options = {"stop": ["\nObservation"], "temperature": 0}
            )
        
        output = response.message.content
        print(f"LLM Output : {output}")

        answer_match =  re.search(r"Final Answer:\s*(.+)", output)
        if answer_match:
            final_answer = answer_match.group(1).strip()
            print("\n" + "=" * 60)
            print(f"Final Answer : {final_answer}")
            return final_answer
        
        # Parse tool call
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print("Cannot find the tool or tool name")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        
        print(f"Tool call detected: {tool_name} with args {tool_input_raw}")

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=",1)[-1].strip().strip("'\"") for x in raw_args]

        print(f" Tool Executing {tool_name}({args})..")

        if tool_name not in tools:
            observation = f"{tool_name} tool not found"
        else:
            observation = str(tools[tool_name](*args))

       
        print(f"Result: {observation}")
        scratchpad += f"{output}\nObservation: {observation}\nThought:"


    print("Max iterations reached without a final answer.")
    return "Sorry, I couldn't find the answer within the iteration limit."

if __name__ == "__main__":
    print("Welcome to the ReAct agent demo!")
    print()
    user_question = "What is the price of a laptop with a premium discount?"
    print(f"User question: {user_question}")
    result = run_agent(user_question)
    print(f"Final answer: {result}")