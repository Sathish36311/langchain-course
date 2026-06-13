from dotenv import load_dotenv

load_dotenv()


from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# --- Tools (LangChain @tool decorator) ---

@tool
def get_product_price(product_name: str) -> float:
    """Look up the price of the product."""
    print(f"    >> Executing get_product_price for '{product_name}'")
    product_prices = { "laptop": 999.99,"keyboard": 49.99,"mouse": 19.99,"monitor": 199.99,"headphones": 89.99 }
    return product_prices.get(product_name.lower(), -1.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to the price and return the discounted price. Available discount tiers are: 'regular' (0% discount), 'premium' (10% discount), and 'vip' (20% discount)."""
    print(f"    >> Executing apply_discount for price {price} with discount tier {discount_tier}")
    discount_percentages = {"regular": 0,  "premium": 10, "vip": 20}
    discount_percentage = discount_percentages.get(discount_tier.lower(), 0)
    if price < 0 or discount_percentage < 0 or discount_percentage > 100:
        raise ValueError("Invalid price or discount percentage.")
    return price * (1 - discount_percentage / 100)


   # --- Agent Loop (LangSmith @traceable decorator) --- 

@traceable(project="ReAct Under The Hood", name="Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages =[
        SystemMessage(
            content= (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received " 
                "a price from get_product_price. Pass the exact price " 
                "returned by get_product_price do NOT pass a made-up number.\n" 
                "3. NEVER calculate discounts yourself using math. " 
                "Always use the apply_discount tool.\n" 
                "4. If the user does not specify a discount tier, " 
                "ask them which tier to use do NOT assume one."
            ),
        ),
        HumanMessage(content=question),
    ]
    
    for iteration in range(1,MAX_ITERATIONS +1):
        print(f"\n--- Iteration {iteration} ---")
        ai_messages = llm_with_tools.invoke(messages)

        tool_calls = ai_messages.tool_calls

        # If no tool calls, we assume the model is done and has returned a final answer
        if not tool_calls:
            print("No tool calls detected, assuming final answer.")
            return ai_messages.content
        
        # Process only the first tool call in this iteration (if multiple, the model will call the rest in subsequent iterations)
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")  
        print(f"Tool call detected: {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if not tool_to_use:
            print(f"Final Answer: {ai_messages.content}")
            return ai_messages.content

        observation = tool_to_use.invoke(tool_args)
        print(f"Observation from tool call: {observation}")

        messages.append(ai_messages)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("Max iterations reached without a final answer.")
    return "Sorry, I couldn't find the answer within the iteration limit."

if __name__ == "__main__":
    print("Welcome to the ReAct agent demo!")
    print()
    user_question = "What is the price of a laptop with a premium discount?"
    print(f"User question: {user_question}")
    result = run_agent(user_question)
    print(f"Final answer: {result}")