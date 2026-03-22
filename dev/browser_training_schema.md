# Browser Automation Training Schema (JSONL)
# Format: Each line contains a full "Thought -> Action -> Observation -> Final Answer" trajectory.

# Schema Breakdown:
# - <|user_start|>: The original user request (e.g., "Find the price of a MacBook Pro on Apple.com").
# - <|thought_start|>: The model's reasoning process (e.g., "I need to navigate to Apple.com and search for MacBook Pro").
# - <|action_start|>: A JSON tool call to the browser manager (e.g., {"action": "navigate", "params": {"url": "https://apple.com"}}).
# - <|observation_start|>: The environment's response after executing the action (e.g., "Successfully navigated. Page title: Apple (United States)").
# - <|assistant_start|>: The final output or next reasoning step.

# Example Dataset Entry (Single Trajectory):
{
  "messages": [
    {"role": "user", "content": "Tell me the price of the base MacBook Air 13-inch."},
    {"role": "thought", "content": "I'll start by navigating to apple.com to find the MacBook Air product page."},
    {"role": "action", "content": "{\"action\": \"navigate\", \"params\": {\"url\": \"https://www.apple.com/macbook-air/\"}}"},
    {"role": "observation", "content": "Successfully navigated to 'https://www.apple.com/macbook-air/'. Page title: 'MacBook Air - Apple'"},
    {"role": "thought", "content": "Now I need to find the interactive 'Buy' or 'Pricing' link to see specific model prices."},
    {"role": "action", "content": "{\"action\": \"click\", \"params\": {\"selector\": \"a.ac-gn-link-mac\"}}"},
    {"role": "observation", "content": "Clicked element matching 'a.ac-gn-link-mac'. Page updated."},
    {"role": "thought", "content": "I've reached the pricing section. The summary shows the 13-inch MacBook Air with M2 starts at $999."},
    {"role": "assistant", "content": "The base 13-inch MacBook Air with the M2 chip starts at $999."}
  ],
  "metadata": {
    "task": "browser_navigation",
    "success": true,
    "total_steps": 2
  }
}

# Implementation Notes for NanoChat Training:
# 1. Modify nanochat/tokenizer.py to include <|thought_start|>, <|action_start|>, <|observation_start|> as special tokens.
# 2. Use scripts/chat_sft.py to fine-tune on these trajectories.
# 3. Use the reward signals from browser_tool.py (e.g., did the page load? was the element found?) to generate RL data for scripts/chat_rl.py.
