from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from typing import List
from agent import Agent 

    
class Chatbot:
    def __init__(self, agent: Agent, thread_id: str = "user-123"):
        self.agent = agent
        self.thread = {"configurable": {"thread_id": thread_id}}
        self.history: List = []

    def send(self, user_input: str) -> List[str]:
        self.history.append(HumanMessage(content=user_input))
        responses = []

        for event in self.agent.graph.stream({"messages": self.history}, self.thread):
            for output in event.values():
                if isinstance(output, dict) and "messages" in output:
                    new_msgs = output["messages"]
                    self.history.extend(new_msgs)
                    for msg in new_msgs:
                        if hasattr(msg, "content") and msg.content:
                            responses.append(msg.content)
        return responses