from dotenv import load_dotenv

load_dotenv()

from typing import List, Sequence
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import MessageGraph, END

from chains import generate_chain, reflect_chain

REFLECT = "Reflect"
GENERATE = "Generate"


def generation_node(state: Sequence[BaseMessage]):
    """Node to generate a tweet based on user input and critique."""
    return generate_chain.invoke({"messages": state})


def reflection_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
    """Node to reflect on the generated tweet and provide critique."""
    result = reflect_chain.invoke({"messages": state})
    return [HumanMessage(content=result.content)]


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT


flow = MessageGraph()
flow.add_node(GENERATE, generation_node)
flow.add_node(REFLECT, reflection_node)
flow.add_conditional_edges(GENERATE, should_continue, {END: END, REFLECT: REFLECT})
flow.add_edge(REFLECT, GENERATE)
flow.set_entry_point(GENERATE)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="reflection_flow.png")

if __name__ == "__main__":
    inputs = HumanMessage(
        content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
    )
    response = app.invoke(inputs)
    print(response[-1].content)
