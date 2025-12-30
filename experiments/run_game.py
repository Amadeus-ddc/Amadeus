import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from amadeus.core.graph import MemoryGraph
from amadeus.agents.builder import BuilderAgent
from amadeus.agents.questioner import QuestionerAgent
from amadeus.agents.answerer import AnswererAgent
from amadeus.engine.optimizer import AdversarialOptimizer

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    # 1. Init Components
    graph = MemoryGraph(storage_path="data/test_game_graph.json")
    builder = BuilderAgent(graph)
    questioner = QuestionerAgent()
    answerer = AnswererAgent(graph)
    optimizer = AdversarialOptimizer(questioner, builder, answerer)

    # 2. Simulated Data Stream (LoCoMo style)
    stream = [
        "--- Session Context: 2023-05-01 ---",
        "Caroline: I'm so excited about my pottery class tomorrow.",
        "Melanie: That sounds fun! Where is it?",
        "Caroline: It's at the community center on 5th Avenue.",
        "Melanie: Oh, I used to go there for yoga.",
        "Caroline: Really? I didn't know you did yoga.",
        "Melanie: Yeah, years ago. Before I hurt my knee.",
    ]
    
    buffer_content = "\n".join(stream)
    
    print("=== 1. BUILDER PHASE ===")
    builder.process_buffer(buffer_content)
    
    print("\n=== 2. OPTIMIZER PHASE (SELF-PLAY) ===")
    optimizer.step(buffer_content)
    
    print("\n=== 3. FINAL GUIDELINES ===")
    print("Builder Guidelines:")
    print(builder._format_guidelines())
    print("Answerer Guidelines:")
    print(answerer._format_guidelines())

if __name__ == "__main__":
    main()
