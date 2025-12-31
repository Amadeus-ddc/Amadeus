"""
Integration test demonstrating Buffer + Builder + Graph working together.
"""
import os
from dotenv import load_dotenv
from amadeus.core.graph import MemoryGraph
from amadeus.core.buffer import ShortTermBuffer
from amadeus.agent.builder import BuilderAgent


def test_lightmem_integration():
    """
    Test the complete LightMem workflow:
    1. Add content to Buffer (short-term memory)
    2. Builder processes buffer and extracts facts into Graph (long-term memory)
    3. Builder returns texts that need to WAIT (ambiguous info)
    4. Buffer keeps only the deferred texts for next round
    """
    print("🧪 Testing LightMem Integration...\n")
    
    # Load environment variables
    load_dotenv()
    
    # 1. Initialize components
    print("1️⃣ Initializing Memory System...")
    graph = MemoryGraph("data/test_lightmem_graph.json")
    buffer = ShortTermBuffer(max_size=5)
    builder = BuilderAgent(graph)
    print(f"   Graph: {graph.graph.number_of_nodes()} nodes")
    print(f"   Buffer: {buffer}")
    print()
    
    # 2. Simulate first conversation round
    print("2️⃣ First Round - Adding to Buffer...")
    buffer.add("User: Hi, I'm Alice. I work at Google.")
    buffer.add("AI: Nice to meet you, Alice.")
    buffer.add("User: He is my colleague.")  # Ambiguous - who is "he"?
    buffer.add("User: We're working on a search algorithm.")
    
    print(f"   Buffer size: {buffer.size()}")
    print(f"   Buffer content:\n{buffer.get_all_content()}")
    print()
    
    # 3. Builder processes buffer
    print("3️⃣ Builder Processing Buffer...")
    buffer_content = buffer.get_all_content()
    texts_to_keep = builder.process_buffer(buffer_content)
    
    print(f"   Builder returned {len(texts_to_keep)} texts to WAIT on:")
    for text in texts_to_keep:
        print(f"   - {text}")
    print()
    
    # 4. Update buffer based on Builder's decision
    print("4️⃣ Updating Buffer (keeping deferred items)...")
    buffer.keep(texts_to_keep)
    print(f"   Buffer size after keep: {buffer.size()}")
    if not buffer.is_empty():
        print(f"   Remaining in buffer:\n{buffer.get_all_content()}")
    print()
    
    # 5. Check Graph state
    print("5️⃣ Graph State After First Round:")
    print(graph.get_full_state())
    print()
    
    # 6. Second round - resolve ambiguity
    print("6️⃣ Second Round - Resolving Ambiguity...")
    buffer.add("User: His name is Bob. He joined Google last year.")
    
    print(f"   Buffer size: {buffer.size()}")
    print(f"   Buffer content:\n{buffer.get_all_content()}")
    print()
    
    print("7️⃣ Builder Processing Buffer Again...")
    buffer_content = buffer.get_all_content()
    texts_to_keep = builder.process_buffer(buffer_content)
    buffer.keep(texts_to_keep)
    
    print(f"   Buffer size after second round: {buffer.size()}")
    print()
    
    # 8. Final Graph state
    print("8️⃣ Final Graph State:")
    print(graph.get_full_state())
    print()
    
    # 9. Test retrieval
    print("9️⃣ Testing Memory Retrieval...")
    print("   Searching for 'Alice':")
    print(graph.search("Alice"))
    print()
    print("   Searching for 'Google':")
    print(graph.search("Google"))
    print()
    
    print("✅ LightMem Integration Test Complete!\n")


def test_buffer_overflow_scenario():
    """
    Test buffer overflow behavior with max_size limit.
    """
    print("🧪 Testing Buffer Overflow Scenario...\n")
    
    buffer = ShortTermBuffer(max_size=3)
    
    print("Adding 5 items to a buffer with max_size=3...")
    for i in range(1, 6):
        buffer.add(f"Message {i}")
        print(f"   Added 'Message {i}' - Buffer size: {buffer.size()}")
    
    print(f"\nFinal buffer content:")
    print(buffer.get_all_content())
    print("\n✅ Buffer Overflow Test Complete!\n")


if __name__ == "__main__":
    # Run without API key requirement for basic buffer tests
    test_buffer_overflow_scenario()
    
    # Run integration test only if OpenAI API key is available
    if os.getenv("OPENAI_API_KEY"):
        test_lightmem_integration()
    else:
        print("⚠️  Skipping integration test (requires OPENAI_API_KEY)")
        print("   Set OPENAI_API_KEY to test full Builder+Buffer+Graph integration")
