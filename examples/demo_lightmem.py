#!/usr/bin/env python3
"""
LightMem Demo: Demonstrates the lightweight memory buffer feature.

This script shows how the ShortTermBuffer works without requiring an OpenAI API key.
It demonstrates:
1. Adding content to the buffer
2. FIFO overflow behavior when max_size is reached
3. Selective keeping/clearing of buffer content
"""

import logging
from amadeus.core.buffer import ShortTermBuffer

# Enable colorful logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def demo_basic_operations():
    """Demo 1: Basic buffer operations."""
    print("=" * 60)
    print("DEMO 1: Basic Buffer Operations")
    print("=" * 60)
    
    buffer = ShortTermBuffer(max_size=5)
    
    print("\n1. Adding items to buffer...")
    buffer.add("User: Hello, I'm Alice.")
    buffer.add("AI: Hi Alice! Nice to meet you.")
    buffer.add("User: I work at Google.")
    
    print(f"\n2. Current buffer state (size: {buffer.size()}):")
    print(buffer.get_all_content())
    
    print("\n3. Adding more items...")
    buffer.add("AI: That's great! What do you do at Google?")
    buffer.add("User: I'm a software engineer working on search algorithms.")
    
    print(f"\n4. Buffer state after additions (size: {buffer.size()}):")
    print(buffer.get_all_content())
    
    print("\n✓ Demo 1 complete\n")


def demo_overflow_behavior():
    """Demo 2: FIFO overflow when buffer exceeds max_size."""
    print("=" * 60)
    print("DEMO 2: Buffer Overflow (FIFO)")
    print("=" * 60)
    
    buffer = ShortTermBuffer(max_size=3)
    
    print("\n1. Buffer with max_size=3, adding 5 items...")
    for i in range(1, 6):
        print(f"   Adding: Message {i}")
        buffer.add(f"Message {i}")
    
    print(f"\n2. Final buffer state (size: {buffer.size()}):")
    print(buffer.get_all_content())
    print("\n   ⚠️  Notice: Messages 1 and 2 were automatically removed (FIFO)")
    
    print("\n✓ Demo 2 complete\n")


def demo_selective_keeping():
    """Demo 3: Selective keeping (simulating Builder's WAIT decision)."""
    print("=" * 60)
    print("DEMO 3: Selective Keeping (WAIT Decision)")
    print("=" * 60)
    
    buffer = ShortTermBuffer(max_size=10)
    
    print("\n1. Simulating conversation with ambiguous information...")
    buffer.add("User: Alice works at Google.")
    buffer.add("User: She loves Python programming.")
    buffer.add("User: He is working on a new project.")  # Ambiguous: who is "he"?
    buffer.add("User: The project is about AI.")
    
    print(f"\n2. Current buffer (size: {buffer.size()}):")
    print(buffer.get_all_content())
    
    print("\n3. Builder decides to WAIT on ambiguous text...")
    print("   (Simulating: keeping only 'He is working on a new project')")
    
    texts_to_keep = ["He is working on a new project"]
    buffer.keep(texts_to_keep)
    
    print(f"\n4. Buffer after keep (size: {buffer.size()}):")
    if not buffer.is_empty():
        print(buffer.get_all_content())
    else:
        print("   (empty)")
    
    print("\n✓ Demo 3 complete\n")


def demo_clear_operation():
    """Demo 4: Clearing the buffer."""
    print("=" * 60)
    print("DEMO 4: Clearing the Buffer")
    print("=" * 60)
    
    buffer = ShortTermBuffer()
    
    print("\n1. Adding some items...")
    buffer.add("Item 1")
    buffer.add("Item 2")
    buffer.add("Item 3")
    
    print(f"\n2. Buffer before clear (size: {buffer.size()}):")
    print(buffer.get_all_content())
    
    print("\n3. Clearing buffer...")
    buffer.clear()
    
    print(f"\n4. Buffer after clear (size: {buffer.size()}):")
    print("   (empty)" if buffer.is_empty() else buffer.get_all_content())
    
    print("\n✓ Demo 4 complete\n")


def main():
    """Run all demos."""
    print("\n" + "🌟" * 30)
    print("   LightMem Feature Demo")
    print("   Lightweight Short-Term Memory Buffer")
    print("🌟" * 30 + "\n")
    
    demo_basic_operations()
    demo_overflow_behavior()
    demo_selective_keeping()
    demo_clear_operation()
    
    print("=" * 60)
    print("✨ All demos completed successfully!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   - Run 'python test/test_buffer.py' for unit tests")
    print("   - Run 'python test/test_lightmem_integration.py' for integration demo")
    print("   - Set OPENAI_API_KEY to test with Builder agent")
    print()


if __name__ == "__main__":
    main()
