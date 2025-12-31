"""
Unit tests for the ShortTermBuffer class.
"""
import logging
from amadeus.core.buffer import ShortTermBuffer

# Set up logging for tests
logging.basicConfig(level=logging.INFO)


def test_buffer_initialization():
    """Test buffer initialization with default and custom max_size."""
    buffer1 = ShortTermBuffer()
    assert buffer1.max_size == 10
    assert buffer1.is_empty()
    assert buffer1.size() == 0
    
    buffer2 = ShortTermBuffer(max_size=5)
    assert buffer2.max_size == 5
    print("✅ Buffer initialization test passed")


def test_buffer_add():
    """Test adding content to buffer."""
    buffer = ShortTermBuffer(max_size=3)
    
    # Add normal content
    buffer.add("User: Hello, I'm Alice.")
    assert buffer.size() == 1
    assert not buffer.is_empty()
    
    # Add more content
    buffer.add("AI: Nice to meet you, Alice.")
    buffer.add("User: I work at Google.")
    assert buffer.size() == 3
    
    # Test empty content (should be skipped)
    buffer.add("")
    buffer.add("   ")
    assert buffer.size() == 3
    
    print("✅ Buffer add test passed")


def test_buffer_overflow():
    """Test FIFO overflow behavior."""
    buffer = ShortTermBuffer(max_size=3)
    
    buffer.add("Item 1")
    buffer.add("Item 2")
    buffer.add("Item 3")
    assert buffer.size() == 3
    
    # Adding 4th item should remove the first
    buffer.add("Item 4")
    assert buffer.size() == 3
    
    content = buffer.get_all_content()
    assert "Item 1" not in content
    assert "Item 2" in content
    assert "Item 3" in content
    assert "Item 4" in content
    
    print("✅ Buffer overflow test passed")


def test_buffer_get_all_content():
    """Test retrieving all content as text."""
    buffer = ShortTermBuffer()
    
    # Empty buffer
    assert buffer.get_all_content() == ""
    
    # Add items
    buffer.add("First message")
    buffer.add("Second message")
    buffer.add("Third message")
    
    content = buffer.get_all_content()
    assert "[1] First message" in content
    assert "[2] Second message" in content
    assert "[3] Third message" in content
    
    print("✅ Buffer get_all_content test passed")


def test_buffer_clear():
    """Test clearing the buffer."""
    buffer = ShortTermBuffer()
    
    buffer.add("Item 1")
    buffer.add("Item 2")
    assert buffer.size() == 2
    
    buffer.clear()
    assert buffer.size() == 0
    assert buffer.is_empty()
    assert buffer.get_all_content() == ""
    
    print("✅ Buffer clear test passed")


def test_buffer_keep():
    """Test keeping specific texts (simulating Builder's WAIT decision)."""
    buffer = ShortTermBuffer()
    
    buffer.add("User: Hello")
    buffer.add("AI: Hi there")
    buffer.add("User: He went to the store")  # Ambiguous pronoun - should be kept
    buffer.add("User: Alice bought some milk")
    
    assert buffer.size() == 4
    
    # Simulate Builder deciding to keep only the ambiguous text
    texts_to_keep = ["He went to the store"]
    buffer.keep(texts_to_keep)
    
    assert buffer.size() == 1
    content = buffer.get_all_content()
    assert "He went to the store" in content
    assert "Hello" not in content
    
    print("✅ Buffer keep test passed")


def test_buffer_keep_empty_list():
    """Test keep with empty list (should clear everything)."""
    buffer = ShortTermBuffer()
    
    buffer.add("Item 1")
    buffer.add("Item 2")
    assert buffer.size() == 2
    
    buffer.keep([])
    assert buffer.size() == 0
    assert buffer.is_empty()
    
    print("✅ Buffer keep with empty list test passed")


def test_buffer_with_metadata():
    """Test adding content with metadata."""
    buffer = ShortTermBuffer()
    
    buffer.add("Message 1", metadata={"source": "user"})
    buffer.add("Message 2", metadata={"source": "ai", "confidence": 0.9})
    
    assert buffer.size() == 2
    assert buffer.items[0]["metadata"]["source"] == "user"
    assert buffer.items[1]["metadata"]["confidence"] == 0.9
    
    print("✅ Buffer with metadata test passed")


def test_buffer_repr():
    """Test string representation."""
    buffer = ShortTermBuffer(max_size=5)
    buffer.add("Item 1")
    buffer.add("Item 2")
    
    repr_str = repr(buffer)
    assert "ShortTermBuffer" in repr_str
    assert "2/5" in repr_str
    
    print("✅ Buffer repr test passed")


def run_all_tests():
    """Run all buffer tests."""
    print("\n🧪 Running ShortTermBuffer Tests...\n")
    
    test_buffer_initialization()
    test_buffer_add()
    test_buffer_overflow()
    test_buffer_get_all_content()
    test_buffer_clear()
    test_buffer_keep()
    test_buffer_keep_empty_list()
    test_buffer_with_metadata()
    test_buffer_repr()
    
    print("\n✨ All buffer tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
