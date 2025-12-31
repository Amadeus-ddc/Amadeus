import os
from dotenv import load_dotenv
from amadeus.core.graph import MemoryGraph
from amadeus.core.buffer import ShortTermBuffer
from amadeus.agent.builder import BuilderAgent

def main():
    # 加载环境变量
    load_dotenv()
    
    # 1. 初始化 (with LightMem Buffer)
    print("Initializing Amadeus with LightMem...")
    graph = MemoryGraph("data/my_research_graph.json")
    buffer = ShortTermBuffer(max_size=10)
    builder = BuilderAgent(graph)
    print(f"Graph: {graph.graph.number_of_nodes()} nodes | Buffer: {buffer}")

    # 2. 模拟第一轮对话 (使用 Buffer 管理短期记忆)
    print("\n--- Round 1: Adding to Buffer ---")
    buffer.add("User: Hi, I'm Linus. I'm working on a new OS kernel called Linux.")
    buffer.add("AI: That sounds interesting.")
    buffer.add("User: It's just a hobby, won't be big and professional like GNU.")
    
    print(f"Buffer size: {buffer.size()}")
    print("\n--- Processing Buffer (Round 1) ---")
    buffer_content = buffer.get_all_content()
    texts_to_keep = builder.process_buffer(buffer_content)
    buffer.keep(texts_to_keep)
    print(f"Buffer after processing: {buffer.size()} items kept")
    
    # 3. 模拟第二轮对话 (增加新信息)
    print("\n--- Round 2: Adding More to Buffer ---")
    buffer.add("User: I uploaded Linux to an FTP server in Finland.")
    buffer.add("User: It's written in C and Assembly.")
    
    print(f"Buffer size: {buffer.size()}")
    print("\n--- Processing Buffer (Round 2) ---")
    buffer_content = buffer.get_all_content()
    texts_to_keep = builder.process_buffer(buffer_content)
    buffer.keep(texts_to_keep)
    print(f"Buffer after processing: {buffer.size()} items kept")

    # 4. 打印最终图谱状态
    print("\n--- Final Memory State (White-box Inspection) ---")
    print(graph.get_full_state())

    # 5. 测试检索
    print("\n--- Testing Retrieval ---")
    print(graph.search("Linux"))

if __name__ == "__main__":
    main()