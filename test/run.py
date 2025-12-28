import os
from dotenv import load_dotenv
from amadeus.core.graph import MemoryGraph
from amadeus.agent.builder import BuilderAgent

def main():
    # 加载环境变量
    load_dotenv()
    
    # 1. 初始化
    print("Initializing Amadeus...")
    graph = MemoryGraph("data/my_research_graph.json")
    builder = BuilderAgent(graph)

    # 2. 模拟第一轮对话 (Buffer)
    buffer_text_1 = """
    User: Hi, I'm Linus. I'm working on a new OS kernel called Linux.
    AI: That sounds interesting.
    User: It's just a hobby, won't be big and professional like GNU.
    """
    
    print("\n--- Processing Batch 1 ---")
    builder.process_buffer(buffer_text_1)
    
    # 3. 模拟第二轮对话 (增加新信息)
    buffer_text_2 = """
    User: I uploaded Linux to an FTP server in Finland. 
    It's written in C and Assembly.
    """
    
    print("\n--- Processing Batch 2 ---")
    builder.process_buffer(buffer_text_2)

    # 4. 打印最终图谱状态
    print("\n--- Final Memory State (White-box Inspection) ---")
    print(graph.get_full_state())

    # 5. 测试检索
    print("\n--- Testing Retrieval ---")
    print(graph.search("Linux"))

if __name__ == "__main__":
    main()