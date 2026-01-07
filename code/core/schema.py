from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class OpType(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    WAIT = "WAIT"

class MemoryOperation(BaseModel):
    op: OpType = Field(..., description="操作类型")
    entity_name: str = Field(..., description="实体名称，如 'Alice'")
    entity_type: Optional[str] = Field(None, description="实体类型，如 'Person'")
    content: Optional[str] = Field(None, description="要存储或更新的关系/属性内容")
    target_entity: Optional[str] = Field(None, description="关系的目标对象（如果是边）")
    reason: str = Field(..., description="Builder 执行此操作的理由（对后续博弈和梯度更新至关重要）")

class BuilderOutput(BaseModel):
    operations: List[MemoryOperation]