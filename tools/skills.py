"""skill相关工具"""
from langchain.tools import tool
from pathlib import Path
from typing import TypedDict

class SKILL(TypedDict):
    name: str # 每个skill的唯一标识
    description: str # 1-2句话描述skill的功能在系统提示词中
    content: str # 使用详细指令提供的全部的skill的内容 

# skill字典
_SKILLS: list[SKILL] | None = None
SKILLS_TTL=60

# 获取当前文件的相对路径,并根据相对路径生成skill路径,使用该路径来加载所有skill到内存中
def get_skills() -> list[SKILL]:
    """懒加载单例：只扫描一次 skills 目录并解析全部 SKILL.md。"""
    global _SKILLS
    if _SKILLS is None:
        result = []                              # 先构建到局部变量
        skills_dir = Path(__file__).parent / "skills"
        for sub_dir in skills_dir.iterdir():
            if sub_dir.is_dir():
                skill_file = sub_dir / "SKILL.md"
                if skill_file.exists():
                    full_content = skill_file.read_text(encoding="utf-8")
                    result.append(_parse_content(full_content))
        _SKILLS = result                          # 循环全部成功后才置位
    return _SKILLS


def _parse_content(text:str):
    """将完整的skill内容拆分成结构化字典并写入SKILLS列表中"""
    parts=text.split("---",2)
    import yaml
    try:
        meta=yaml.safe_load(parts[1])
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML metadata: {e}")
    content={'content': parts[2].strip()}
    return meta | content


@tool
def load_skill(skill_name: str) -> str:
    """
    加载skill
    Args:
        skill_name: 要加载的skill的名称
    """
    skills = get_skills()
    for skill in skills:
        if skill["name"] == skill_name:
            return f"已加载skill: {skill_name}\n\n{skill['content']}"
    available = ", ".join(s["name"] for s in skills)
    return f"未找到skill: {skill_name}. 可用的skill有: {available}"

if __name__ == "__main__":
    get_skills()
    print("meta:",_SKILLS)