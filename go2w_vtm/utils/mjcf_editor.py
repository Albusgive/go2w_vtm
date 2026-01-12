# utils/mjcf_editor.py

import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any

class MJCFEditor:
    def __init__(self, xml_path: str):
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        # 验证根标签确实是 mujoco
        if self.root.tag != "mujoco":
            raise ValueError(f"Root tag is '{self.root.tag}', expected 'mujoco'.")

    def add_sub_element(
        self,
        parent_tag: str,
        child_tag: str,
        attrib: Optional[Dict[str, Any]] = None,
        parent_index: int = 0,
        child_text: Optional[str] = None,
    ) -> ET.Element:
        """
        在指定父标签下添加子元素。
        特别支持 parent_tag="mujoco"（即根节点）。
        """
        if parent_tag == "mujoco":
            # 根节点就是 self.root，且只有一个
            if parent_index != 0:
                raise IndexError("Only one root <mujoco> element exists.")
            parent = self.root
        else:
            # 查找所有后代中匹配 parent_tag 的元素
            parents = self.root.findall(f".//{parent_tag}")
            if not parents:
                raise ValueError(f"No element with tag '{parent_tag}' found in MJCF.")
            if parent_index >= len(parents):
                raise IndexError(f"Parent index {parent_index} out of range for tag '{parent_tag}'.")
            parent = parents[parent_index]

        # 确保属性值是字符串
        str_attrib = {k: str(v) for k, v in attrib.items()} if attrib else {}
        child = ET.SubElement(parent, child_tag, str_attrib)
        if child_text is not None:
            child.text = child_text
        return child

    def to_string(self, pretty_print: bool = True) -> str:
        if pretty_print and hasattr(ET, 'indent'):
            ET.indent(self.root)
        elif pretty_print:
            self._indent(self.root)
        return ET.tostring(self.root, encoding="unicode")

    def _indent(self, elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def save(self, output_path: str):
        with open(output_path, "w") as f:
            f.write(self.to_string())