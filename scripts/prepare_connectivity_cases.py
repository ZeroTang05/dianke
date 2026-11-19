"""生成自编医疗格式案例，用真实网关验证流程；不读取任何验证集。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d01.inputs import write_json


CASES = {
    "task1": [
        ("DEMO_A_S1", "虚构案例。就诊记录：就诊编号V101，患者编号P101，患者姓名样例甲，证件号码DEMOID101，患者性别女，机构演示医院，初诊。"),
        ("DEMO_A_S2", "虚构案例。V101的检查记录编号R101，检查项目E010为右肾超声检查，检查类别影像，部位右肾。2030-02-11报告：右肾缺如，因既往右肾切除。"),
        ("DEMO_A_S3", "虚构案例。V101的同一份检查记录R101在复核版本中记载：右肾存在，形态大小正常。检查项目E010为右肾超声检查。2030-02-11报告。"),
        ("DEMO_B_S1", "虚构案例。就诊记录：就诊编号V109，患者编号P109，患者姓名样例乙，证件号码DEMOID109，患者性别男，机构演示医院，复诊。"),
        ("DEMO_B_S2", "虚构案例。V109的检查记录编号R109，检查项目E010为右肾超声检查，检查类别影像，部位右肾。2030-02-14报告：未见明显异常。"),
    ],
    "task2": [
        ("DEMO_A_S1", "虚构案例。就诊记录：就诊编号V201，患者编号P201，患者姓名样例丙，证件号码DEMOID201，患者性别女，机构演示医院，初诊。"),
        ("DEMO_A_S2", "虚构案例。V201的检查记录编号R201，检查项目编号E010，检查项目右肾超声检查，检查类别影像，部位右肾，2030-03-01的检查结果为右肾集合系统轻度扩张。"),
        ("DEMO_A_S3", "虚构案例。V201的复查检查记录编号R202，检查项目编号E010，检查项目右肾超声检查，检查类别影像，部位右肾，2030-03-08的检查结果为右肾集合系统未见扩张。对比初次检查，积水已消退。"),
        ("DEMO_A_S4", "虚构案例。V201的复查检查记录编号R203，检查项目编号E010，检查项目右肾超声检查，检查类别影像，部位右肾，2030-03-15的检查结果为右肾集合系统未见扩张。对比初次检查，积水已吸收。"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    for task, rows in CASES.items():
        write_json(args.out / task / "inputs.json", [{"id": ref, "content": content} for ref, content in rows])


if __name__ == "__main__":
    main()
