"""用真实模型抽取过的虚构病例，检查长组切分是否保留分散的证据。"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d01.comparison import (StatusPairChoice, adjacent_report_options, comparison_jobs, prune_nonadjacent_pairs,
                            resolve_status_pairs)
from d01.contracts import Schema, StatusPair
from d01.discovery import Discovery, Findings
from d01.evaluation import Answers, candidate_coverage
from d01.graph import Graph
from d01.inputs import Sources
from d01.records import Locator, Record, RecordDraft
from d01.stages import json_size


def check_merged_pairs(reference: Answers, sources: Sources) -> None:
    """复用真实模型的三条状态，验证切窗后生成的对遇到中间报告会被移除。"""
    timeline = reference.timelines[0]
    early, middle, late = timeline.values
    pair, = timeline.statusPairs
    before = pair.priorStatus.model_copy(update={"ref": early.ref, "time": early.time, "value": early.value})
    after = pair.postStatus.model_copy(update={"ref": late.ref, "time": late.time, "value": late.value})
    across = StatusPair(priorStatus=before, postStatus=after)
    partial = timeline.model_copy(update={"values": [early, late], "statusPairs": [across]})
    options = adjacent_report_options([partial])
    selected, = resolve_status_pairs([StatusPairChoice(option="P1", prior_quote=str(before.value),
                                                       post_quote=str(after.value))], options, sources.texts)
    assert selected.model_dump(exclude={"timeline_index"}) == across.model_dump()
    partial.values = [early, middle, late]
    options = adjacent_report_options([partial])
    assert [(a[0], b[0]) for _, a, b in options.values()] == [(early.ref, middle.ref), (middle.ref, late.ref)]
    pair_option = next(key for key, (_, a, b) in options.items()
                       if (a[0], b[0]) == (pair.priorStatus.ref, pair.postStatus.ref))
    selected, = resolve_status_pairs([StatusPairChoice(option=pair_option, prior_quote=str(pair.priorStatus.value),
                                                       post_quote=str(pair.postStatus.value))], options, sources.texts)
    assert selected.model_dump(exclude={"timeline_index"}) == pair.model_dump()
    partial.statusPairs.append(pair)
    assert prune_nonadjacent_pairs(partial) == [across]
    assert partial.statusPairs == [pair] and partial.values == timeline.values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", required=True, type=Path)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--work-run", required=True, type=Path)
    args = parser.parse_args()

    reference = Answers.load(args.answers)
    reference.verify(args.in_dir)
    if len(reference.timelines) != 1 or len(reference.timelines[0].values) != 3:
        raise ValueError("需要含 3 个时间点的虚构连通性案例")
    early, middle, late = (value.ref for value in reference.timelines[0].values)
    original = Sources.load(args.in_dir, "task2")
    check_merged_pairs(reference, original)
    anchor, = [ref for ref in original.texts if ref not in {early, middle, late}]
    # 保留原文和模型记录，只把最近一次报告挪到中间，并收紧窗口预算。
    refs = [anchor, early, late, middle]
    sources = Sources("task2", {ref: original.texts[ref] for ref in refs})
    discovery = Discovery(Findings(pattern=None, signals=[], priority_refs=[]), refs, [refs])
    snapshot = json.loads((args.work_run / "schema.json").read_text(encoding="utf-8"))
    schema = Schema.model_validate(snapshot["model"])
    path = args.work_run / "task2" / f"records-{snapshot['version']}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result["records"] or any(not review["accepted"] for review in result["reviews"]):
        raise ValueError("需要一份逐条审核通过的模型运行记录")
    records = [Record(row["slot"], Locator.model_validate(row["locator"]), RecordDraft.model_validate(row["facts"]))
               for row in result["records"]]

    graph = Graph(records, schema)
    candidates = graph.candidates(sources, discovery, "indexed")
    short = comparison_jobs(candidates, sources, schema, 5000)
    assert len(short) == 1 and short[0].refs == refs
    limit = max(json_size(row) + 3 for row in sources.selected(refs))
    sequential = comparison_jobs(candidates, sources, schema, limit, use_tracks=False)
    tracked = comparison_jobs(candidates, sources, schema, limit)
    base_sets = [set(row.refs) for row in sequential]
    tracked_sets = [set(row.refs) for row in tracked]
    assert tracked_sets[:len(base_sets)] == base_sets
    assert len(tracked_sets) == len({frozenset(window) for window in tracked_sets})
    assert all(anchor in row.refs and len(row.patients) == 1 for row in tracked)
    for window in tracked:
        ids = set(window.refs)
        slots = {row.slot for row in window.records}
        for row in window.records:
            assert set(row.locator.refs) <= ids
            assert all(link.target_slot in slots for link in row.draft.connections)
            assert all(evidence.check(sources, ids)
                       for item in (*row.draft.facts, *row.draft.connections, *row.draft.proposed_fields)
                       for evidence in item.evidence)
    baseline = candidate_coverage(reference, {"task1": [], "task2": base_sets})["timeline_pairs"]
    improved = candidate_coverage(reference, {"task1": [], "task2": tracked_sets})["timeline_pairs"]
    assert (baseline["covered"], improved["covered"], improved["expected"]) == (1, 2, 2)
    print(json.dumps({"sequential": {"windows": len(sequential), **baseline},
                      "tracked": {"windows": len(tracked), **improved}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
