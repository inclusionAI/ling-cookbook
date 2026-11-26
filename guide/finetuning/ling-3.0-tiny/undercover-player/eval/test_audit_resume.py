"""CPU regression checks against the published, recorded model outputs."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'eval'), str(ROOT / 'data')]
from build_state_dataset import upgrade_record
from final_vote_audit import validate_result


class AuditResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [upgrade_record(json.loads(line))
                    for line in (ROOT / 'data/valid.jsonl').read_text().splitlines() if line.strip()]
        cls.recorded = json.loads((ROOT / 'eval/verification/v2-state-7.json').read_text())

    def validate(self, result):
        return validate_result(result, 'v2-state', 7, self.rows, True)

    def test_published_job_is_valid(self):
        self.assertIs(self.validate(self.recorded), self.recorded)

    def test_wrong_condition_or_seed_is_rejected(self):
        for key, value in [('condition', 'permuted'), ('seed', 42)]:
            changed = copy.deepcopy(self.recorded)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(changed)

    def test_missing_duplicate_and_reordered_records_are_rejected(self):
        for mutation in ['missing', 'duplicate', 'reordered']:
            changed = copy.deepcopy(self.recorded)
            if mutation == 'missing':
                changed['records'].pop()
            elif mutation == 'duplicate':
                changed['records'][1] = changed['records'][0]
            else:
                changed['records'].reverse()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate(changed)

    def test_scores_are_recomputed_from_raw(self):
        for field in ['raw', 'format_ok']:
            changed = copy.deepcopy(self.recorded)
            changed['records'][0][field] = '' if field == 'raw' else not changed['records'][0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(changed)

    def test_metrics_and_reversed_results_are_checked(self):
        for mutation in ['metrics', 'reversed_records', 'consistency']:
            changed = copy.deepcopy(self.recorded)
            if mutation == 'metrics':
                changed['metrics']['speech_format'] = -1
            else:
                changed[mutation].pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate(changed)


if __name__ == '__main__':
    unittest.main()
