"""CPU checks for the strict evaluation protocol."""
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'data'), str(ROOT / 'eval'), str(ROOT)]
from prompting import clean_output, build_vote_prompt, build_messages, render_player_state
from build_state_dataset import upgrade_record
from evaluate import eval_sample, sample_seed
from selfplay import parse_vote, run_game
from synthesize import make_game
from word_pairs import WORD_PAIRS


class ProtocolTests(unittest.TestCase):
    def sample(self, task='vote'):
        return {'task': task, 'word': '火锅',
                'meta': {'my_seat': 1, 'n_players': 6},
                'messages': [{'content': ''}, {'content': '我投2号'}]}

    def test_seed_changes_even_with_legacy_subset_metadata(self):
        row = {"meta": {"eval_seed": 59}}  # original row 17
        self.assertEqual([sample_seed(row, seed, 0) for seed in [7, 42, 123]],
                         [24, 59, 140])

    def test_paired_seeds_survive_subset_reordering(self):
        row = {"meta": {"eval_seed": 59}}
        self.assertEqual(sample_seed(row, 7, 0), sample_seed(row, 7, 12))
        self.assertEqual(sample_seed({}, 7, 17), sample_seed(row, 7, 0))

    def test_invalid_seed_metadata_is_rejected(self):
        for value in [True, "59", 59.0, 41]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                sample_seed({"meta": {"eval_seed": value}}, 42, 0)

    def test_reasoning_is_removed(self):
        self.assertEqual(clean_output('分析</think>我投2号<|role_end|>', True), '我投2号')

    def test_unfinished_reasoning_is_not_an_answer(self):
        self.assertEqual(clean_output('还在推理', True), '')

    def test_vote_must_be_entire_answer(self):
        for text in ['解释：我投2号', '我投2号\n因为可疑', '我弃权吧', '我投99号']:
            self.assertFalse(eval_sample(self.sample(), text)['format_ok'], text)

    def test_self_vote_is_invalid(self):
        self.assertFalse(eval_sample(self.sample(), '我投1号')['format_ok'])

    def test_dead_seat_is_invalid(self):
        self.assertEqual(parse_vote('我投3号', 1, {1, 2}), (None, False))

    def test_multiline_speech_is_invalid(self):
        self.assertFalse(eval_sample(self.sample('speak'), '热乎乎\n很好吃')['format_ok'])

    def test_game_roles_match_undercover_seat(self):
        for role in ['平民', '卧底']:
            game = make_game(random.Random(42), WORD_PAIRS[0], 6, 1, role, 2)
            self.assertEqual(game['roles'].count('卧底'), 1)
            self.assertEqual(game['roles'][game['uc_seat'] - 1], '卧底')

    def test_random_game_is_reproducible(self):
        a = run_game(None, 'random', 42, WORD_PAIRS[0])
        self.assertEqual(a, run_game(None, 'random', 42, WORD_PAIRS[0]))
        self.assertIn(a['winner'], ['civilian', 'undercover'])

    def test_state_fields_are_shared_by_messages_and_inference(self):
        kwargs = {'own_seat': 2, 'alive_seats': [4, 2, 1]}
        prompt = build_vote_prompt('平民', '火锅', [], **kwargs)
        self.assertEqual(prompt, build_messages('vote', '平民', '火锅', [], '我投4号', **kwargs)[0]['content'])
        self.assertIn('你的座位：2号', prompt)
        self.assertIn('存活玩家：1号、2号、4号', prompt)

    def test_partial_or_invalid_state_is_rejected(self):
        for own, alive in [(None, [1]), (1, None), (True, [1]), (1, [2]), (1, [1, 1])]:
            with self.assertRaises(ValueError):
                render_player_state(own, alive)

    def test_upgrade_preserves_label_and_source(self):
        row = self.sample()
        row.update(id='vote-synthetic-1', pair_key='火锅/烧烤')
        row['messages'][0]['content'] = build_vote_prompt('平民', '火锅', [])
        upgraded = upgrade_record(row)
        self.assertNotIn('alive_seats', row['meta'])
        self.assertEqual(row['messages'][1], upgraded['messages'][1])
        self.assertEqual(upgraded['meta']['alive_seats'], [1, 2, 3, 4, 5, 6])

    def test_state_evaluation_rejects_dead_vote(self):
        row = self.sample()
        row['meta']['alive_seats'] = [1, 3, 4]
        self.assertFalse(eval_sample(row, '我投2号')['format_ok'])

    def test_state_selfplay_updates_survivors(self):
        import re

        class FakeModel:
            def __init__(self):
                self.sizes = []

            def generate(self, prompt):
                own = int(re.search(r'你的座位：(\d+)号', prompt)[1])
                alive = [int(s) for s in re.findall(r'(\d+)号', re.search(r'存活玩家：(.*)', prompt)[1])]
                self.sizes.append(len(alive))
                if '现在开始投票' in prompt:
                    return f'我投{max(s for s in alive if s != own)}号'
                return '暖和的感觉'

        model = FakeModel()
        run_game(model, 'model', 1, WORD_PAIRS[0], prompt_protocol='state-v2')
        self.assertIn(6, model.sizes)
        self.assertTrue(any(n < 6 for n in model.sizes))


if __name__ == '__main__':
    unittest.main()
