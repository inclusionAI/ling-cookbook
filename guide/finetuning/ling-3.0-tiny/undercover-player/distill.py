"""专家蒸馏飞轮：离线调用大模型教师，对「谁是卧底」状态生成高质量动作标签。

与 flywheel.py（真值投票 + 自我模仿）互补：本脚本把**同款学生提示词**发给
Anthropic 兼容 Messages API 上的教师模型（如 mimo-v2.6-pro），拿回发言/投票
动作作为训练标签，产出 (state -> expert action) 蒸馏样本。

安全边界（务必保持）：
- 仅离线数据管线使用，**绝不进入游戏循环**——线上引擎身份是「Apple MLX 本地
  端侧模型引擎」，对局中的大脑必须是本机微调小模型；教师数据只影响下一轮
  训练，在 AI_USAGE.md 里如实披露。
- API 凭据只放 ~/.distill-api.env（chmod 600），脚本从环境变量读取，
  永不进 Git、日志里也不回显。

采样规则（v1，四条防标签冲突纪律）：
1. 真实对局（上帝视角）：
   - speak：每个实际发言座位 × 每轮，**跳过我们自己的座位**（那是 flywheel
     自我模仿的地盘——同一状态绝不允许出现两个不同标签）；
   - vote：**只做卧底座位**的投票状态（平民投票有上帝视角真值，flywheel 已
     覆盖；教师判断不该和真值抢标签）。卧底投票是策略模仿，真值不适用。
2. 合成对局：synthesize.make_game 同源生成器（seed 偏移，避免与 train 同流），
   每局取 1 个 speak 状态 + 1 个 vote 状态，教师动作替换规则式标签——直接
   攻击「规则投票标签噪声不可约」这个已证实的瓶颈。
3. 状态去重：所有状态按学生提示词哈希去重，撞上 train/valid/live 的已有状态
   直接跳过（从源头杜绝同输入异标签）。
4. 留出纪律：命中 test.jsonl 留出词对的整局跳过；合成状态只用训练词对。

标签质检（不达标丢弃，教师侧重试 1 次）：
- speak：clean_output 后 1..25 字、looks_natural、不含本人词语任何字；
- vote：取输出中最后一次「我投N号」/「我弃权」匹配，N 必须是在座的其他玩家。
- 教师侧 system 额外告知「你是几号玩家」（学生提示词不含座位号，教师不知
  道自己是几号就没法避免自投）；该信息只影响教师判断，不进训练样本。

用法：
    python distill.py --out data/api-distill-v1.jsonl       # 全量（约百余次调用）
    python distill.py --limit 3 --dry-run                   # 只看状态不调 API
    python distill.py --limit 4 --out /tmp/smoke.jsonl      # 冒烟
产出后照 flywheel.py 的 merge 流程按比例合并（默认 ≤ 原集 30%）+ 固定
test.jsonl 评测门槛，达标才收养。
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "data"))
from prompting import build_speak_prompt, build_vote_prompt, build_messages, clean_output  # noqa: E402
from flywheel import iter_games, load_holdout, looks_natural  # noqa: E402
from synthesize import make_game  # noqa: E402
from word_pairs import WORD_PAIRS, TEST_PAIR_INDICES  # noqa: E402

MAX_SPEAK_CHARS = 25
FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

TEACHER_SYSTEM = {
    "speak": "你是「谁是卧底」游戏的资深玩家。只输出你的发言本身，"
             "不超过25个字，不要说出自己词语里的任何一个字，"
             "不要输出任何解释、前缀或引号。",
    "vote": "你是「谁是卧底」游戏的资深玩家。可以先简短推理谁最可疑，"
            "但最后一行必须是且仅是「我投N号」或「我弃权」。"
            "注意：N 必须是在座的其他玩家，绝不能投给你自己。",
}


# ---------------------------------------------------------------- 状态构造

def history_through(game, up_to_round, precede_seat=None):
    """与 flywheel.py 同款的历史裁剪约定：
    vote 时刻 = 本轮发言全部结束之后（含本轮全部发言）；
    speak 时刻 = 轮到自己之前（只含本轮排在自己前面的发言）。
    """
    hist = []
    for rr in game["rounds"]:
        speeches = rr.get("speeches") or {}
        if not speeches or rr["round"] > up_to_round:
            continue
        order = rr.get("order") or sorted(int(k) for k in speeches)
        utts = []
        for k, v in sorted(speeches.items(), key=lambda kv: int(kv[0])):
            seat_i = int(k)
            if rr["round"] == up_to_round and precede_seat is not None:
                if seat_i == precede_seat:
                    continue
                if precede_seat in order and seat_i in order and \
                        order.index(seat_i) > order.index(precede_seat):
                    continue
            utts.append((seat_i, v["raw"] if isinstance(v, dict) else v))
        if utts:
            hist.append({"round": rr["round"], "utterances": utts})
    return hist


def real_states(game, me, holdout):
    """真实对局 -> 待标注状态（speak 全座位除自己；vote 仅卧底座位）。"""
    civilian = game["words"]["civilian"]
    undercover = game["words"]["undercover"]
    pair_key = f"{civilian}/{undercover}"
    if pair_key in holdout:
        return
    seats = {s["seat"]: s for s in game["seats"]}
    uc_seat = next((s["seat"] for s in game["seats"] if s["word"] == undercover), None)
    if uc_seat is None:
        return
    sid_short = game["session_id"].split(":")[-1][:8]
    n_players = len(game["seats"])

    for r in game["rounds"]:
        if r.get("pk"):
            continue
        speeches = r.get("speeches") or {}
        votes = r.get("votes") or {}

        # speak：实际发过言的座位（跳过我们自己）
        # 角色按词推导成训练口径的「平民/卧底」——对局状态里的 role 是英文
        for k in speeches:
            s = seats.get(int(k))
            if s is None or s["display"] == me:
                continue
            yield {
                "id": f"distill-speak-{pair_key}-{sid_short}-{r['round']}-s{s['seat']}",
                "task": "speak", "pair_key": pair_key, "seat": s["seat"],
                "role": "卧底" if s["word"] == undercover else "平民",
                "word": s["word"], "n_players": n_players,
                "rounds": history_through(game, r["round"], precede_seat=s["seat"]),
                "ctx": {"source": "live", "session": game["session_id"],
                        "round": r["round"]},
            }

        # vote：只做卧底座位（平民真值票归 flywheel）
        if votes:
            uc = seats.get(uc_seat)
            eliminated = uc.get("eliminated_round") if uc else None
            if uc and (eliminated is None or eliminated >= r["round"]):
                yield {
                    "id": f"distill-vote-{pair_key}-{sid_short}-{r['round']}-s{uc_seat}",
                    "task": "vote", "pair_key": pair_key, "seat": uc_seat,
                    "role": "卧底", "word": undercover, "n_players": n_players,
                    "rounds": history_through(game, r["round"]),
                    "ctx": {"source": "live", "session": game["session_id"],
                            "round": r["round"]},
                }


def synthetic_states(rng, n_games):
    """合成对局 -> 待标注状态（make_game 同源；seed 偏移避免与 train 同流）。"""
    train_idx = [i for i in range(len(WORD_PAIRS)) if i not in TEST_PAIR_INDICES]
    for gno in range(n_games):
        pair = WORD_PAIRS[rng.choice(train_idx)]
        pair_key = f"{pair['civilian']}/{pair['undercover']}"
        n_players = rng.randint(4, 6)
        my_seat = rng.randint(1, n_players)
        my_role = rng.choice(["平民", "卧底"])
        n_rounds = rng.randint(1, 3)
        game = make_game(rng, pair, n_players, my_seat, my_role, n_rounds)
        r = rng.randint(1, n_rounds)
        word = game["words"][my_seat]
        ctx = {"source": "synthetic", "session": f"syn-{gno}", "round": r}

        # speak 时刻的历史（与 synthesize.build_split 同款：前 r-1 轮全量 +
        # 第 r 轮在我之前发言的人）
        hist = [{"round": x["round"], "utterances": x["utterances"]}
                for x in game["rounds"][: r - 1]]
        hist.append({"round": r, "utterances": [(s, t) for s, t
                     in game["rounds"][r - 1]["utterances"] if s < my_seat]})
        yield {
            "id": f"distill-speak-{pair_key}-syn{gno}-{r}-s{my_seat}",
            "task": "speak", "pair_key": pair_key, "seat": my_seat,
            "role": my_role, "word": word, "n_players": n_players,
            "rounds": hist, "ctx": ctx,
        }
        yield {
            "id": f"distill-vote-{pair_key}-syn{gno}-{r}-s{my_seat}",
            "task": "vote", "pair_key": pair_key, "seat": my_seat,
            "role": my_role, "word": word, "n_players": n_players,
            "rounds": game["rounds"][:r], "ctx": ctx,
        }


def state_prompt(st):
    if st["task"] == "speak":
        return build_speak_prompt(st["role"], st["word"], st["rounds"])
    return build_vote_prompt(st["role"], st["word"], st["rounds"])


def prompt_key(st):
    return hashlib.md5(state_prompt(st).encode()).hexdigest()


# ---------------------------------------------------------------- 教师调用

class Teacher:
    def __init__(self, sleep=0.5):
        import anthropic  # 延迟导入：--dry-run 不需要 SDK
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=4, timeout=120.0)
        self.model = os.environ.get("ANTHROPIC_MODEL", "mimo-v2.6-pro")
        self.sleep = sleep

    def ask(self, task, prompt, seat=None, n_players=None):
        """返回教师原始文本；失败返回 None（调用方负责重试与质检）。

        seat/n_players 只进教师侧 system（学生提示词不含座位号，教师不知道
        「自己是几号」就没法避免自投/甩锅给自己）；训练消息仍按标准格式重建，
        座位信息不进样本。
        """
        system = TEACHER_SYSTEM[task]
        if seat is not None:
            system += f"\n补充信息：你是{seat}号玩家（全场{'' if n_players is None else n_players}人）。"
        for attempt in range(3):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
            except self._anthropic.RateLimitError:
                time.sleep(15 * (attempt + 1))
                continue
            except self._anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    time.sleep(5 * (attempt + 1))
                    continue
                print(f"  ! API {e.status_code}: {str(e)[:200]}", file=sys.stderr)
                return None
            except self._anthropic.APIConnectionError:
                time.sleep(5 * (attempt + 1))
                continue
            if resp.stop_reason == "max_tokens":
                return None
            text = "".join(b.text for b in resp.content
                           if getattr(b, "type", "") == "text").strip()
            if text:
                return text
        return None


# ---------------------------------------------------------------- 标签质检

def parse_speak(raw, word):
    text = clean_output(raw or "")
    candidates = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(candidates) > 1:
        candidates = [candidates[0], candidates[-1]]
    for cand in candidates:
        if not (1 <= len(cand) <= MAX_SPEAK_CHARS):
            continue
        if not looks_natural(cand):
            continue
        if any(ch in cand for ch in set(word)):  # 泄词：含本人词语任一字
            continue
        return cand
    return None


def parse_vote(raw, my_seat, n_players):
    ms = list(re.finditer(r"我投\s*([0-9０-９]+)\s*号|我弃权", raw or ""))
    if not ms:
        return None
    m = ms[-1]  # 教师允许先推理：取最后一次匹配
    if m.group(1) is None:
        return "我弃权"
    num = int(m.group(1).translate(FULLWIDTH_DIGITS))
    if not (1 <= num <= n_players) or num == my_seat:
        return None
    return f"我投{num}号"


# ---------------------------------------------------------------- 主流程

def load_env(path):
    path = Path(path).expanduser()
    if not path.exists():
        sys.exit(f"缺 {path}——应含 ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN / "
                 f"ANTHROPIC_MODEL（chmod 600，永不进 Git）")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
    if not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        sys.exit("env 文件里没有 ANTHROPIC_AUTH_TOKEN")


def load_seen(paths):
    """已有样本的学生提示词哈希集合（同输入绝不允许两个标签）。"""
    seen = set()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.open():
            try:
                rec = json.loads(line)
                msgs = rec.get("messages") or []
                if msgs and msgs[0].get("role") == "user":
                    seen.add(hashlib.md5(msgs[0]["content"].encode()).hexdigest())
            except Exception:
                continue
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(HERE / "data" / "api-distill-v1.jsonl"))
    ap.add_argument("--env-file", default="~/.distill-api.env")
    ap.add_argument("--state-dir", default=str(Path.home() / ".openclaw" / "undercover-game"))
    ap.add_argument("--train", default=str(HERE / "data" / "train.jsonl"))
    ap.add_argument("--valid", default=str(HERE / "data" / "valid.jsonl"))
    ap.add_argument("--test", default=str(HERE / "data" / "test.jsonl"))
    ap.add_argument("--live", default=str(HERE / "data" / "live-v1.jsonl"))
    ap.add_argument("--me", default="MLX小林", help="我们玩家的 display 名（真实对局 speak 跳过）")
    ap.add_argument("--synthetic-games", type=int, default=40)
    ap.add_argument("--per-game-cap", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="最多产出多少条（0=不限）")
    ap.add_argument("--sleep", type=float, default=0.5, help="每次调用间隔秒")
    ap.add_argument("--seed", type=int, default=142,
                    help="默认与 synthesize 的 42 错开，避免合成对局与 train 同流")
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只枚举状态，不调 API")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    holdout = load_holdout(args.test)
    seen = load_seen([args.train, args.valid, args.live])
    out_path = Path(args.out)

    ids, per_game, kept = set(), {}, 0
    if out_path.exists():
        for line in out_path.open():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ids.add(rec["id"])
            per_game[rec["meta"].get("session")] = \
                per_game.get(rec["meta"].get("session"), 0) + 1
            kept += 1
            if rec["messages"][0].get("role") == "user":
                seen.add(hashlib.md5(rec["messages"][0]["content"].encode()).hexdigest())
        print(f"续跑：已有 {kept} 条 -> {out_path}")

    states = []
    if not args.no_real:
        n_games = 0
        for g in iter_games(args.state_dir):
            n_games += 1
            states.extend(real_states(g, args.me, holdout))
        print(f"真实对局 {n_games} 局 -> {len(states)} 个待标注状态")
    if not args.no_synthetic:
        n0 = len(states)
        states.extend(synthetic_states(rng, args.synthetic_games))
        print(f"合成对局 {args.synthetic_games} 局 -> {len(states) - n0} 个待标注状态")

    if args.dry_run:
        for st in states[:20]:
            print(f"  [{st['id']}] {st['task']} role={st['role']} word={st['word']} "
                  f"rounds={len(st['rounds'])} pk={prompt_key(st)[:8]}")
        print(f"dry-run：共 {len(states)} 个状态（展示前 20），未调用 API")
        return

    load_env(args.env_file)
    teacher = Teacher(sleep=args.sleep)

    dropped = {"dup": 0, "cap": 0, "quality": 0}
    with out_path.open("a") as out:
        for i, st in enumerate(states):
            if args.limit and kept >= args.limit:
                break
            if st["id"] in ids:
                continue
            pk = prompt_key(st)
            if pk in seen:
                dropped["dup"] += 1
                continue
            sid = st["ctx"].get("session")
            if per_game.get(sid, 0) >= args.per_game_cap:
                dropped["cap"] += 1
                continue

            prompt = state_prompt(st)
            label = None
            for _attempt in range(2):  # API 失败或质检不过都允许再试一次
                raw = teacher.ask(st["task"], prompt,
                                  seat=st["seat"], n_players=st["n_players"])
                if raw is not None:
                    if st["task"] == "speak":
                        label = parse_speak(raw, st["word"])
                    else:
                        label = parse_vote(raw, st["seat"], st["n_players"])
                    if label:
                        break
                time.sleep(1)
            time.sleep(args.sleep)

            if not label:
                dropped["quality"] += 1
                continue

            msgs = build_messages(st["task"], st["role"], st["word"], st["rounds"], label)
            rec = {
                "id": st["id"], "task": st["task"], "pair_key": st["pair_key"],
                "role": st["role"], "word": st["word"], "messages": msgs,
                "meta": {"source": "api-distill", "teacher": teacher.model,
                         "label": "expert-strategy" if (
                             st["task"] == "vote" and st["role"] == "卧底")
                         else "expert-judgment" if st["task"] == "vote"
                         else "expert-imitation",
                         "session": sid, "round": st["ctx"].get("round"),
                         "seat": st["seat"], "weight": 1.0, "iter": "v1"},
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            ids.add(st["id"])
            seen.add(pk)
            per_game[sid] = per_game.get(sid, 0) + 1
            kept += 1
            if kept % 10 == 0:
                print(f"  进度：kept={kept} dropped={dropped}")

    print(f"完成：kept={kept} -> {out_path}（丢弃 {dropped}）")
    print("注意：先跑评测门槛再合并——见 README「数据飞轮」一节。")


if __name__ == "__main__":
    main()
