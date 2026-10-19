#!/usr/bin/env python3
"""
data/generate_data.py
=====================
程序化合成一个中文电商订单领域的 NL2SQL 数据集 (Ling-3.0-tiny 微调 Recipe 配套数据脚本)。

产物 (默认输出到本文件所在目录):
  1. database.sql   —— SQLite 可执行的建表 + 示例数据, 供评测脚本执行 SQL 计算「执行准确率 EX」;
  2. train.jsonl    —— SFT 训练集 (messages 三段式: system / user / assistant);
  3. val.jsonl      —— 验证集;
  4. test.jsonl     —— 留出测试集 (评测用, 训练不涉及)。

用法:
    python3 data/generate_data.py                     # 默认 600/100/200
    python3 data/generate_data.py --n-train 100 --n-val 20 --n-test 30   # 快速冒烟
    python3 data/generate_data.py --outdir /tmp/nl2sql

依赖: 仅 Python 标准库 (random / json / sqlite3 / pathlib / argparse)。
"""

import argparse
import json
import random
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# 领域常量
# ---------------------------------------------------------------------------
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都"]
LEVELS = ["普通", "金卡", "白金"]
CATEGORIES = ["手机", "电脑", "家电", "书籍", "服饰", "美妆"]
STATUSES = ["已完成", "待支付", "已取消", "配送中"]
METHODS = ["支付宝", "微信", "银行卡"]

SYSTEM_PROMPT = (
    "你是一个中文 NL2SQL 助手。根据用户的中文问题和下方数据库结构，生成一条可执行的 SQL 查询语句。"
    "只输出 SQL，不要输出任何解释或多余文字。"
)

SCHEMA = """数据库表结构如下：

CREATE TABLE customers (
  customer_id INTEGER PRIMARY KEY,  -- 客户ID
  name TEXT,                        -- 客户姓名
  city TEXT,                        -- 所在城市
  level TEXT,                       -- 客户等级(普通/金卡/白金)
  register_date TEXT                -- 注册日期 YYYY-MM-DD
);
CREATE TABLE products (
  product_id INTEGER PRIMARY KEY,   -- 商品ID
  name TEXT,                        -- 商品名称
  category TEXT,                    -- 商品类别(手机/电脑/家电/书籍/服饰/美妆)
  price REAL,                       -- 单价(元)
  stock INTEGER                     -- 库存数量
);
CREATE TABLE orders (
  order_id INTEGER PRIMARY KEY,     -- 订单ID
  customer_id INTEGER,              -- 下单客户ID
  order_date TEXT,                  -- 下单日期 YYYY-MM-DD
  status TEXT,                      -- 订单状态(已完成/待支付/已取消/配送中)
  total_amount REAL                 -- 订单总金额(元)
);
CREATE TABLE order_items (
  order_item_id INTEGER PRIMARY KEY,-- 订单明细ID
  order_id INTEGER,                 -- 所属订单ID
  product_id INTEGER,               -- 商品ID
  quantity INTEGER,                 -- 购买数量
  unit_price REAL                   -- 成交单价(元)
);
CREATE TABLE payments (
  payment_id INTEGER PRIMARY KEY,   -- 支付ID
  order_id INTEGER,                 -- 关联订单ID
  payment_date TEXT,                -- 支付日期 YYYY-MM-DD
  amount REAL,                      -- 支付金额(元)
  method TEXT                       -- 支付方式(支付宝/微信/银行卡)
);"""


# ---------------------------------------------------------------------------
# 数据生成
# ---------------------------------------------------------------------------
def build_database(rng: random.Random) -> dict:
    """构造领域相关的示例行数据, 返回 {表名: [行dict, ...]}."""
    customers = []
    for i in range(1, 13):
        customers.append({
            "customer_id": i,
            "name": f"客户{i:02d}",
            "city": rng.choice(CITIES),
            "level": rng.choice(LEVELS),
            "register_date": f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        })

    products = []
    for i in range(1, 21):
        category = rng.choice(CATEGORIES)
        products.append({
            "product_id": i,
            "name": f"{category}-{i:02d}",
            "category": category,
            "price": round(rng.uniform(20, 8000), 2),
            "stock": rng.randint(5, 500),
        })

    orders = []
    for i in range(1, 41):
        orders.append({
            "order_id": i,
            "customer_id": rng.randint(1, len(customers)),
            "order_date": f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "status": rng.choice(STATUSES),
            "total_amount": 0.0,
        })

    order_items = []
    item_id = 1
    for order in orders:
        n_items = rng.randint(1, 4)
        chosen = rng.sample(products, n_items)
        total = 0.0
        for p in chosen:
            qty = rng.randint(1, 5)
            unit = p["price"]
            order_items.append({
                "order_item_id": item_id,
                "order_id": order["order_id"],
                "product_id": p["product_id"],
                "quantity": qty,
                "unit_price": unit,
            })
            total += qty * unit
            item_id += 1
        order["total_amount"] = round(total, 2)

    payments = []
    pid = 1
    paid_orders = [o for o in orders if o["status"] in ("已完成", "配送中")]
    for order in paid_orders:
        payments.append({
            "payment_id": pid,
            "order_id": order["order_id"],
            "payment_date": order["order_date"],
            "amount": order["total_amount"],
            "method": rng.choice(METHODS),
        })
        pid += 1

    return {
        "customers": customers,
        "products": products,
        "orders": orders,
        "order_items": order_items,
        "payments": payments,
    }


def dump_sql(db: dict) -> str:
    """生成 SQLite 可执行的建表 + 插入语句。"""
    lines = []
    lines.append("DROP TABLE IF EXISTS payments;")
    lines.append("DROP TABLE IF EXISTS order_items;")
    lines.append("DROP TABLE IF EXISTS orders;")
    lines.append("DROP TABLE IF EXISTS products;")
    lines.append("DROP TABLE IF EXISTS customers;")
    lines.append(
        "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, "
        "city TEXT, level TEXT, register_date TEXT);"
    )
    lines.append(
        "CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT, "
        "category TEXT, price REAL, stock INTEGER);"
    )
    lines.append(
        "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, "
        "order_date TEXT, status TEXT, total_amount REAL);"
    )
    lines.append(
        "CREATE TABLE order_items (order_item_id INTEGER PRIMARY KEY, order_id INTEGER, "
        "product_id INTEGER, quantity INTEGER, unit_price REAL);"
    )
    lines.append(
        "CREATE TABLE payments (payment_id INTEGER PRIMARY KEY, order_id INTEGER, "
        "payment_date TEXT, amount REAL, method TEXT);"
    )

    def esc(v):
        if isinstance(v, str):
            return "'" + v.replace("'", "''") + "'"
        return str(v)

    for c in db["customers"]:
        lines.append(
            f"INSERT INTO customers VALUES ({c['customer_id']}, {esc(c['name'])}, "
            f"{esc(c['city'])}, {esc(c['level'])}, {esc(c['register_date'])});"
        )
    for p in db["products"]:
        lines.append(
            f"INSERT INTO products VALUES ({p['product_id']}, {esc(p['name'])}, "
            f"{esc(p['category'])}, {p['price']}, {p['stock']});"
        )
    for o in db["orders"]:
        lines.append(
            f"INSERT INTO orders VALUES ({o['order_id']}, {o['customer_id']}, "
            f"{esc(o['order_date'])}, {esc(o['status'])}, {o['total_amount']});"
        )
    for oi in db["order_items"]:
        lines.append(
            f"INSERT INTO order_items VALUES ({oi['order_item_id']}, {oi['order_id']}, "
            f"{oi['product_id']}, {oi['quantity']}, {oi['unit_price']});"
        )
    for pa in db["payments"]:
        lines.append(
            f"INSERT INTO payments VALUES ({pa['payment_id']}, {pa['order_id']}, "
            f"{esc(pa['payment_date'])}, {pa['amount']}, {esc(pa['method'])});"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 问答 (question, sql) 模板 —— 全量枚举参数组合, 保证去重后数量充足
# ---------------------------------------------------------------------------
def _qs(question: str, sql: str):
    return (question, sql)


def make_instances(db: dict):
    """基于已生成的数据, 枚举全部 (question, sql) 实例。"""
    customers = db["customers"]
    products = db["products"]
    orders = db["orders"]
    dates = sorted({o["order_date"] for o in orders})

    instances = []

    # 1. 单表 WHERE 过滤
    for c in customers:
        instances.append(_qs(
            f"查询所有来自{c['city']}的客户姓名和等级。",
            f"SELECT name, level FROM customers WHERE city = '{c['city']}'",
        ))

    # 2. 每城市客户数
    instances.append(_qs(
        "统计每个城市的客户数量。",
        "SELECT city, COUNT(*) AS cnt FROM customers GROUP BY city",
    ))

    # 3. 分类平均价格
    for cat in CATEGORIES:
        instances.append(_qs(
            f"查询{cat}类商品的平均价格。",
            f"SELECT AVG(price) FROM products WHERE category = '{cat}'",
        ))

    # 4. 低库存过滤 + 排序
    for cat in CATEGORIES:
        instances.append(_qs(
            f"查询库存低于100的{cat}类商品名称和库存。",
            f"SELECT name, stock FROM products WHERE category = '{cat}' AND stock < 100 ORDER BY stock ASC",
        ))

    # 5. 多表 JOIN
    for c in customers:
        instances.append(_qs(
            f"查询{c['name']}名下的所有订单号和下单日期。",
            f"SELECT o.order_id, o.order_date FROM customers c JOIN orders o "
            f"ON c.customer_id = o.customer_id WHERE c.name = '{c['name']}'",
        ))

    # 6. JOIN + SUM 按客户统计消费
    instances.append(_qs(
        "查询每个客户的总消费金额，按金额从高到低排序。",
        "SELECT c.name, SUM(o.total_amount) AS total FROM customers c "
        "LEFT JOIN orders o ON c.customer_id = o.customer_id "
        "GROUP BY c.name ORDER BY total DESC",
    ))

    # 7. 订单明细 JOIN 商品
    for o in orders:
        instances.append(_qs(
            f"查询订单{o['order_id']}包含的商品名称和购买数量。",
            f"SELECT p.name, oi.quantity FROM order_items oi "
            f"JOIN products p ON oi.product_id = p.product_id WHERE oi.order_id = {o['order_id']}",
        ))

    # 8. TOP-N 销量
    for top in (3, 4, 5):
        instances.append(_qs(
            f"查询销量最高的前{top}个商品名称及总销量。",
            f"SELECT p.name, SUM(oi.quantity) AS total_qty FROM order_items oi "
            f"JOIN products p ON oi.product_id = p.product_id "
            f"GROUP BY p.name ORDER BY total_qty DESC LIMIT {top}",
        ))

    # 9. 日期范围 (对去重日期做两两组合)
    for i in range(len(dates)):
        for j in range(i + 1, len(dates)):
            lo, hi = dates[i], dates[j]
            instances.append(_qs(
                f"查询{lo}到{hi}之间下单的订单数量。",
                f"SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '{lo}' AND '{hi}'",
            ))

    # 10. HAVING
    for n in (2, 3, 4):
        instances.append(_qs(
            f"查询下单次数超过{n}次的客户ID及其下单次数。",
            f"SELECT customer_id, COUNT(*) AS cnt FROM orders "
            f"GROUP BY customer_id HAVING COUNT(*) > {n}",
        ))

    # 11. 支付方式分组统计
    instances.append(_qs(
        "统计每种支付方式的支付笔数和总金额。",
        "SELECT method, COUNT(*) AS cnt, SUM(amount) AS total FROM payments GROUP BY method",
    ))

    # 12. 状态 + 金额双重过滤
    for status in STATUSES:
        for amt in (300, 500, 1000, 2000, 5000):
            instances.append(_qs(
                f"查询状态为{status}且金额大于{amt}元的订单号和金额。",
                f"SELECT order_id, total_amount FROM orders WHERE status = '{status}' AND total_amount > {amt}",
            ))

    # 13. 最高消费客户
    instances.append(_qs(
        "查询消费总额最高的客户姓名。",
        "SELECT c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id "
        "GROUP BY c.name ORDER BY SUM(o.total_amount) DESC LIMIT 1",
    ))

    # 14. 总销售额
    instances.append(_qs(
        "查询所有已完成订单的总销售额。",
        "SELECT SUM(total_amount) FROM orders WHERE status = '已完成'",
    ))

    # 15. 某客户支付方式
    for c in customers:
        instances.append(_qs(
            f"查询{c['name']}所有订单的支付方式。",
            f"SELECT DISTINCT p.method FROM payments p JOIN orders o ON p.order_id = o.order_id "
            f"JOIN customers c ON o.customer_id = c.customer_id WHERE c.name = '{c['name']}'",
        ))

    # 16. 分类平均库存
    instances.append(_qs(
        "查询每个类别的平均库存。",
        "SELECT category, AVG(stock) AS avg_stock FROM products GROUP BY category",
    ))

    # 17. 某类商品最高价
    for cat in CATEGORIES:
        instances.append(_qs(
            f"查询{cat}类商品中价格最高的商品名称和价格。",
            f"SELECT name, price FROM products WHERE category = '{cat}' ORDER BY price DESC LIMIT 1",
        ))

    # 18. 金额最高的订单
    instances.append(_qs(
        "查询金额最高的订单号、客户姓名和金额。",
        "SELECT o.order_id, c.name, o.total_amount FROM orders o "
        "JOIN customers c ON o.customer_id = c.customer_id "
        "ORDER BY o.total_amount DESC LIMIT 1",
    ))

    # 19. 分类商品数量
    for cat in CATEGORIES:
        instances.append(_qs(
            f"查询{cat}类商品的数量。",
            f"SELECT COUNT(*) FROM products WHERE category = '{cat}'",
        ))

    # 20. 商品总销量 (按商品)
    for p in products:
        instances.append(_qs(
            f"查询商品{p['name']}的总销量。",
            f"SELECT COALESCE(SUM(oi.quantity), 0) FROM order_items oi WHERE oi.product_id = {p['product_id']}",
        ))

    # 21. 状态订单数
    for status in STATUSES:
        instances.append(_qs(
            f"查询状态为{status}的订单数量。",
            f"SELECT COUNT(*) FROM orders WHERE status = '{status}'",
        ))

    # 22. 日期 + 状态组合过滤
    for status in STATUSES:
        for d in dates:
            instances.append(_qs(
                f"查询{d}之前状态为{status}的订单数量。",
                f"SELECT COUNT(*) FROM orders WHERE status = '{status}' AND order_date < '{d}'",
            ))

    # 23. 城市 + 等级客户
    for city in CITIES:
        for level in LEVELS:
            instances.append(_qs(
                f"查询来自{city}的{level}级客户姓名。",
                f"SELECT name FROM customers WHERE city = '{city}' AND level = '{level}'",
            ))

    return instances


# ---------------------------------------------------------------------------
# 采样与落盘
# ---------------------------------------------------------------------------
def generate_samples(db: dict, rng: random.Random, n: int):
    instances = make_instances(db)
    rng.shuffle(instances)
    samples = []
    seen = set()
    for q, sql in instances:
        if q in seen:
            continue
        seen.add(q)
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{SCHEMA}\n\n问题：{q}"},
                {"role": "assistant", "content": sql},
            ]
        })
        if len(samples) >= n:
            break
    return samples


def write_jsonl(path: Path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(__file__).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    db = build_database(rng)

    # 落库 SQL (供评测执行 EX)
    (outdir / "database.sql").write_text(dump_sql(db), encoding="utf-8")

    # 由于 require 较高 (train+val+test 独立且去重), 一次性采样并顺序切分
    total = args.n_train + args.n_val + args.n_test
    all_samples = generate_samples(db, rng, total)
    train = all_samples[: args.n_train]
    val = all_samples[args.n_train : args.n_train + args.n_val]
    test = all_samples[args.n_train + args.n_val :]

    write_jsonl(outdir / "train.jsonl", train)
    write_jsonl(outdir / "val.jsonl", val)
    write_jsonl(outdir / "test.jsonl", test)

    print(f"[data] database.sql -> {outdir / 'database.sql'}")
    print(f"[data] train={len(train)} val={len(val)} test={len(test)}")
    if len(all_samples) < total:
        print(f"[warn] 仅生成 {len(all_samples)} 条去重样本 (目标 {total}), 请增大数据规模或模板多样性")


if __name__ == "__main__":
    main()