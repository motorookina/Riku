# -*- coding: utf-8 -*-
"""将 stock_basic.csv 整理为按市场分组的 Markdown 文档。"""
import pandas as pd
from datetime import datetime

df = pd.read_csv("data/stock_basic.csv", dtype={"list_date": str})
df = df.sort_values(["market", "ts_code"]).reset_index(drop=True)

# 上市日期格式化为 YYYY-MM-DD（容忍异常值）
def fmt_date(d):
    d = str(d).strip()
    if d.isdigit() and len(d) == 8:
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return d

df["list_date"] = df["list_date"].map(fmt_date)

total = len(df)
market_order = ["主板", "创业板", "科创板", "北交所"]
market_counts = df["market"].value_counts().to_dict()

lines = []
lines.append("# A 股上市公司名单")
lines.append("")
lines.append(f"> 数据来源：Tushare `stock_basic` 接口（`list_status=L`，即当前上市状态）  ")
lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
lines.append(f"> 上市公司总数：**{total}** 只")
lines.append("")
lines.append("## 概览")
lines.append("")
lines.append("| 市场 | 数量 | 占比 |")
lines.append("| --- | ---: | ---: |")
for m in market_order:
    c = market_counts.get(m, 0)
    lines.append(f"| {m} | {c} | {c / total * 100:.1f}% |")
lines.append("")

# 每个市场一个章节
for m in market_order:
    sub = df[df["market"] == m]
    lines.append(f"## {m}（{len(sub)} 只）")
    lines.append("")
    lines.append("| 代码 | 名称 | 行业 | 地区 | 上市日期 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for _, r in sub.iterrows():
        lines.append(f"| {r['ts_code']} | {r['name']} | {r['industry']} | {r['area']} | {r['list_date']} |")
    lines.append("")

out = "A股上市公司名单.md"
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"已生成 {out}，共 {len(lines)} 行，{total} 只股票")
