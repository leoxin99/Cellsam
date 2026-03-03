import re

with open(r'd:\AI\paper\CellSam\docs\agent_inbox.md', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

msgs = []
current = None
for i, line in enumerate(lines):
    s = line.strip()
    m = re.match(r'^## \[(2026-02-\d\d \d\d:\d\d)\]\s*(.*)', s)
    if m:
        if current:
            msgs.append(current)
        current = {
            'date': m.group(1),
            'title': m.group(2)[:90],
            'line': i + 1,
            'status': 'no status found'
        }
    elif current and '**status**' in s.lower():
        sm = re.search(r'\*\*status\*\*:?\s*(.*)', s, re.IGNORECASE)
        if sm:
            current['status'] = sm.group(1).strip()[:60]

if current:
    msgs.append(current)

with open(r'd:\AI\paper\CellSam\docs\tmp_inbox_audit.txt', 'w', encoding='utf-8') as out:
    count = 0
    for msg in msgs:
        if msg['date'] >= '2026-02-16':
            count += 1
            out.write(f"{msg['date']} | {msg['status'][:45]:45s} | {msg['title'][:75]}\n")
    out.write(f"\nTotal: {count} messages from 02-16 onward\n")

print("Done")
