import json
from matplotlib import pyplot as plt
import collections

minimind = json.load(open('./evaluate/minimind/all_2025-08-05T11-26-51.631302.json', 'r'))['results']
qwen = json.load(open('./evaluate/qwen/all_2025-08-05T11-08-15.669325.json', 'r'))['results']


minimind_result = collections.defaultdict(list)
qwen_result = collections.defaultdict(list)

for key, value in minimind_result.items():
    if key.startswith('ceval_valid'):
        minimind_result['ceval_valid'].append(value)
    elif key.startswith('cmmlu'):
        minimind_result['cmmlu'].append(value)
    elif key.startswith('aclue'):
        minimind_result['aclue'].append(value)
    elif key.startswith('tmmluplus'):
        minimind_result['tmmluplus'].append(value)

for key, value in qwen_result.items():
    if key.startswith('ceval_valid'):
        qwen_result['ceval_valid'].append(value)
    elif key.startswith('cmmlu'):
        qwen_result['cmmlu'].append(value)
    elif key.startswith('aclue'):
        qwen_result['aclue'].append(value)
    elif key.startswith('tmmluplus'):
        qwen_result['tmmluplus'].append(value)

minimind_result = {key: sum(value) / len(value) for key, value in minimind_result.items()}
qwen_result = {key: sum(value) / len(value) for key, value in qwen_result.items()}

plt.bar(minimind_result.keys(), minimind_result.values())
plt.bar(qwen_result.keys(), qwen_result.values())
plt.show()

plt.savefig('result.png')