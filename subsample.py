'''
Reads a jsonl file and randomly selects 100 sammples from that
'''
import json
import random

def main():
    with open('ComplexFuncBench/data/ComplexFuncBench.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    random.seed(42)
    random.shuffle(data)
    with open('ComplexFuncBench/data/ComplexFuncBench_subsampled.jsonl', 'w') as f:
        for d in data[:100]:
            f.write(json.dumps(d) + '\n')

def test_jsonl(file_path):
    try:
        data = [json.loads(line) for line in open(file_path, 'r')]
        assert len(data) == 100
        return True
    except:
        return False


if __name__ == '__main__':
    main()
    print(test_jsonl('ComplexFuncBench/data/ComplexFuncBench_subsampled.jsonl'))
