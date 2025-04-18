import re


def extract_answers(solution_str):
    # 正则表达式，匹配多个 <answer>...</answer> 标签
    answer_pattern = r'<answer>(.*?)</answer>'

    # 使用 re.DOTALL 以便匹配跨行内容
    match = re.finditer(answer_pattern, solution_str, flags=re.DOTALL)

    # 将所有匹配项放入一个列表
    matches = list(match)

    # 如果找到了匹配项，返回它们的内容
    if matches:
        final_answers = [match.group(1).strip() for match in matches]
    else:
        final_answers = None

    return final_answers


# 测试字符串
solution_str = """
<think>
First, let's break down the problem: We need to use the numbers [6, 79, 5] in any combination of +, -, *, / to equal 79. However, we have the constraint that one of the numbers must be a constant 5.
<answer>
6 * 5 = 30 + 79 - 5 = 65 + 79 - 5 = 79
</answer>
<answer>
Another possible solution could be 5 * 5 + 79 - 5 = 79.
</answer>
"""

# 提取答案
answers = extract_answers(solution_str)
print(answers)
