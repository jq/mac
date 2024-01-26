import random
import string

def generate_random_string(length=10):
    """生成一个指定长度的随机字符串。"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def collision(n, m):
    """生成n个随机字符串，计算它们的哈希值对m取模后的冲突率。"""
    # 生成随机字符串并计算它们的哈希值模m
    hash_mod_results = [hash(generate_random_string()) % m for _ in range(n)]

    # 计算冲突次数
    counts = {}
    for result in hash_mod_results:
        if result in counts:
            counts[result] += 1
        else:
            counts[result] = 1

    # 计算冲突的总数（非唯一值的出现次数）
    collisions = sum(count-1 for count in counts.values() if count > 1)

    # 计算冲突率
    collision_rate = collisions / n

    return collision_rate

# 示例用法
n = 100000  # 生成的随机字符串数量
m = 200000   # 模数
collision_rate = collision(n, m)
print(f"{n} {collision_rate:.2%}")