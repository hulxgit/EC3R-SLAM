import multiprocessing

class LoopIndicesManager:
    """
    管理 SLAM 回环检测中的 loop_indices + 记录每个 index 的分数
    - 只有 index < current_index-50 的才会被添加
    - 分数逻辑: map=+2, loop=+10, 重复=+1
    - 没出现的 index 分数-1，最小0，自动剔除
    - 只有 self.begin=True 才能往里面 add_batch，source=loop 时自动激活 self.begin
    """
    def __init__(self, manager):
        self.loop_indices = manager.list()  # ✅ 多进程共享 list（通过 set 转换使用）
        self.scores = manager.dict()        # ✅ 多进程共享 dict
        self.lock = manager.Lock()          # ✅ 共享锁
        self.begin = manager.Value('b', False)  # ✅ 共享 bool (True/False)

    def add_batch(self, indices, current_index, source="map"):
        with self.lock:
            if source == "loop":
                self.begin.value = True  # ✅ manager.Value 用 .value

            if not self.begin.value:
                # 如果未开始，直接跳过
                return []

            # 生成 ±30 范围
            merged_indices = set()
            for idx in indices:
                start = max(1, idx - 40)
                end = idx + 40
                merged_indices.update(range(start, end + 1))

            # 只允许小于 current_index - 50 的 index
            filtered_indices = {idx for idx in merged_indices if idx < (current_index - 50)}

            # ✅ list → set，合并，更新 list
            current_indices = set(self.loop_indices)
            current_indices.update(filtered_indices)
            self.loop_indices[:] = list(current_indices)

            appeared = set(filtered_indices)

            # 更新分数
            for idx in appeared:
                if idx not in self.scores:
                    self.scores[idx] = 5 if source == "map" else 20
                else:
                    self.scores[idx] += 1

            # 没出现的 index -1，自动剔除
            to_remove = []
            for idx in list(self.scores.keys()):
                if idx not in appeared:
                    self.scores[idx] = max(0, self.scores[idx] - 1)
                    if self.scores[idx] == 0:
                        to_remove.append(idx)

            # 删除 0 分数的 index
            for idx in to_remove:
                del self.scores[idx]
            self.loop_indices[:] = [idx for idx in self.loop_indices if idx not in to_remove]

            return sorted(filtered_indices)

    def get_all(self):
        with self.lock:
            return sorted(self.loop_indices)

    def get_scores(self):
        with self.lock:
            return dict(self.scores)

    def clear(self):
        with self.lock:
            self.loop_indices[:] = []
            self.scores.clear()
            self.begin.value = False  # ✅ clear 时，也重置开关

    def __len__(self):
        with self.lock:
            return len(self.loop_indices)

    def __str__(self):
        with self.lock:
            return (f"LoopIndicesManager(Indices: {sorted(self.loop_indices)}, "
                    f"Scores: {dict(self.scores)}, Begin: {self.begin.value})")
