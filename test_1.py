import math

class CosineAnnealingLR:
    def __init__(self, eta_min: float, eta_max: float, T_max: int):
        """
        初始化余弦退火调度器
        :param eta_min: 最小学习率
        :param eta_max: 最大学习率 (初始学习率)
        :param T_max: 总的退火周期步数
        """
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.T_max = T_max

    def get_lr(self, step: int) -> float:
        """
        根据当前步数计算学习率
        :param step: 当前已经进行的步数 (从 0 开始)
        :return: 当前步数对应的学习率
        """
        # 边界处理：当 step >= T_max 时，学习率保持为 eta_min
        if step >= self.T_max:
            return self.eta_min
        
        # 核心公式计算
        # η_t = η_min + 0.5 * (η_max - η_min) * (1 + cos(step / T_max * π))
        cosine_decay = 0.5*(1 + math.cos((step / self.T_max) * 3.14))
        print("#########cosine_decay",cosine_decay,0.5*cosine_decay, self.eta_min, self.eta_max)
        current_lr = self.eta_min + (self.eta_max - self.eta_min) * cosine_decay
        print("#########current_lr",current_lr)        
        return current_lr

if __name__ == "__main__":
    try:
        import sys
        
        # 默认演示数据
        eta_min_in = 0.05
        eta_max_in = 0.1
        T_max_in = 20
        print(f"未检测到输入，使用默认参数演示: eta_min={eta_min_in}, eta_max={eta_max_in}, T_max={T_max_in}")

        scheduler = CosineAnnealingLR(eta_min_in, eta_max_in, T_max_in)

        query_steps = [0, 5, 10, 15]
        
        results = []
        for s in query_steps:
            lr = scheduler.get_lr(s)
            results.append(lr)

        for res in results:
            print(f"{res:.2f}")
            
    except Exception as e:
        pass
