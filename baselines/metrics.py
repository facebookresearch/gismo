class Metrics:
    def __init__(self):
        super(Metrics, self).__init__()
        self.mrr = 0.0
        self.hits = {1: 0, 3: 0, 10: 0}
        self.norm = 0.0

    def update(self, rank):
        if rank <= 1:
            self.hits[1] += 1
        if rank <= 3:
            self.hits[3] += 1
        if rank <= 10:
            self.hits[10] += 1
        self.mrr += 1.0 / rank
        self.norm += 1

    def normalize(self):
        self.mrr = self.mrr / self.norm * 100
        for key in self.hits:
            self.hits[key] = self.hits[key] / self.norm * 100
        return self.mrr, self.hits[1], self.hits[3], self.hits[10]
