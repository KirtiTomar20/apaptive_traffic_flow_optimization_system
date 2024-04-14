
class TrafficMetricsStore:
    _instance = None

    @staticmethod
    def Global():
        if TrafficMetricsStore._instance is None:
            TrafficMetricsStore._instance = TrafficMetricsStore()
        return TrafficMetricsStore._instance

    def __init__(self):
        self.queueLength = None
        if TrafficMetricsStore._instance is not None:
            raise Exception("This class is a singleton!")

    def getQueueLength(self):
        return self.queueLength

    def setQueueLength(self, queueLength):
        self.queueLength = queueLength
