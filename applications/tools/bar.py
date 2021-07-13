import time


class ProgressBar:
    def __init__(self, min_value=0, max_value=None, width=15, name=""):
        self._min, self._max = min_value, max_value
        self._task_length = int(max_value - min_value) if (
            min_value is not None and max_value is not None
        ) else None
        self._counter = min_value
        self._bar_width = int(width)
        self._bar_name = " " if not name else " # {:^12s} # ".format(name)
        self._terminated = False
        self._started = True
        self._ended = False


    def _flush(self):
        if self._ended:
            return False
        if not self._started:
            print("Progress bar not started yet.")
            return False
        
        if (not self._terminated) and self._counter>2:
            passed = int(self._counter * self._bar_width / self._max)
            pct = round(((self._counter)*100/(self._max)),1)
            print(
                "\r" + "##{}[".format(
                    self._bar_name
                ) + "-" * passed + " " * (self._bar_width - passed) + "] : {} %".format(
                    pct
                ) if self._counter != self._min else "##{}Progress bar initialized  ##".format(
                    self._bar_name
                ), end=""
            )            
            if self._counter >= self._max:
                self._terminated = True
                return self._flush()
            return True
        elif self._terminated:
            if self._counter == self._min:
                self._counter = self._min + 1
            print(
                "\r" +
                "##{}({:d} -> {:d}) Task Finished. ".format(
                    self._bar_name, self._min, self._counter - self._min
                ) + " ##\n", end=""
            )
            self._ended = True
            return True

    def update(self, new_value=None):
        if new_value is None:
            new_value = self._counter + 1
        if new_value != self._min:
            self._counter = self._max if new_value >= self._max else int(new_value)
            return self._flush()




if __name__ == '__main__':

    def task(cost=0.1, epoch=30, name="", _sub_task=None):
        def _sub():
            bar = ProgressBar(max_value=epoch, name=name)
            for _ in range(epoch):
                time.sleep(cost)
                bar.update()
        return _sub


    task(name="Task1")()