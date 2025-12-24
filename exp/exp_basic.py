from models import TSMA, gpt4ts


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "TSMA": TSMA,
            "gpt4ts": gpt4ts,
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
